"""E2/E3: CNN-GRU point forecaster on temporally blocked dev folds.

E2 target: PV(t+15) in kW (trained on kW / 30).
E3 target: clear-sky index K(t+15) = PV / max(P_clear, min_clear); reconstructed to kW with the
           (known in advance) clear-sky PV at the target time. P_clear is fitted on the fold's
           TRAINING days only.
Both: identical inputs (16 chronological 64x64 frames + PV history / 30), identical model, optimiser and
budget; model selection on validation RMSE in kW.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.baselines import split_scores
from src.clear_sky import ClearSkyPV, clear_sky_index, from_clear_sky_index, smart_persistence
from src.models import Forecaster, PointHead, SmallSkyEncoder
from src.skippd import SkippdForecastDataset

PV_SCALE = 30.0  # kW, ~nameplate; input/target scaling only


class GpuSeq:
    """GPU-resident chronological stacks: images [B,16,3,64,64], pv history [B,16], target index."""

    def __init__(self, ds: SkippdForecastDataset, device):
        self.images = ds.images.to(device)
        self.pv = ds.pv.to(device)
        self.hist = torch.from_numpy(ds.hist_rows).to(device)
        self.tgt = torch.from_numpy(ds.target_rows).to(device)

    def __call__(self, idx):
        rows = self.hist[idx]
        x = self.images[rows].permute(0, 1, 4, 2, 3).float().div_(255.0)
        return x, self.pv[rows] / PV_SCALE


def build_model(cfg) -> torch.nn.Module:
    enc = SmallSkyEncoder(image_features=cfg["image_features"], hidden_size=cfg["hidden_size"])
    return Forecaster(enc, PointHead(cfg["hidden_size"], 1))


@torch.no_grad()
def predict_raw(model, data, idx, batch=1024):
    model.eval()
    return torch.cat([model(*data(idx[s:s + batch])).squeeze(-1) for s in range(0, len(idx), batch)])


def run_fold(ds, data, fold, target_kind, cfg, out_dir: Path, log) -> dict:
    device = data.pv.device
    day_str = ds.day.astype(str)
    train_days, val_days = set(fold["train_days"]), set(fold["val_days"])
    tr = np.flatnonzero(np.isin(day_str, list(train_days)))
    va = np.flatnonzero(np.isin(day_str, list(val_days)))

    # clear-sky anchor fitted on training-day records only
    rec_times = ds.record_times
    rec_days = rec_times.astype("datetime64[D]").astype(str)
    train_rec = np.isin(rec_days, list(train_days))
    cs = ClearSkyPV().fit(rec_times[train_rec], ds.pv.numpy()[train_rec])
    pclear_rec = cs(rec_times)
    pclear_tgt = pclear_rec[ds.target_rows]
    pclear_now = pclear_rec[ds.hist_rows[:, -1]]
    y_kw = ds.pv.numpy()[ds.target_rows].astype(float)
    if target_kind == "kw":
        y = y_kw / PV_SCALE
    else:
        y = clear_sky_index(y_kw, pclear_tgt, cfg["min_clear_kw"])
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    def to_kw(raw: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return raw * PV_SCALE if target_kind == "kw" else from_clear_sky_index(raw, pclear_tgt[idx], cfg["min_clear_kw"])

    torch.manual_seed(cfg["seed"])
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    tr_t, va_t = torch.as_tensor(tr, device=device), torch.as_tensor(va, device=device)

    best, wait, hist = float("inf"), 0, {"train_loss": [], "val_rmse_kw": []}
    for epoch in range(cfg["max_epochs"]):
        t0 = time.time()
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), generator=gen, device=device)]
        tot, n = 0.0, 0
        for s in range(0, len(perm), cfg["batch_size"]):
            b = perm[s:s + cfg["batch_size"]]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp"]):
                pred = model(*data(b)).squeeze(-1)
            loss = F.mse_loss(pred.float(), y_t[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            n += len(b)
        val_kw = to_kw(predict_raw(model, data, va_t).cpu().numpy().astype(float), va)
        val_rmse = float(np.sqrt(np.mean((val_kw - y_kw[va]) ** 2)))
        hist["train_loss"].append(tot / n)
        hist["val_rmse_kw"].append(val_rmse)
        improved = val_rmse < best
        if improved:
            best, wait = val_rmse, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            wait += 1
        log(f"  epoch {epoch + 1:3d}  train loss {tot / n:.5f}  val RMSE {val_rmse:.3f} kW{'  *' if improved else ''}"
            f"  ({time.time() - t0:.0f}s)")
        if wait >= cfg["patience"]:
            break

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    pred_kw = to_kw(predict_raw(model, data, va_t).cpu().numpy().astype(float), va)
    pv_now = ds.pv.numpy()[ds.hist_rows[va, -1]].astype(float)
    pers = smart_persistence(pv_now, pclear_now[va], pclear_tgt[va], cfg["min_clear_kw"])
    labels = np.array([fold["day_labels"][d] for d in day_str[va]])
    metrics = {
        "best_epoch": int(np.argmin(hist["val_rmse_kw"])) + 1, "epochs_run": len(hist["val_rmse_kw"]),
        "clear_sky_coef": cs.coef.tolist(),
        "model": split_scores(pred_kw, y_kw[va], labels),
        "smart_persistence": split_scores(pers, y_kw[va], labels),
    }
    metrics["skill_vs_smart_persistence"] = {
        g: 1 - metrics["model"][g]["rmse_kw"] / metrics["smart_persistence"][g]["rmse_kw"] for g in metrics["model"]}
    np.savez_compressed(out_dir / "preds_val.npz", sample_ids=ds.sample_ids[va], target=y_kw[va], pred=pred_kw,
                        persistence=pers, day=day_str[va], clear_day=labels)
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(a):
    cfg = {"seed": a.seed, "lr": 3e-4, "weight_decay": 1e-4, "batch_size": a.batch_size, "max_epochs": a.max_epochs,
           "patience": a.patience, "image_features": 64, "hidden_size": 64, "min_clear_kw": 1.0, "amp": not a.no_amp}
    out_root = Path(a.out) / a.target / f"seed{a.seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_f = open(out_root / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    folds_doc = json.loads(Path(a.folds).read_text())
    import subprocess

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip())
    (out_root / "config.json").write_text(json.dumps({**cfg, "git_commit": commit, "src_dirty": dirty,
                                                      "target": a.target, "manifest": a.manifest,
                                                      "folds": a.folds, "folds_sha256": folds_doc["content_sha256"]}, indent=2))
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ds = SkippdForecastDataset(a.h5, a.manifest, "dev", preload=True)
    m = np.load(a.manifest)
    ds.record_times = m["times_us"].astype("datetime64[us]")[m["hdf5_group"] == 0][np.argsort(m["hdf5_row"][m["hdf5_group"] == 0])]
    if a.subset_days:
        for fd in folds_doc["folds"]:
            fd["train_days"], fd["val_days"] = fd["train_days"][:a.subset_days], fd["val_days"][:a.subset_days]
    data = GpuSeq(ds, device)
    summary = {}
    for k in a.fold_ids:
        fold = {**folds_doc["folds"][k], "day_labels": folds_doc["day_labels"]}
        fold_dir = out_root / f"fold_{k}"
        fold_dir.mkdir(exist_ok=True)
        log(f"[{a.target}] fold {k}: val {fold['val_range'] if 'val_range' in fold else ''}")
        summary[k] = run_fold(ds, data, fold, a.target, cfg, fold_dir, log)
        s = summary[k]
        log(f"[{a.target}] fold {k} done: best epoch {s['best_epoch']}  RMSE "
            + "  ".join(f"{g} {s['model'][g]['rmse_kw']:.3f} (pers {s['smart_persistence'][g]['rmse_kw']:.3f})" for g in s["model"]))
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    log_f.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="E2 (kW) / E3 (clear-sky index) CNN-GRU on blocked folds")
    p.add_argument("--target", choices=["kw", "csi"], required=True)
    p.add_argument("--fold-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--subset-days", type=int, default=0, help="debug: limit train/val days per fold")
    p.add_argument("--h5", default="data/raw/2017_2019_images_pv_processed.hdf5")
    p.add_argument("--manifest", default="data/manifests/forecast_tol30_v2.npz")
    p.add_argument("--folds", default="data/manifests/folds_blocked5_v1.json")
    p.add_argument("--out", default="experiments/E2E3_cnn_gru/v1")
    main(p.parse_args())
