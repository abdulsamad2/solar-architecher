"""E1: faithful PyTorch port of SUNSET_forecast.ipynb (TensorFlow 2.4 original).

Matched to the original:
  * input: 16 frames newest-first stacked as 48 channels (channel = frame*3 + colour), /255;
    PV history: 16 raw kW values, newest first
  * Conv(24,3x3,same)->ReLU->BN->MaxPool(2,2) ; Conv(48)->ReLU->BN->MaxPool ; flatten ; concat PV ;
    Dense1024-ReLU-Dropout0.4 x2 ; Dense1
  * Keras defaults: glorot_uniform weights, zero bias, BN momentum 0.99 / eps 1e-3, Adam eps 1e-7
  * Adam lr 3e-6, MSE, batch 256, max 200 epochs, early stopping patience 5 on val loss, best checkpoint
  * day_block_shuffle (np.random.seed(1)) and cv_split (contiguous sample slices) copied verbatim,
    including the quirk that a day can straddle two folds
  * test prediction = mean of the 10 fold models

Deviations (documented in reports/protocol_notes.md):
  * one epoch == one full shuffled pass (Keras used steps_per_epoch = n//256 + 1 on a repeated dataset)
  * validation loss is the exact MSE over the validation fold (Keras evaluated n//256+1 batches of 500
    from a repeated, shuffled stream, i.e. ~2x the fold with some repeats)
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baselines import split_scores
from src.skippd import SkippdForecastDataset


class SunsetNet(nn.Module):
    def __init__(self, n_frames: int = 16, img_side: int = 64, num_filters: int = 24, dense: int = 1024, drop: float = 0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(3 * n_frames, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters, momentum=0.01, eps=1e-3)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters * 2, momentum=0.01, eps=1e-3)
        flat = num_filters * 2 * (img_side // 4) ** 2
        self.fc1 = nn.Linear(flat + n_frames, dense)
        self.fc2 = nn.Linear(dense, dense)
        self.out = nn.Linear(dense, 1)
        self.drop = nn.Dropout(drop)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, pv: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(self.bn1(F.relu(self.conv1(x))), 2, 2)
        x = F.max_pool2d(self.bn2(F.relu(self.conv2(x))), 2, 2)
        x = torch.cat([x.flatten(1), pv], dim=1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        return self.out(x).squeeze(-1)


def day_block_shuffle(dates: np.ndarray) -> np.ndarray:
    """Verbatim logic of the official day_block_shuffle (np.random.seed(1))."""
    unique_dates = np.unique(dates)
    blocks = [np.where(dates == d)[0] for d in unique_dates]
    np.random.seed(1)
    np.random.shuffle(blocks)
    return np.asarray(list(itertools.chain.from_iterable(blocks)))


def cv_split(split_data: np.ndarray, fold_index: int, num_fold: int):
    """Verbatim logic of the official cv_split."""
    num_samples = len(split_data)
    indices = np.arange(num_samples)
    val_mask = np.zeros(len(indices), dtype=bool)
    val_mask[int(fold_index / num_fold * num_samples):int((fold_index + 1) / num_fold * num_samples)] = True
    val_indices = indices[val_mask]
    train_indices = indices[np.logical_not(val_mask)]
    np.random.seed(fold_index)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    return split_data[train_indices], split_data[val_indices]


class GpuBatches:
    """Keeps the image/PV arrays on the GPU and gathers forecast stacks by index."""

    def __init__(self, ds: SkippdForecastDataset, device: torch.device):
        self.images = ds.images.to(device)  # [R, 64, 64, 3] uint8
        self.pv = ds.pv.to(device)
        self.hist = torch.from_numpy(ds.hist_rows).to(device)  # newest first
        self.tgt = torch.from_numpy(ds.target_rows).to(device)

    def __call__(self, idx: torch.Tensor):
        rows = self.hist[idx]
        x = self.images[rows]  # [B, 16, 64, 64, 3]
        b, t, h, w, c = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w).float().div_(255.0)
        return x, self.pv[rows], self.pv[self.tgt[idx]]


@torch.no_grad()
def predict(model: nn.Module, data: GpuBatches, idx: torch.Tensor, batch: int = 2000) -> torch.Tensor:
    model.eval()
    return torch.cat([model(*data(idx[s:s + batch])[:2]) for s in range(0, len(idx), batch)])


def train_fold(data, train_idx, val_idx, out_dir: Path, seed: int, cfg: dict, log) -> dict:
    torch.manual_seed(seed)
    model = SunsetNet().to(train_idx.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], eps=1e-7)
    gen = torch.Generator(device=train_idx.device).manual_seed(seed)
    _, _, y_val = data(val_idx)
    best, wait, hist = float("inf"), 0, {"train_mse": [], "val_mse": []}
    for epoch in range(cfg["max_epochs"]):
        t0 = time.time()
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), generator=gen, device=train_idx.device)]
        total, n = 0.0, 0
        for s in range(0, len(perm), cfg["batch_size"]):
            x, pv, y = data(perm[s:s + cfg["batch_size"]])
            loss = F.mse_loss(model(x, pv), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        val = F.mse_loss(predict(model, data, val_idx), y_val).item()
        hist["train_mse"].append(total / n)
        hist["val_mse"].append(val)
        improved = val < best
        if improved:
            best, wait = val, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            wait += 1
        log(f"  epoch {epoch + 1:3d}  train RMSE {np.sqrt(total / n):.3f}  val RMSE {np.sqrt(val):.3f}"
            f"{'  *' if improved else ''}  ({time.time() - t0:.1f}s)")
        if wait >= cfg["patience"]:
            break
    best_epoch = int(np.argmin(hist["val_mse"]))
    summary = {"best_epoch": best_epoch + 1, "epochs_run": len(hist["val_mse"]),
               "best_val_rmse": float(np.sqrt(hist["val_mse"][best_epoch])),
               "train_rmse_at_best": float(np.sqrt(hist["train_mse"][best_epoch])), **hist}
    (out_dir / "history.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(h5: str, manifest: str, out: str, num_fold: int, cfg: dict, persistence_preds: str | None, subset: int | None):
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train.log", "a", encoding="utf-8")

    def log(msg: str):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (out_dir / "config.json").write_text(json.dumps({"h5": h5, "manifest": manifest, "num_fold": num_fold,
                                                     "subset": subset, **cfg}, indent=2))
    dev = SkippdForecastDataset(h5, manifest, "dev", preload=True, newest_first=True)
    test = SkippdForecastDataset(h5, manifest, "test", preload=True, newest_first=True)
    dev_data, test_data = GpuBatches(dev, device), GpuBatches(test, device)
    log(f"device {device}; dev samples {len(dev):,}; test samples {len(test):,}")

    shuffled = day_block_shuffle(dev.day)
    if subset:
        shuffled = shuffled[:subset]
    test_idx = torch.arange(len(test), device=device)
    fold_preds, fold_summaries = [], []
    for k in range(num_fold):
        fold_dir = out_dir / f"fold_{k + 1}"
        fold_dir.mkdir(exist_ok=True)
        tr, va = cv_split(shuffled, k, num_fold)
        if (fold_dir / "history.json").exists() and (fold_dir / "best.pt").exists():
            log(f"fold {k + 1}/{num_fold}: already trained, reusing checkpoint")
            summary = json.loads((fold_dir / "history.json").read_text())
        else:
            log(f"fold {k + 1}/{num_fold}: train {len(tr):,}  val {len(va):,}")
            summary = train_fold(dev_data, torch.as_tensor(tr, device=device), torch.as_tensor(va, device=device),
                                 fold_dir, seed=k, cfg=cfg, log=log)
        log(f"fold {k + 1}: best epoch {summary['best_epoch']}  val RMSE {summary['best_val_rmse']:.3f}")
        fold_summaries.append(summary)
        model = SunsetNet().to(device)
        model.load_state_dict(torch.load(fold_dir / "best.pt", map_location=device))
        fold_preds.append(predict(model, test_data, test_idx).cpu().numpy())

    preds = np.stack(fold_preds)
    ensemble = preds.mean(0)
    target = test.pv[test.target_rows].numpy().astype(float)
    results = {"test_ensemble": split_scores(ensemble, target, test.is_sunny),
               "mean_val_rmse": float(np.mean([s["best_val_rmse"] for s in fold_summaries])),
               "per_fold_test_rmse": [float(np.sqrt(np.mean((p - target) ** 2))) for p in preds]}
    if persistence_preds:
        per = np.load(persistence_preds)
        assert np.array_equal(per["sample_ids"], test.sample_ids), "persistence preds misaligned"
        ref = split_scores(per["official_csi_persistence"], target, test.is_sunny)
        results["forecast_skill_vs_csi_persistence"] = {
            g: 1 - results["test_ensemble"][g]["rmse_kw"] / ref[g]["rmse_kw"] for g in ref}
    np.savez_compressed(out_dir / "preds_test.npz", sample_ids=test.sample_ids, target=target,
                        fold_preds=preds, ensemble=ensemble, day=test.day.astype(str), is_sunny=test.is_sunny)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    log(json.dumps(results, indent=2))
    log_f.close()
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="E1: SUNSET PyTorch port, 10-fold day-block CV + test ensemble")
    p.add_argument("--h5", default="data/raw/2017_2019_images_pv_processed.hdf5")
    p.add_argument("--manifest", default="data/manifests/forecast_tol30_v1.npz")
    p.add_argument("--out", default="experiments/E1_sunset_port/tol30_v1")
    p.add_argument("--persistence", default="experiments/E0_persistence/tol30_v1/preds_test.npz")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--subset", type=int, default=None, help="debug: use only the first N shuffled dev samples")
    a = p.parse_args()
    cfg = {"lr": 3e-6, "batch_size": 256, "max_epochs": a.max_epochs, "patience": 5}
    main(a.h5, a.manifest, a.out, a.folds, cfg, a.persistence, a.subset)
