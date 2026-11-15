"""Revision R-H0 (pilot): the direct expert with added image information at two lead times
(protocol revision/protocols/R-H0_resolution_pilot.md, amendment 2). Reuses src.train.run_fold unchanged (same
target, quantile head, pinball loss, optimiser, EMA, early stopping, folds and calibration split as E4-B-fit).

BASE: the E4-B-fit network on all development samples (needed at 5 min only; at 15 min the frozen E4-B-fit seed 42
checkpoints are the base). Every other arm is the base network plus a second image branch, started as exactly the
base forecaster (base weights loaded, recurrent input weights of the new features zero) and trained on the pilot
samples. Outputs go outside the repository (default D:/skippd_hires/runs2).

Arms: BASE | FULL-I64, FULL-I128, FULL-I256 | SUN-ctl, SUN-hi | RING-ctl, RING-hi
"""
from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.train as T
from src.models import Forecaster, NonCrossingQuantileHead, SmallSkyEncoder
from src.skippd import SkippdForecastDataset

ARMS = ("BASE", "FULL-I64", "FULL-I128", "FULL-I256", "SUN-ctl", "SUN-hi", "RING-ctl", "RING-hi")
CFG = {"lr": 1e-4, "weight_decay": 1e-4, "batch_size": 128, "max_epochs": 15, "patience": 5, "image_features": 64, "hidden_size": 64,
       "min_clear_kw": 1.0, "cnn_bf16": True, "schedule": "cosine", "clip": 1.0, "ema_decay": 0.999, "pool": "avg", "input": "csi",
       "head": "quantile", "calib_holdout": True, "residual": False}
ARM_LR = 3e-5  # arms start from the trained base network
FROZEN15 = "experiments/E4_calib/v1/csi/seed42"  # base network at 15 min
INIT = None  # base checkpoint of the fold being trained (set in main before run_fold)


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2))


def full_cnn():
    """E4-B-fit frame CNN behind a stem that reduces 256 x 256 frames to 64 x 64 (stride-2 conv, then conv + pool)."""
    stem = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.BatchNorm2d(8), nn.ReLU(inplace=True), _block(8, 8))
    return nn.Sequential(stem, *SmallSkyEncoder(in_channels=8).cnn)


class DualEncoder(nn.Module):
    """E4-B-fit frame CNN on the benchmark frames plus a second CNN on an extra image tensor."""

    def __init__(self, full=False):
        super().__init__()
        base = SmallSkyEncoder()
        self.cnn = base.cnn
        self.cnn2 = full_cnn() if full else SmallSkyEncoder().cnn
        self.gru = nn.GRU(64 + 64 + 1, 64, batch_first=True)
        self.hidden_size, self.cnn_bf16 = 64, False

    def forward(self, images, pv_history, extra):
        b, t = images.shape[:2]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cnn_bf16 and images.is_cuda):
            f1, f2 = self.cnn(images.flatten(0, 1)), self.cnn2(extra.flatten(0, 1))
        seq = torch.cat([f1.float().view(b, t, -1), f2.float().view(b, t, -1), pv_history.unsqueeze(-1)], -1)
        self.gru.flatten_parameters()
        return self.gru(seq)[1][-1]


class PilotData:
    """The interface run_fold expects from GpuSeq, with the arm's image tensors."""

    def __init__(self, ds, arm, hires, device):
        self.arm, self.pv = arm, ds.pv.to(device)
        self.hist = torch.from_numpy(ds.hist_rows).to(device)
        self.hist_feat = None
        rowmap = torch.full((len(ds.pv),), -1, dtype=torch.long)
        rowmap[torch.from_numpy(hires["row"][:].astype(np.int64))] = torch.arange(len(hires["row"]))
        self.rowmap = rowmap.to(device)
        fam = arm.split("-")[0]
        self.images = ds.images.to(device)
        if fam != "BASE":
            key = {"FULL": "full256", "SUN-ctl": "sun_ctl", "SUN-hi": "sun_hi", "RING-ctl": "ring_ctl", "RING-hi": "ring_hi"}[arm if fam != "FULL" else "FULL"]
            n = hires[key].shape[0]
            self.table = torch.empty((n, *hires[key].shape[1:]), dtype=torch.uint8, device=device)
            for s in range(0, n, 4096):
                self.table[s:s + 4096] = torch.from_numpy(hires[key][s:s + 4096]).to(device)

    def set_history_feature(self, per_record):
        self.hist_feat = torch.as_tensor(per_record, dtype=torch.float32, device=self.pv.device)

    def _extra(self, rows):
        x = self.table[self.rowmap[rows]].permute(0, 1, 4, 2, 3).float().div_(255.0)
        level = {"FULL-I64": 4, "FULL-I128": 2}.get(self.arm)
        if level:  # information level: area-average, then bilinear upsampling back to the tensor size
            b, t = x.shape[:2]
            x = F.interpolate(F.avg_pool2d(x.flatten(0, 1), level), scale_factor=level, mode="bilinear", align_corners=False).view(b, t, *x.shape[2:])
        return x

    def __call__(self, idx):
        rows = self.hist[idx]
        x = self.images[rows].permute(0, 1, 4, 2, 3).float().div_(255.0)
        if self.arm == "BASE":
            return x, self.hist_feat[rows]
        return x, self.hist_feat[rows], self._extra(rows)


def build_model(arm, cfg):
    enc = SmallSkyEncoder() if arm == "BASE" else DualEncoder(full=arm.startswith("FULL"))
    model = Forecaster(enc, NonCrossingQuantileHead(enc.hidden_size, 1, T.QUANTILES))
    if arm != "BASE" and INIT is not None:
        warm_start(model, INIT)
    return model


def warm_start(model, path):
    """Load the base network; the new features' recurrent input weights start at zero (the arm equals the base)."""
    sd = torch.load(path, map_location="cpu")
    w = sd.pop("encoder.gru.weight_ih_l0")  # [3 * 64, 64 + 1]: base image features, then the PV history value
    r = model.load_state_dict(sd, strict=False)
    assert not r.unexpected_keys and all(k.startswith("encoder.cnn2.") or k == "encoder.gru.weight_ih_l0" for k in r.missing_keys), r
    with torch.no_grad():
        g = model.encoder.gru.weight_ih_l0
        g.zero_()
        g[:, :64], g[:, 128] = w[:, :64], w[:, 64]
    model.head.init_from_targets = lambda *a, **k: None  # keep the base head (run_fold would reset it to climatology)


def lead_targets(rec_us, origin_us, lead_min, tol_us=30_000_000):
    """Row of the record at origin + lead (within 30 s), -1 if missing; robust to unsorted records."""
    order = np.argsort(rec_us, kind="stable")
    t = rec_us[order]
    want = origin_us + lead_min * 60_000_000
    j = np.clip(np.searchsorted(t, want), 1, len(t) - 1)
    j = np.where(np.abs(t[j - 1] - want) <= np.abs(t[j] - want), j - 1, j)
    return np.where(np.abs(t[j] - want) <= tol_us, order[j], -1)


def pilot_dataset(a, hires):
    ds = SkippdForecastDataset(a.h5, a.manifest, "dev", preload=True)
    m = np.load(a.manifest)
    rec = m["times_us"].astype("datetime64[us]")[m["hdf5_group"] == 0][np.argsort(m["hdf5_row"][m["hdf5_group"] == 0])]
    rec_us, org_us = rec.astype(np.int64), ds.origin_time.astype("datetime64[us]").astype(np.int64)
    have = np.zeros(len(ds.pv), bool)
    have[hires["row"][:].astype(np.int64)] = True
    t5 = lead_targets(rec_us, org_us, 5)
    t15 = lead_targets(rec_us, org_us, 15)
    assert np.array_equal(t15, ds.target_rows), "STOP: 15-min targets differ from the frozen manifest"
    keep = have[ds.hist_rows].all(1) & (t5 >= 0)  # common population of every arm and both leads
    tgt = t5 if a.lead == 5 else ds.target_rows
    if a.arm == "BASE":  # base network: all development samples with a target at this lead
        keep = tgt >= 0
    out = SimpleNamespace(pv=ds.pv, images=ds.images, record_times=rec, day=ds.day[keep], hist_rows=ds.hist_rows[keep], target_rows=tgt[keep],
                          origin_time=ds.origin_time[keep], target_time=rec[tgt[keep]], sample_ids=ds.sample_ids[keep])
    return out, int(keep.sum()), int(have[ds.hist_rows].all(1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--lead", type=int, required=True, choices=(5, 15))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold-ids", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--hires", default="D:/skippd_hires/pilot_v1.h5")
    ap.add_argument("--out", default="D:/skippd_hires/runs2")
    ap.add_argument("--h5", default="data/raw/2017_2019_images_pv_processed.hdf5")
    ap.add_argument("--manifest", default="data/manifests/forecast_tol30_v2.npz")
    ap.add_argument("--folds", default="data/manifests/folds_blocked5_v1.json")
    a = ap.parse_args()
    global INIT
    cfg = {**CFG, "seed": a.seed, "arm": a.arm, "lead_min": a.lead, **({} if a.arm == "BASE" else {"lr": ARM_LR})}
    base_root = Path(FROZEN15) if a.lead == 15 else Path(a.out) / "BASE" / "lead5" / f"seed{a.seed}"
    out_root = Path(a.out) / a.arm / f"lead{a.lead}" / f"seed{a.seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_f = open(out_root / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    hires = h5py.File(a.hires, "r")
    ds, n_keep, n_hist = pilot_dataset(a, hires)
    data = PilotData(ds, a.arm, hires, device)
    T.build_model = lambda c: build_model(a.arm, c)  # run_fold builds the arm's model; nothing else changes
    if a.arm.startswith("FULL"):
        T.predict_raw = functools.partial(T.predict_raw, batch=128)  # 256 x 256 frames: memory
    folds_doc = json.loads(Path(a.folds).read_text())
    n_par = sum(p.numel() for p in build_model(a.arm, cfg).parameters())
    (out_root / "config.json").write_text(json.dumps({**cfg, "n_parameters": n_par, "n_samples": n_keep, "n_samples_complete_history": n_hist,
                                                      "hires": a.hires, "folds_sha256": folds_doc["content_sha256"]}, indent=2))
    log(f"[{a.arm} lead {a.lead}] parameters {n_par}  pilot samples {n_keep} (complete raw history {n_hist})")
    summary = {}
    for k in a.fold_ids:
        if (out_root / f"fold_{k}" / "metrics.json").exists():
            continue
        fold = {**folds_doc["folds"][k], "day_labels": folds_doc["day_labels"]}
        if not np.isin(ds.day.astype(str), fold["val_days"]).any():
            log(f"[{a.arm} lead {a.lead}] fold {k}: no validation day in the pilot months, skipped")
            continue
        fold_dir = out_root / f"fold_{k}"
        fold_dir.mkdir(exist_ok=True)
        t0 = time.time()
        if a.arm != "BASE":
            INIT = base_root / f"fold_{k}" / "best.pt"
            assert INIT.exists(), f"STOP: base network missing ({INIT})"
        s = T.run_fold(ds, data, fold, "csi", cfg, fold_dir, log)
        summary[k] = {"best_epoch": s["best_epoch"], "val_crps_kw": s["probabilistic"]["overall"]["crps_kw"], "minutes": (time.time() - t0) / 60}
        log(f"[{a.arm} lead {a.lead}] fold {k} done: best epoch {s['best_epoch']}  val CRPS {summary[k]['val_crps_kw']:.4f} kW  ({summary[k]['minutes']:.1f} min)")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
