"""Temporally blocked development folds and parameter-free day sky labels.

Folds: dev days sorted chronologically, cut into ``n_folds`` contiguous blocks with equal day counts.
For fold k the validation set is block k; training days within ``purge_days`` calendar days of the
block's edges are dropped (forecast windows never cross days, so this guards against
multi-day weather persistence, not window overlap). The official test days are never present.

Day sky label (for stratified dev metrics only, never as a model input):
  sunny  <=> corr(PV, clear-sky POA) > 0.995 and mean |dPV/dmin| < 0.15 kW
Scale-free in the clear-sky model, so no parameters are fitted on validation days.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.clear_sky import clear_sky_poa
from src.skippd import SUNNY_DATES, TEST_DATES

CAMERA_SHIFT_HYPOTHESIS = dt.date(2019, 5, 22)  # unconfirmed; recorded as a regime flag only


def day_sky_labels(times, pv_kw, corr_min: float = 0.995, smooth_max: float = 0.15) -> pd.DataFrame:
    df = pd.DataFrame({"pv": np.asarray(pv_kw, float), "poa": clear_sky_poa(times),
                       "day": np.asarray(times, dtype="datetime64[D]")})
    g = df.groupby("day")
    out = pd.DataFrame({
        "corr": g.apply(lambda d: np.corrcoef(d.pv, d.poa)[0, 1] if len(d) > 2 else np.nan),
        "smooth_kw": g["pv"].apply(lambda s: np.abs(np.diff(s.to_numpy())).mean() if len(s) > 1 else np.nan),
        "n_records": g.size(),
    })
    out["sunny"] = (out["corr"] > corr_min) & (out["smooth_kw"] < smooth_max)
    return out


def blocked_folds(days, n_folds: int = 5, purge_days: int = 1) -> list[dict]:
    days = sorted(set(days))
    blocks = np.array_split(np.arange(len(days)), n_folds)
    folds = []
    for k, blk in enumerate(blocks):
        val = [days[i] for i in blk]
        lo, hi = val[0] - dt.timedelta(days=purge_days), val[-1] + dt.timedelta(days=purge_days)
        train = [d for d in days if d not in set(val) and not (lo <= d <= hi)]
        folds.append({"fold": k, "val_days": val, "train_days": train,
                      "purged_days": [d for d in days if d not in set(val) and lo <= d <= hi]})
    return folds


def build(raw_dir: str | Path, out_path: str | Path, n_folds: int = 5, purge_days: int = 1) -> dict:
    raw_dir = Path(raw_dir)
    report = {}
    with h5py.File(raw_dir / "2017_2019_images_pv_processed.hdf5", "r") as f:
        pv_tv, pv_te = f["trainval/pv_log"][...], f["test/pv_log"][...]
    t_tv = np.load(raw_dir / "times_trainval.npy", allow_pickle=True)
    t_te = np.load(raw_dir / "times_test.npy", allow_pickle=True)

    # rule check against the official labels (labels only; no test performance involved)
    lab_te = day_sky_labels(t_te, pv_te)
    official = np.array([d.astype(object) in SUNNY_DATES for d in lab_te.index.values.astype("datetime64[D]")])
    report["label_rule_vs_official_test_days"] = {
        "agree": int((lab_te["sunny"].to_numpy() == official).sum()), "n": int(len(official)),
        "disagree_days": [str(d)[:10] for d, a, b in zip(lab_te.index, lab_te["sunny"], official) if a != b],
    }

    lab = day_sky_labels(t_tv, pv_tv)
    days = [d.astype(object) for d in lab.index.values.astype("datetime64[D]")]
    assert not set(days) & TEST_DATES, "official test day found in dev"
    folds = blocked_folds(days, n_folds, purge_days)
    sunny = dict(zip(days, lab["sunny"].to_numpy()))
    for fd in folds:
        v = fd["val_days"]
        fd["val_range"] = [str(v[0]), str(v[-1])]
        fd["n_val_days"], fd["n_train_days"] = len(v), len(fd["train_days"])
        fd["n_val_sunny_days"] = int(sum(sunny[d] for d in v))
        fd["n_val_days_after_camera_hypothesis"] = int(sum(d >= CAMERA_SHIFT_HYPOTHESIS for d in v))
        for key in ("val_days", "train_days", "purged_days"):
            fd[key] = [str(d) for d in fd[key]]
    payload = {
        "n_folds": n_folds, "purge_days": purge_days,
        "day_labels": {str(d): bool(sunny[d]) for d in days},
        "folds": folds, "report": report,
    }
    payload["content_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=1))
    return payload


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build temporally blocked dev folds")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out", default="data/manifests/folds_blocked5_v1.json")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--purge-days", type=int, default=1)
    a = p.parse_args()
    r = build(a.raw_dir, a.out, a.n_folds, a.purge_days)
    print(json.dumps(r["report"], indent=2))
    for fd in r["folds"]:
        print({k: fd[k] for k in ("fold", "val_range", "n_val_days", "n_train_days", "n_val_sunny_days",
                                  "n_val_days_after_camera_hypothesis")}, "purged", len(fd["purged_days"]))
    print("dev sunny days:", sum(r["day_labels"].values()), "of", len(r["day_labels"]), "| sha256", r["content_sha256"][:16])
