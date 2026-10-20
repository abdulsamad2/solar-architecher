"""Add per-sample context fields to prediction files written before train.py saved them (seed-42 E4 runs).

Fields: origin/target time, P_clear at origin and target, PV now, fold ID. P_clear is recomputed with the
fold's full-training-day anchor, exactly as in training; the saved persistence forecast must be
reproduced to 1e-9 or the backfill aborts. Predictions are never modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.clear_sky import ClearSkyPV, smart_persistence
from src.skippd import SkippdForecastDataset

FIELDS = ("origin_time_us", "target_time_us", "pclear_target_kw", "pclear_now_kw", "pv_now_kw", "fold")


def main(run_dirs, h5, manifest, folds_path) -> int:
    ds = SkippdForecastDataset(h5, manifest, "dev", preload=False)
    m = np.load(manifest)
    g0 = m["hdf5_group"] == 0
    rec_t = m["times_us"][g0][np.argsort(m["hdf5_row"][g0])].astype("datetime64[us]")
    rec_day = rec_t.astype("datetime64[D]").astype(str)
    pv = ds.pv.numpy().astype(float)
    pos = {sid: i for i, sid in enumerate(ds.sample_ids)}
    folds = json.loads(Path(folds_path).read_text())["folds"]
    anchors = {}
    for run in map(Path, run_dirs):
        for k in range(5):
            f = run / f"fold_{k}" / "preds_val.npz"
            z = dict(np.load(f))
            if all(x in z for x in FIELDS):
                print(f"{f}: already has context fields")
                continue
            if k not in anchors:
                tr = np.isin(rec_day, folds[k]["train_days"])
                anchors[k] = ClearSkyPV().fit(rec_t[tr], pv[tr])(rec_t)
            pc = anchors[k]
            idx = np.array([pos[s] for s in z["sample_ids"]])
            hist_now, tgt = ds.hist_rows[idx, -1], ds.target_rows[idx]
            pers = smart_persistence(pv[hist_now], pc[hist_now], pc[tgt], 1.0)
            if not np.allclose(pers, z["persistence"], atol=1e-9, rtol=0):
                print(f"ABORT {f}: recomputed persistence differs (max {np.abs(pers - z['persistence']).max():.2e})")
                return 1
            z.update(origin_time_us=ds.origin_time[idx].astype("datetime64[us]").astype(np.int64),
                     target_time_us=ds.target_time[idx].astype("datetime64[us]").astype(np.int64),
                     pclear_target_kw=pc[tgt], pclear_now_kw=pc[hist_now], pv_now_kw=pv[hist_now],
                     fold=np.full(len(idx), k, np.int8))
            np.savez_compressed(f, **z)
            print(f"{f}: backfilled ({len(idx):,} samples; persistence reproduced exactly)")
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--h5", default="data/raw/2017_2019_images_pv_processed.hdf5")
    p.add_argument("--manifest", default="data/manifests/forecast_tol30_v2.npz")
    p.add_argument("--folds", default="data/manifests/folds_blocked5_v1.json")
    a = p.parse_args()
    sys.exit(main(a.runs, a.h5, a.manifest, a.folds))
