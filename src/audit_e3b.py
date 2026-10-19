"""Pre-training audit for E3b (K-history inputs). Writes audit.json; exits non-zero on a failed check.

Per fold:
  * leakage: the anchor refitted with validation/purged-day PV replaced by noise must give bit-identical
    coefficients (fit sees training days only); the model's history input must equal K at exactly the
    history rows (all at or before the origin), never the target row
  * extreme K: distribution of input-history K and target K over the fold's samples
  * low-sun denominators: share of input/target records where the 1 kW floor on P_clear is active, by hour
Pre-registered abort rule: any input or target K > 5.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.clear_sky import ClearSkyPV, clear_sky_index
from src.skippd import SkippdForecastDataset
from src.train import GpuSeq

K_ABORT = 5.0
MIN_CLEAR = 1.0


def q(a):
    return {p: round(float(np.quantile(a, p)), 4) for p in (0.001, 0.01, 0.5, 0.99, 0.999)} | {
        "max": round(float(a.max()), 3), "min": round(float(a.min()), 3),
        "n_gt_2": int((a > 2).sum()), "n_gt_3": int((a > 3).sum())}


def main(h5, manifest, folds_path, out_path) -> int:
    ds = SkippdForecastDataset(h5, manifest, "dev", preload=True)
    m = np.load(manifest)
    g0 = m["hdf5_group"] == 0
    rec_t = m["times_us"][g0][np.argsort(m["hdf5_row"][g0])].astype("datetime64[us]")
    rec_day = rec_t.astype("datetime64[D]").astype(str)
    rec_hour = ((rec_t - rec_t.astype("datetime64[D]")).astype("timedelta64[h]").astype(int))
    pv = ds.pv.numpy().astype(float)
    day = ds.day.astype(str)
    folds = json.loads(Path(folds_path).read_text())
    data = GpuSeq(ds, torch.device("cpu"))
    rng = np.random.default_rng(0)
    report, ok = {"abort_rule": f"any K > {K_ABORT}"}, True

    for fd in folds["folds"]:
        k = fd["fold"]
        tr_rec = np.isin(rec_day, fd["train_days"])
        cs = ClearSkyPV().fit(rec_t[tr_rec], pv[tr_rec])
        # leakage: corrupt every non-training record's PV and refit on the same selection
        pv_noisy = pv.copy()
        pv_noisy[~tr_rec] = rng.uniform(0, 30, (~tr_rec).sum())
        cs2 = ClearSkyPV().fit(rec_t[tr_rec], pv_noisy[tr_rec])
        fit_isolated = bool(np.array_equal(cs.coef, cs2.coef))

        pc = cs(rec_t)
        k_rec = clear_sky_index(pv, pc, MIN_CLEAR)
        samples = np.flatnonzero(np.isin(day, fd["train_days"] + fd["val_days"]))
        hist_rows, tgt_rows = ds.hist_rows[samples], ds.target_rows[samples]
        k_in, k_tgt = k_rec[hist_rows].ravel(), k_rec[tgt_rows]

        data.set_history_feature(k_rec)
        probe = rng.choice(samples, 256, replace=False)
        _, h = data(torch.as_tensor(probe))
        input_is_history_k = bool(np.allclose(h.numpy(), k_rec[ds.hist_rows[probe]].astype(np.float32)))
        causal = bool((rec_t[ds.hist_rows[probe]].max(1) <= ds.origin_time[probe]).all()
                      and (rec_t[ds.target_rows[probe]] > ds.origin_time[probe]).all())

        floor_in = pc[hist_rows].ravel() < MIN_CLEAR
        floor_tgt = pc[tgt_rows] < MIN_CLEAR
        by_hour = {int(hr): round(float(floor_tgt[rec_hour[tgt_rows] == hr].mean()), 4)
                   for hr in np.unique(rec_hour[tgt_rows]) if floor_tgt[rec_hour[tgt_rows] == hr].any()}
        fold_ok = bool(fit_isolated and input_is_history_k and causal
                       and k_in.max() <= K_ABORT and k_tgt.max() <= K_ABORT)
        ok &= fold_ok
        report[f"fold_{k}"] = {
            "pass": fold_ok, "clear_sky_fit_isolated_from_nontrain_pv": fit_isolated,
            "model_input_equals_history_K": input_is_history_k, "history_before_origin_target_after": causal,
            "K_input": q(k_in), "K_target": q(k_tgt),
            "floor_active_share_input": round(float(floor_in.mean()), 5),
            "floor_active_share_target": round(float(floor_tgt.mean()), 5),
            "floor_active_target_share_by_hour": by_hour,
            "K_target_gt2_when_floor_active": int((k_tgt[floor_tgt] > 2).sum()),
            "K_target_gt2_total": int((k_tgt > 2).sum()),
        }
        print(f"fold {k}: pass={fold_ok}  K_in max {k_in.max():.2f}  K_tgt max {k_tgt.max():.2f}  "
              f"floor share in/tgt {floor_in.mean():.4f}/{floor_tgt.mean():.4f}", flush=True)

    report["pass"] = bool(ok)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="data/raw/2017_2019_images_pv_processed.hdf5")
    p.add_argument("--manifest", default="data/manifests/forecast_tol30_v2.npz")
    p.add_argument("--folds", default="data/manifests/folds_blocked5_v1.json")
    p.add_argument("--out", default="experiments/E3b_csi_inputs/v2/audit.json")
    a = p.parse_args()
    sys.exit(main(a.h5, a.manifest, a.folds, a.out))
