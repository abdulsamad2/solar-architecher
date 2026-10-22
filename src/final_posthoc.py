"""Post-hoc DESCRIPTIVE analysis of the committed one-shot test predictions (no model is re-run).

Inputs: experiments/final/test_predictions_v1.npz (raw predictions written by run_test), test labels, frozen
calibration state. Uses the identical frozen functions of src/final_eval.evaluate and first checks that it
reproduces the committed coverage / CRPS numbers of reports/test_results.json exactly.
Adds: per-day CRPS, upper / lower miss rates per stratum (incl. ramp-conditioned), day-bootstrap CIs of
G3''-F - comparator per stratum, up-/down-ramp split. None of this is a pre-registered hypothesis test.
Output: reports/test_posthoc_descriptive.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.baselines import official_persistence
from src.cal2 import ALPHA
from src.calibrate import base_interval
from src.clear_sky import ClearSkyPV
from src.final_eval import Split, day_bootstrap, nearest_init, rolling_conformal_daily
from src.metrics import crps_quantile_interp
from src.skippd import SUNNY_DATES

LEVELS = np.array([.05, .1, .2, .3, .5, .7, .8, .9, .95])


def main():
    committed = json.loads(Path("reports/test_results.json").read_text())
    p = np.load("experiments/final/test_predictions_v1.npz")
    split = Split("test", torch.device("cuda"))
    assert np.array_equal(p["sample_ids"], split.ds.sample_ids)
    n = len(split.ds.sample_ids)
    y = split.pv[split.ds.target_rows]
    day = split.ds.day.astype(str)
    sunny = np.array([np.datetime64(d).astype(object) in SUNNY_DATES for d in day])
    rd = json.loads(Path("reports/test_ramp_definition_frozen.json").read_text())
    k = split.pv / np.maximum(ClearSkyPV(rd["anchor_coef"])(split.rec_t), 1.0)
    dk = k[split.ds.target_rows] - k[split.ds.hist_rows[:, -1]]
    ramp = np.abs(dk) >= rd["tau_test"]
    h60 = split.h60_ok
    masks = {"overall": np.ones(n, bool), "sunny": sunny, "cloudy": ~sunny, "ramp": ramp, "non-ramp": ~ramp,
             "sunrise ramp": ramp & ~h60, "up-ramp": ramp & (dk > 0), "down-ramp": ramp & (dk < 0),
             "60-min eligible": h60, "fallback (no 60-min history)": ~h60}
    pers = official_persistence(split.ds.origin_time.astype(object), split.pv[split.ds.hist_rows[:, -1]])
    q = {m: p[f"q|{m}"] for m in ("G3''-F", "E4-B-fit", "E6-S", "G4", "D4")}
    crps = {m: crps_quantile_interp(q[m], y, LEVELS) for m in ("G3''-F", "E4-B-fit", "E6-S", "G4")}
    crps["persistence"] = np.abs(pers - y)
    crps["SUNSET port"] = np.abs(p["sunset"] - y)
    # consistency with the committed run
    for m in ("G3''-F", "E4-B-fit"):
        assert abs(crps[m].mean() - committed["scores"][m]["overall"]["crps_kw"]) < 1e-12, f"STOP: CRPS mismatch {m}"
    out = {"note": "POST-HOC DESCRIPTIVE analysis of committed test predictions; not pre-registered hypothesis tests.",
           "per_day": {}, "stratum_diff_ci": {}, "intervals": {}}
    for d in np.unique(day):
        s = day == d
        out["per_day"][d] = {"sunny": bool(sunny[s][0]), "n": int(s.sum()), "n_ramp": int(ramp[s].sum()),
                             **{f"crps|{m}": float(c[s].mean()) for m, c in crps.items()}}
    for comp in ("E4-B-fit", "G4", "E6-S", "persistence", "SUNSET port"):
        out["stratum_diff_ci"][f"G3''-F - {comp}"] = {
            g: day_bootstrap(crps["G3''-F"][v] - crps[comp][v], day[v], sunny[v]) for g, v in masks.items() if v.any()}
    out["stratum_diff_ci"]["D4 - E4-B-fit (60-min eligible)"] = {}
    c_d4 = crps_quantile_interp(q["D4"][h60], y[h60], LEVELS)
    for g, v in masks.items():
        vv = v[h60]
        if vv.any():
            out["stratum_diff_ci"]["D4 - E4-B-fit (60-min eligible)"][g] = day_bootstrap(
                c_d4[vv] - crps["E4-B-fit"][h60][vv], day[h60][vv], sunny[h60][vv])
    cal = np.load("experiments/final/dev_oof_calibration_v1.npz")
    meta = json.loads(Path("experiments/final/dev_oof_calibration_v1.json").read_text())
    org = split.ds.origin_time.astype("datetime64[us]").astype(np.int64)
    tgt = split.ds.target_time.astype("datetime64[us]").astype(np.int64)
    for m in ("G3''-F", "E4-B-fit", "E6-S", "G4"):
        out["intervals"][m] = {}
        for nm, a in ALPHA.items():
            lo, hi = base_interval(q[m], LEVELS, nm)
            W = meta["W"][m]
            inits = {d: nearest_init(cal["origin_us"], cal[f"{m}|{nm}"], org[day == d].min(), W) for d in np.unique(day)}
            clo, chi = rolling_conformal_daily(org, tgt, day, lo, hi, y, a, W, inits)
            cov = float(np.mean((y >= clo) & (y <= chi)))
            assert abs(cov - committed["intervals"][m][nm]["overall"]["coverage"]) < 1e-12, f"STOP: coverage mismatch {m} {nm}"
            out["intervals"][m][nm] = {g: {"coverage": float(np.mean((y[v] >= clo[v]) & (y[v] <= chi[v]))),
                                           "miss_below_lower": float(np.mean(y[v] < clo[v])),
                                           "miss_above_upper": float(np.mean(y[v] > chi[v])),
                                           "width_kw": float(np.mean(chi[v] - clo[v])),
                                           "uncalibrated_coverage": float(np.mean((y[v] >= lo[v]) & (y[v] <= hi[v])))}
                                       for g, v in masks.items() if v.any()}
    out["consistency_with_committed_results"] = "CRPS (G3''-F, E4-B-fit) and overall coverage (all models, 50/80/90) reproduced exactly"
    Path("reports/test_posthoc_descriptive.json").write_text(json.dumps(out, indent=1, default=float))
    print("post-hoc done; consistency checks passed")


if __name__ == "__main__":
    main()
