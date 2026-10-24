"""Pre-registered E5 lag analysis for Gatton (reports/phaseE_gatton_lag_procedure.md), computed ONLY from the immutable
raw prediction file written by the evaluation (no model is run, nothing is re-fitted).

  python -m src.gatton_lag experiments/E_gatton/test_predictions_v1.npz reports/phaseE_gatton_test_results.json OUT.json

Method (identical to src/lag_analysis.best_lag, fixed before execution):
  per day, series aligned at the TARGET time on the record grid (1-min step for the test group, 2-min for development);
  first differences of the median forecast and of observed PV; Pearson correlation of forecast increments at time T+L
  with observed increments at T, L = -5 ... +20 min in grid steps; best lag = argmax; also correlation at L = 0 and 15.
  Summary per model: median best lag over days, mean lag-0 and lag-15 correlation; for all days and for cloudy days.
Models: F1 G3''-F, F1 E4-B-fit, F1 E6-S, F0 G3''-F, Z0 G3''-F, smart persistence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.lag_analysis import best_lag

MAN = "data/manifests/gatton_forecast_v1.npz"
MODELS = ("F1|G3''-F", "F1|E4-B-fit", "F1|E6-S", "F0|G3''-F", "Z0|G3''-F")


def main(pred_path: str, results_path: str, out_path: str):
    p = np.load(pred_path)
    res = json.loads(Path(results_path).read_text())
    m = np.load(MAN)
    sid = p["sample_ids"]
    is_test = bool(m["is_test"][sid].all())
    step = 1 if is_test else 2  # fixed by the sample cadence of the group, not chosen from data
    t_tgt = m["times_us"][m["target"][sid]].astype("datetime64[us]").astype("datetime64[m]")
    day = t_tgt.astype("datetime64[D]").astype(str)
    minute = (t_tgt - t_tgt.astype("datetime64[D]")).astype(int)
    series = {k: p[k][:, 4] for k in MODELS}
    series["smart persistence"] = p["persistence"]
    y = p["y"]
    per_day = {}
    for d in np.unique(day):
        s = day == d
        o = np.argsort(minute[s])
        per_day[d] = {k: best_lag(minute[s][o], f[s][o], y[s][o], step) for k, f in series.items()}
    sunny = {d: bool(v.get("sunny", False)) for d, v in res.get("per_day", {}).items()}

    def summary(days):
        return {k: {"median_best_lag_min": float(np.median([per_day[d][k][0] for d in days])),
                    "mean_corr_lag0": float(np.nanmean([per_day[d][k][1] for d in days])),
                    "mean_corr_lag15": float(np.nanmean([per_day[d][k][2] for d in days]))} for k in series} if days else {}

    cloudy = [d for d in per_day if not sunny.get(d, False)]
    out = {"step_min": step, "n_days": len(per_day), "n_cloudy_days": len(cloudy), "all_days": summary(list(per_day)),
           "cloudy_days": summary(cloudy), "per_day": {d: {k: list(v) for k, v in r.items()} for d, r in per_day.items()}}
    Path(out_path).write_text(json.dumps(out, indent=1))
    for k, v in out["all_days"].items():
        print(f"{k:18s} best lag {v['median_best_lag_min']:5.1f} | corr@0 {v['mean_corr_lag0']:+.3f} | corr@15 {v['mean_corr_lag15']:+.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
