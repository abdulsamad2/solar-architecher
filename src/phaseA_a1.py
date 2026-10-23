"""Phase A1 (reports/phaseA_comparability_prereg.md): SKIPP'D-paper metrics on the committed one-shot predictions.

No inference: re-scores experiments/final/test_predictions_v1.npz (hash-checked) on (i) all 13,266 test samples and
(ii) the exact-timing subset (every history frame and the target at exactly t-j min / t+15 min).
Output: reports/phaseA_a1_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.baselines import official_persistence
from src.final_eval import H5, MAN, sha256
from src.skippd import SUNNY_DATES, SkippdForecastDataset

PUBLISHED_SUNSET = {"sunny": {"rmse": 0.61, "mae": 0.50, "fs": -0.4580},
                    "cloudy": {"rmse": 4.27, "mae": 2.95, "fs": 0.1703},
                    "overall": {"rmse": 3.03, "mae": 1.71, "fs": 0.1644}}
SEED = 20261022


def skill_boot(err_m, err_p, day, sunny_day, n_boot=10_000):
    """Day-bootstrap (stratified sunny/cloudy) of RMSE skill 1 - RMSE_m / RMSE_p; returns (estimate, lo, hi)."""
    days, inv = np.unique(day, return_inverse=True)
    sm, sp, c = (np.bincount(inv, weights=err_m ** 2), np.bincount(inv, weights=err_p ** 2), np.bincount(inv))
    sunny = np.zeros(len(days), bool)
    sunny[inv] = sunny_day
    groups = [g for g in (np.flatnonzero(sunny), np.flatnonzero(~sunny)) if len(g)]
    rng = np.random.default_rng(SEED)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        ii = np.concatenate([rng.choice(g, len(g)) for g in groups])
        boots[b] = 1 - np.sqrt(sm[ii].sum() / c[ii].sum()) / np.sqrt(sp[ii].sum() / c[ii].sum())
    est = 1 - np.sqrt(sm.sum() / c.sum()) / np.sqrt(sp.sum() / c.sum())
    return float(est), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main():
    raw = json.loads(Path("reports/test_raw_output_hashes.json").read_text(encoding="utf-8-sig"))
    assert sha256("experiments/final/test_predictions_v1.npz") == raw["test_predictions_v1.npz"], "STOP: predictions hash"
    p = np.load("experiments/final/test_predictions_v1.npz")
    ds = SkippdForecastDataset(H5, MAN, "test", preload=False)
    assert np.array_equal(ds.sample_ids, p["sample_ids"]), "STOP: sample order"
    pv = ds.pv.numpy().astype(float)
    y = pv[ds.target_rows]
    day = ds.day.astype(str)
    sunny = np.array([np.datetime64(d).astype(object) in SUNNY_DATES for d in day])
    pers = official_persistence(ds.origin_time.astype(object), pv[ds.hist_rows[:, -1]])
    # exact-timing subset (official matching rule): history column j (newest first) at t - j min, target at t + 15 min
    m = np.load(MAN)
    t = m["times_us"]
    ids = ds.sample_ids
    t0 = t[m["origin"][ids]]
    exact = (t[m["history"][ids]] == t0[:, None] - np.arange(16)[None, :] * 60_000_000).all(1) & \
            (t[m["target"][ids]] == t0 + 15 * 60_000_000)
    point = {"G3''-F": p["q|G3''-F"][:, 4], "E4-B-fit": p["q|E4-B-fit"][:, 4], "E6-S": p["q|E6-S"][:, 4],
             "G4": p["q|G4"][:, 4], "SUNSET port": p["sunset"], "smart persistence": pers}
    out = {"note": "Phase A1: re-scoring of committed one-shot predictions (no inference); descriptive.",
           "published_SUNSET_SKIPPD_Table5.1": PUBLISHED_SUNSET, "populations": {}}
    for pop, sel in (("all samples (+/-30 s matching)", np.ones(len(y), bool)), ("exact-timing subset", exact)):
        strata = {"overall": sel, "sunny": sel & sunny, "cloudy": sel & ~sunny}
        r = {"n": {g: int(v.sum()) for g, v in strata.items()}, "metrics": {}, "bootstrap": {}}
        for name, f in point.items():
            e = f - y
            r["metrics"][name] = {g: {"rmse": float(np.sqrt(np.mean(e[v] ** 2))), "mae": float(np.mean(np.abs(e[v]))),
                                      "fs_rmse": float(1 - np.sqrt(np.mean(e[v] ** 2)) / np.sqrt(np.mean((pers - y)[v] ** 2)))}
                                  for g, v in strata.items()}
        for name in ("G3''-F", "SUNSET port"):
            r["bootstrap"][f"FS {name} vs persistence"] = {
                g: skill_boot(point[name][v] - y[v], pers[v] - y[v], day[v], sunny[v]) for g, v in strata.items()}
        for comp in ("SUNSET port", "smart persistence"):
            d = np.abs(point["G3''-F"] - y) - np.abs(point[comp] - y)
            from src.final_eval import day_bootstrap
            r["bootstrap"][f"MAE G3''-F - {comp}"] = {g: day_bootstrap(d[v], day[v], sunny[v]) for g, v in strata.items()}
        port = r["metrics"]["SUNSET port"]
        r["port_minus_published"] = {g: {"rmse": port[g]["rmse"] - PUBLISHED_SUNSET[g]["rmse"],
                                         "mae": port[g]["mae"] - PUBLISHED_SUNSET[g]["mae"]} for g in strata}
        out["populations"][pop] = r
    gap = out["populations"]["exact-timing subset"]["port_minus_published"]["overall"]
    out["reporting_rule"] = {"port_within_0.05kW_of_published_on_exact_subset": bool(abs(gap["rmse"]) <= 0.05 and abs(gap["mae"]) <= 0.05)}
    Path("reports/phaseA_a1_results.json").write_text(json.dumps(out, indent=1))
    for pop, r in out["populations"].items():
        print(f"== {pop}  n={r['n']}")
        for name, mm in r["metrics"].items():
            print(f"  {name:18s}" + "  ".join(f"{g}: RMSE {v['rmse']:.3f} MAE {v['mae']:.3f} FS {100 * v['fs_rmse']:+.1f}%"
                                              for g, v in mm.items()))
        print("  port-published", {g: {k: round(x, 3) for k, x in v.items()} for g, v in r["port_minus_published"].items()})
        for k, v in r["bootstrap"].items():
            print("  ", k, {g: [round(x, 4) for x in t_] for g, t_ in v.items()})
    print("rule:", out["reporting_rule"])


if __name__ == "__main__":
    main()
