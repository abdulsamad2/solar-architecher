"""Revision C16 / C20: where the CRPS gain occurs and comparisons with the strongest comparators (frozen test predictions).

Protocol: revision/protocols/R-C16_C17_C20.md. All statistics are post hoc. (C17 interval tables: revision_c07.)
"""
from __future__ import annotations

import numpy as np

from src.baselines import official_persistence
from src.metrics import crps_quantile_interp
from src.revision_common import LEVELS, day_bootstrap, frozen_test_predictions, save, test_context

MODELS = ("G3''-F", "E4-B-fit", "G4", "E6-S")


def main():
    c = test_context()
    q = frozen_test_predictions(c["sample_ids"])
    y, day, sunny, masks = c["y"], c["day"], c["sunny"], c["masks"]
    crps = {m: crps_quantile_interp(q[m], y, LEVELS) for m in MODELS}
    pers = official_persistence(c["ds"].origin_time.astype(object), c["pv"][c["ds"].hist_rows[:, -1]])
    crps["persistence"] = np.abs(pers - y)
    crps["SUNSET port"] = np.abs(q["sunset"] - y)
    n = len(y)
    res = {"protocol": "revision/protocols/R-C16_C17_C20.md", "n": n, "contribution": {}, "bootstrap": {}, "weighting": {}}
    # C16: overall diff = sum_s (n_s / N) * diff_s for each partition
    for comp in ("E4-B-fit", "G4", "E6-S"):
        d = crps["G3''-F"] - crps[comp]
        tot = float(d.mean())
        out = {"overall_diff": tot}
        for part in (("sunny", "cloudy"), ("ramp", "non-ramp")):
            for s in part:
                v = masks[s]
                out[s] = {"n": int(v.sum()), "diff": float(d[v].mean()), "weighted": float(d[v].sum() / n),
                          "share_of_overall": float(d[v].sum() / n / tot) if tot != 0 else None}
        out["per_day"] = {dd: float(d[day == dd].mean()) for dd in np.unique(day)}
        res["contribution"][comp] = out
    # C20: paired day bootstrap of G3''-F vs E6-S and G4 (and E4-B-fit for reference) on every stratum
    sunny_day = sunny
    for comp in ("E6-S", "G4", "E4-B-fit", "persistence", "SUNSET port"):
        d = crps["G3''-F"] - crps[comp]
        res["bootstrap"][comp] = {g: day_bootstrap(d[v], day[v], sunny_day[v]) for g, v in masks.items() if v.any()}
    # sample-weighted (pooled) vs day-weighted (mean of daily means)
    for m, cc in crps.items():
        res["weighting"][m] = {g: {"sample_weighted": float(cc[v].mean()),
                                   "day_weighted": float(np.mean([cc[v & (day == dd)].mean() for dd in np.unique(day[v])]))}
                               for g, v in masks.items() if v.any()}
    p = save("c16_c20_contribution_bootstrap.json", res)
    print("saved", p)
    for comp in ("E4-B-fit", "G4", "E6-S"):
        r = res["contribution"][comp]
        print(comp, f"overall {r['overall_diff']:+.3f}", " ".join(f"{s} {r[s]['share_of_overall']:.2f}" for s in ("sunny", "cloudy", "ramp", "non-ramp")))
    for comp in ("E6-S", "G4"):
        print(comp, {g: tuple(round(x, 3) for x in v) for g, v in res["bootstrap"][comp].items()})


if __name__ == "__main__":
    main()
