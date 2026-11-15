"""Revision R-Z2: proper ramp-weighted (threshold-weighted) CRPS for the ramp comparisons
(protocol revision/protocols/R-Z2_ramp_weighted_crps.md). Post hoc test, frozen predictions, CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.metrics import crps_quantile_interp, quantile_function
from src.revision_common import LEVELS, day_bootstrap, frozen_test_predictions, save, test_context
from src.revision_posthoc_test import members
from src.revision_z_common import ramp_region

FROZEN = ("G3''-F", "E4-B-fit", "E6-S", "G4")
PAIRS = (("G3''-F", "E4-B-fit"), ("G3''-F", "E6-S"), ("G3''-F", "G4"), ("I15", "E4-B-fit"), ("I15", "P15"), ("E4-B-fit", "P15"))
N_GRID = 200


def tw_crps(pred_q, y, lo, hi):
    """(CRPS, twCRPS_ramp, twCRPS_calm): 200-atom CRPS of F, of v(F) and of clip(F) with the chaining functions of the
    ramp weight 1{z <= lo or z >= hi} and of its complement."""
    taus = (np.arange(N_GRID) + 0.5) / N_GRID
    qf = quantile_function(pred_q, LEVELS, taus)
    pin = lambda a, b: 2 * np.maximum(taus * (a[:, None] - b), (taus - 1) * (a[:, None] - b)).mean(1)
    calm = lambda z: np.clip(z, lo[:, None] if z.ndim == 2 else lo, hi[:, None] if z.ndim == 2 else hi)
    ramp = lambda z: z - calm(z)  # = v(z) up to the per-forecast constant -L, which cancels in the score
    return pin(y, qf), pin(ramp(y), ramp(qf)), pin(calm(y), calm(qf))


def main():
    c = test_context()
    y, day, sunny = c["y"], c["day"], c["sunny"]
    _, lo, hi = ramp_region(c)
    q = frozen_test_predictions(c["sample_ids"])
    arms = {m: q[m] for m in FROZEN} | {m: np.nanmean(members(m)["pred_q"], 0) for m in ("I15", "P15")}
    lines = Path("paper/data/t4_main_test.csv").read_text(encoding="utf-8").splitlines()
    ci = lines[0].split(",").index("crps_kw")
    ref = {"E4-B-fit": None, "E6-S": None, "G4": None}
    for row in lines[1:]:
        if row.split(",")[0] in ref:
            ref[row.split(",")[0]] = float(row.split(",")[ci])
    ref["G3''-F"] = float(lines[-1].split(",")[ci])  # last row is the proposed forecaster
    x3 = json.loads(Path("revision/results/x3_images_only.json").read_text())["posthoc_test"]["crps"]["full"]
    ref |= {m: x3[m]["overall"]["crps"] for m in ("I15", "P15")}
    res = {"protocol": "revision/protocols/R-Z2_ramp_weighted_crps.md", "n": int(len(y)), "ramp_share": float(c["ramp"].mean()),
           "scores": {}, "pairs": {}, "checks": {}}
    sc = {}
    for m, qq in arms.items():
        full, tr, tc = tw_crps(qq, y, lo, hi)
        assert np.abs(full - crps_quantile_interp(qq, y, LEVELS)).max() < 1e-12
        assert abs(full.mean() - ref[m]) < 1e-6, f"STOP: CRPS of {m} not reproduced ({full.mean()} vs {ref[m]})"
        add = float(np.abs(full - tr - tc).max())
        assert add <= 1e-9, f"STOP: decomposition not exact ({m}: {add})"
        sc[m] = {"crps": full, "tw_ramp": tr, "tw_calm": tc}
        res["checks"][m] = {"crps_minus_reference": float(full.mean() - ref[m]), "max_abs_decomposition_error": add}
        res["scores"][m] = {"crps": float(full.mean()), "tw_ramp": float(tr.mean()), "tw_calm": float(tc.mean()),
                            "ramp_stratified_crps": float(full[c["ramp"]].mean()), "nonramp_stratified_crps": float(full[~c["ramp"]].mean()),
                            # where the ramp-region score is incurred: on ramp outcomes (missed mass) or non-ramp outcomes (false-alarm mass)
                            "tw_ramp_on_ramp_outcomes": float(tr[c["ramp"]].sum() / len(y)), "tw_ramp_on_nonramp_outcomes": float(tr[~c["ramp"]].sum() / len(y))}
        print(f"{m:9s} CRPS {full.mean():.4f} = ramp region {tr.mean():.4f} + calm region {tc.mean():.4f} | outcome-stratified ramp CRPS {full[c['ramp']].mean():.3f}", flush=True)
    for a, b in PAIRS:
        d = {k: day_bootstrap(sc[a][k] - sc[b][k], day, sunny) for k in ("crps", "tw_ramp", "tw_calm")}
        d["ramp_stratified"] = day_bootstrap((sc[a]["crps"] - sc[b]["crps"])[c["ramp"]], day[c["ramp"]], sunny[c["ramp"]])
        t, s = d["tw_ramp"], d["ramp_stratified"]
        if t[1] <= 0 <= t[2]:
            d["verdict"] = "not confirmed"
        else:
            d["verdict"] = "holds under the proper score" if np.sign(t[0]) == np.sign(s[0]) else "reversed"
        res["pairs"][f"{a} - {b}"] = d
        f = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"
        print(f"{a} - {b}: twCRPS_ramp {f(t)}  twCRPS_calm {f(d['tw_calm'])}  CRPS {f(d['crps'])} | outcome-stratified ramp {f(s)} => {d['verdict']}", flush=True)
    print("saved", save("z2_ramp_weighted_crps.json", res))


if __name__ == "__main__":
    main()
