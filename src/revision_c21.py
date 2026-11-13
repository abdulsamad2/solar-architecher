"""Revision C21: sample-based CRPS of our quantile forecasts with matched sample counts (contextual comparison).

Protocol: revision/protocols/R-C22_C21_latency_external.md. Published SKIPP'D probabilistic results (SkyGPT m = 100,
PV-MM-Diffusion m = 30, SUNSET / oracle m = 10) use the standard empirical-ensemble CRPS, which is biased upward by
about E|X - X'| / (2m). We draw m samples from each forecast by inverse-transform sampling of the piecewise-linear
quantile function (constant tails, as the scorer) and report the standard and the fair (Ferro 2014) estimator.
"""
from __future__ import annotations

import h5py
import numpy as np

from src.calibrate import extended_quantile_function
from src.metrics import crps_quantile_interp
from src.phaseA_a2 import H5X
from src.revision_common import LEVELS, frozen_test_predictions, save, test_context

M_GRID, REPEATS, SEED = (10, 30, 100), 100, 20261113


def sample(q, m, rng):
    u = rng.random((len(q), m))
    taus = np.linspace(0.0005, 0.9995, 2000)
    qf = extended_quantile_function(q, LEVELS, taus, floor=-np.inf)
    j = np.clip(np.searchsorted(taus, u), 0, len(taus) - 1)
    return np.take_along_axis(qf, j, 1)


def crps_ens(x, y, fair: bool):
    """x [N, m]: standard = mean|X-y| - 0.5 mean|X-X'|; fair divides the second term by m(m-1) instead of m^2."""
    m = x.shape[1]
    a = np.abs(x - y[:, None]).mean(1)
    xs = np.sort(x, 1)
    i = np.arange(1, m + 1)
    pair_sum = 2 * ((2 * i - m - 1) * xs).sum(1)  # sum_{j,k} |x_j - x_k|
    return a - pair_sum / (2 * (m * (m - 1) if fair else m * m))


def evaluate(qs: dict, y, rng) -> dict:
    out = {}
    for name, q in qs.items():
        r = {"quantile_crps_exact": float(crps_quantile_interp(q, y, LEVELS).mean())}
        for m in M_GRID:
            for fair in (False, True):
                v = [crps_ens(sample(q, m, rng), y, fair).mean() for _ in range(REPEATS)]
                r[f"m={m} {'fair' if fair else 'standard'}"] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
        out[name] = r
    return out


def main():
    rng = np.random.default_rng(SEED)
    p = np.load("experiments/phaseA/a2_predictions.npz")
    with h5py.File(H5X, "r") as f:
        y_ext = f["test/pv_pred"][:, 14].astype(float)
    res = {"protocol": "revision/protocols/R-C22_C21_latency_external.md", "repeats": REPEATS, "label": "CONTEXTUAL, UNPAIRED"}
    res["skygpt_external_2582"] = evaluate({k.split("|")[1]: p[k] for k in p.files if k.startswith("q|")}, y_ext, rng)
    sun = p["sunset_members"].T
    res["skygpt_external_2582"]["SUNSET port (10 real members)"] = {
        "m=10 standard": float(crps_ens(sun, y_ext, False).mean()), "m=10 fair": float(crps_ens(sun, y_ext, True).mean())}
    c = test_context()
    q = frozen_test_predictions(c["sample_ids"])
    res["official_test_13266"] = evaluate({k: q[k] for k in ("G3''-F", "E4-B-fit", "E6-S", "G4")}, c["y"], rng)
    save("c21_external_crps.json", res)
    for pop in ("skygpt_external_2582", "official_test_13266"):
        print(pop)
        for n, r in res[pop].items():
            print(" ", n, {k: (round(v["mean"], 3) if isinstance(v, dict) else round(v, 3)) for k, v in r.items()})


if __name__ == "__main__":
    main()
