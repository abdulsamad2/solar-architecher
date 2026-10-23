"""Post-hoc CRPS-estimator comparability on SkyGPT's external test (committed one-shot predictions; no inference).

Competitors (SkyGPT, PV-MM-Diffusion) score an empirical ensemble of m samples with the energy form
CRPS = E|X-y| - 0.5 E|X-X'| (V-statistic over the m samples; PV-MM-Diffusion reports m = 30). To compare like with
like, our 9 quantiles are turned into m samples by inverse-CDF sampling and scored the same way.
Variants: m in {10, 30}; tails beyond 5/95% (a) linear extrapolation, (b) clamped at the 5/95% quantiles,
(c) extrapolated x2; samples clipped at 0 kW (variant), 50 RNG seeds each; plus the fair (unbiased) estimator.
Output: reports/estimator_check.json
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from src.metrics import crps_quantile_interp

TAUS = np.array([.05, .1, .2, .3, .5, .7, .8, .9, .95])


def to_samples(q, n, rng, tail="linear", clip0=False):
    u = rng.uniform(0, 1, size=(len(q), n))
    k = {"linear": 1.0, "clamped": 0.0, "wide": 2.0}[tail]
    lo = q[:, 0] - k * (q[:, 1] - q[:, 0])
    hi = q[:, -1] + k * (q[:, -1] - q[:, -2])
    taus = np.concatenate([[0.0], TAUS, [1.0]])
    qq = np.maximum.accumulate(np.column_stack([lo, q, hi]), axis=1)
    # vectorised piecewise-linear inverse CDF
    j = np.clip(np.searchsorted(taus, u, side="right") - 1, 0, len(taus) - 2)
    t0, t1 = taus[j], taus[j + 1]
    r = np.arange(len(q))[:, None]
    x = qq[r, j] + (u - t0) / (t1 - t0) * (qq[r, j + 1] - qq[r, j])
    return np.maximum(x, 0.0) if clip0 else x


def crps_energy(samples, y, fair=False):
    m = samples.shape[1]
    t1 = np.abs(samples - y[:, None]).mean(1)
    t2 = np.abs(samples[:, :, None] - samples[:, None, :]).sum((1, 2))
    t2 = t2 / (2 * m * (m - 1)) if fair else t2 / (2 * m * m)
    return float((t1 - t2).mean())


def main():
    p = np.load("experiments/phaseA/a2_predictions.npz")
    with h5py.File("data/external/skygpt/test_set_2019nov_dec.hdf5", "r") as f:
        y = f["test/pv_pred"][:, 14].astype(float)
    out = {"note": __doc__, "published": {"PV-MM-Diffusion (m=30)": 2.63, "PV-MM-Diffusion (m=10)": 2.78, "SkyGPT (m=100)": 2.81},
           "models": {}}
    for name in ("G3''-F", "G4", "E6-S", "E4-B-fit"):
        q = p[f"q|{name}"]
        r = {"quantile_interp_crps": float(crps_quantile_interp(q, y, TAUS).mean())}
        for m in (10, 30):
            for tail in ("linear", "clamped", "wide"):
                for clip0 in (False, True):
                    for fair in (False, True):
                        vals = [crps_energy(to_samples(q, m, np.random.default_rng(seed), tail, clip0), y, fair)
                                for seed in range(50)]
                        key = f"m={m}|tail={tail}|clip0={clip0}|{'fair' if fair else 'V-stat (as competitors)'}"
                        r[key] = {"mean": float(np.mean(vals)), "sd_over_seeds": float(np.std(vals))}
        out["models"][name] = r
    sun = p["sunset_members"].T  # (N, 10)
    out["SUNSET port (10 members)"] = {"V-stat": crps_energy(sun, y), "fair": crps_energy(sun, y, fair=True)}
    Path("reports/estimator_check.json").write_text(json.dumps(out, indent=1))
    g = out["models"]["G3''-F"]
    print("G3''-F quantile-interp:", round(g["quantile_interp_crps"], 4))
    for k, v in g.items():
        if k != "quantile_interp_crps":
            print(f"  {k:55s} {v['mean']:.4f} (sd {v['sd_over_seeds']:.4f})")
    print("SUNSET port:", out["SUNSET port (10 members)"])


if __name__ == "__main__":
    main()
