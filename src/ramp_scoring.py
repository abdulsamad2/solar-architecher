"""Ramp evaluation without outcome-selection bias, for any quantile forecast (numpy only).

A ramp is defined by the outcome (|k_target - k_now| >= tau), so scores averaged over observed ramps are biased
(forecaster's dilemma, Lerch et al. 2017). Two remedies, as used in revision R-Z1 and R-Z2:

- ramp_weighted_crps: CRPS = ramp-region part + calm-region part (threshold-weighted CRPS, Gneiting & Ranjan 2011),
  with the ramp region of the outcome space known at issue time. Proper, so it can rank forecasters on ramps.
- selection_null: the coverage that intervals would have on ramp outcomes if the forecast distribution were exactly
  correct. Compare observed ramp coverage with this value, not with the nominal level.

Example: python -m src.ramp_scoring
"""
from __future__ import annotations

import numpy as np

from src.metrics import quantile_function


def ramp_band(k_now, tau, p_clear):
    """Non-ramp band (lo, hi) in power units: an outcome z is a ramp iff z <= lo or z >= hi."""
    k_now, p_clear = np.asarray(k_now, float), np.asarray(p_clear, float)
    return (k_now - tau) * p_clear, (k_now + tau) * p_clear


def ramp_weighted_crps(pred_q, levels, y, lo, hi, n_grid=200):
    """Per-forecast (CRPS, ramp-region part, calm-region part); the two parts add up to the CRPS exactly.

    pred_q [n, Q] quantile forecasts at `levels`; y [n] outcomes; lo, hi [n] the non-ramp band of each forecast."""
    y, lo, hi = (np.asarray(v, float) for v in (y, lo, hi))
    taus = (np.arange(n_grid) + 0.5) / n_grid
    qf = quantile_function(pred_q, levels, taus)
    pin = lambda a, b: 2 * np.maximum(taus * (a[:, None] - b), (taus - 1) * (a[:, None] - b)).mean(1)
    calm = lambda z: np.clip(z, lo[:, None] if z.ndim == 2 else lo, hi[:, None] if z.ndim == 2 else hi)
    ramp = lambda z: z - calm(z)  # chaining function of the ramp weight, up to a constant that cancels in the score
    return pin(y, qf), pin(ramp(y), ramp(qf)), pin(calm(y), calm(qf))


def draw_outcomes(nodes, node_levels, u, cap=None, tail="exp"):
    """Outcomes with uniform ranks u [n, m] from the distribution with quantiles `nodes` [n, P] at `node_levels`:
    linear between nodes, exponential ("exp") or linear ("lin") tails scaled by the outermost node spacing."""
    p, v = np.asarray(node_levels, float), np.sort(np.asarray(nodes, float), 1)
    j = np.clip(np.searchsorted(p, u, side="right") - 1, 0, len(p) - 2)
    w = (np.clip(u, p[0], p[-1]) - p[j]) / (p[j + 1] - p[j])
    out = np.take_along_axis(v, j, 1) * (1 - w) + np.take_along_axis(v, j + 1, 1) * w
    s_lo, s_hi = np.maximum(v[:, 1] - v[:, 0], 1e-6)[:, None], np.maximum(v[:, -1] - v[:, -2], 1e-6)[:, None]
    t_lo, t_hi = np.clip(u / p[0], 1e-300, 1.0), np.clip((1 - u) / (1 - p[-1]), 1e-300, 1.0)
    e_lo, e_hi = (-np.log(t_lo), -np.log(t_hi)) if tail == "exp" else (1 - t_lo, 1 - t_hi)
    out = np.where(u < p[0], v[:, :1] - s_lo * e_lo, out)
    out = np.where(u > p[-1], v[:, -1:] + s_hi * e_hi, out)
    return np.clip(out, 0.0, np.inf if cap is None else np.asarray(cap, float)[:, None])


def selection_null(nodes, node_levels, lower, upper, lo, hi, y=None, cap=None, tail="exp", m=2000, chunk=100, seed=0):
    """Coverage of the intervals [lower, upper] on ramp outcomes when outcomes are drawn from the forecast itself.

    Returns the null ramp coverage, its 95% range over simulated data sets, the null ramp rate and, when y is given,
    the observed ramp coverage and the gap (observed - null)."""
    lower, upper, lo, hi = (np.asarray(v, float) for v in (lower, upper, lo, hi))
    rng = np.random.default_rng(seed)
    n = len(lower)
    num = den = 0.0
    rep_n, rep_c = [], []
    for _ in range(m // chunk):
        ys = draw_outcomes(nodes, node_levels, rng.random((n, chunk)), cap, tail)
        ev = (ys <= lo[:, None]) | (ys >= hi[:, None])
        ec = ev & (ys >= lower[:, None]) & (ys <= upper[:, None])
        num, den = num + ec.sum(), den + ev.sum()
        rep_n.append(ev.sum(0)); rep_c.append(ec.sum(0))
    rn, rc = np.concatenate(rep_n), np.concatenate(rep_c)
    res = {"null_ramp_coverage": float(num / den), "null_ramp_rate": float(den / (n * (m // chunk) * chunk)),
           "null_ramp_coverage_range": [float(x) for x in np.quantile(rc / np.maximum(rn, 1), [0.025, 0.975])]}
    if y is not None:
        y = np.asarray(y, float)
        r = (y <= lo) | (y >= hi)
        res["observed_ramp_coverage"] = float(((y >= lower) & (y <= upper))[r].mean())
        res["gap"] = res["observed_ramp_coverage"] - res["null_ramp_coverage"]
    return res


def _example():
    """Synthetic check: a forecaster that is exactly correct still covers fewer than 90% of ramp outcomes."""
    rng = np.random.default_rng(1)
    n, levels = 20_000, np.array([.05, .10, .25, .50, .75, .90, .95])
    from statistics import NormalDist
    z = np.array([NormalDist().inv_cdf(p) for p in levels])
    mu, sd = rng.uniform(5, 25, n), rng.uniform(0.5, 3.0, n)
    pred_q = mu[:, None] + sd[:, None] * z
    y = np.clip(draw_outcomes(pred_q, levels, rng.random((n, 1)))[:, 0], 0, None)  # outcomes from the forecast itself
    p_clear = np.full(n, 30.0)
    lo, hi = ramp_band(mu / p_clear, 0.10, p_clear)  # ramp: a change of at least 0.10 in clear-sky index
    crps, tw_ramp, tw_calm = ramp_weighted_crps(pred_q, levels, y, lo, hi)
    assert np.abs(crps - tw_ramp - tw_calm).max() < 1e-9
    res = selection_null(pred_q, levels, pred_q[:, 0], pred_q[:, -1], lo, hi, y=y, seed=2)
    ramp = (y <= lo) | (y >= hi)
    print(f"correct forecaster, 90% intervals: overall coverage {np.mean((y >= pred_q[:, 0]) & (y <= pred_q[:, -1])):.3f}, "
          f"coverage on ramp outcomes {res['observed_ramp_coverage']:.3f} (ramp rate {ramp.mean():.3f})")
    print(f"selection null for ramp outcomes {res['null_ramp_coverage']:.3f} {np.round(res['null_ramp_coverage_range'], 3).tolist()}, gap {res['gap']:+.3f}")
    print(f"CRPS {crps.mean():.4f} = ramp region {tw_ramp.mean():.4f} + calm region {tw_calm.mean():.4f}")


if __name__ == "__main__":
    _example()
