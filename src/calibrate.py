"""Post-hoc calibration of quantile forecasts (Gate 2; protocol: reports/gate2_calibration_prereg.md).

Everything is fitted on calibration-day predictions only and applied to fold-validation predictions.
Split-conformal (CQR) gives at most an approximate marginal guarantee under exchangeability; no
conditional coverage guarantee is implied.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.metrics import quantile_function

INTERVALS = {"50": (0.25, 0.75), "80": (0.10, 0.90), "90": (0.05, 0.95)}


def base_interval(pred_q, levels, name: str):
    lo_t, hi_t = INTERVALS[name]
    return (quantile_function(pred_q, levels, [lo_t])[:, 0], quantile_function(pred_q, levels, [hi_t])[:, 0])


def cqr_fit(lo, hi, y, nominal: float, scale=None) -> float:
    """Split-conformal correction q_hat: the ceil((n+1)*nominal)-th smallest conformity score."""
    s = np.maximum(np.asarray(lo) - y, np.asarray(y) - hi)
    if scale is not None:
        s = s / scale
    n = len(s)
    k = min(int(np.ceil((n + 1) * nominal)), n)
    return float(np.sort(s)[k - 1])


def cqr_apply(lo, hi, q_hat: float, scale=None):
    w = q_hat if scale is None else q_hat * np.asarray(scale)
    return np.asarray(lo) - w, np.asarray(hi) + w


def _z(t):
    from scipy.special import ndtri

    return ndtri(np.asarray(t, float))


def extended_quantile_function(pred_q, levels: Sequence[float], taus, floor: float = 0.0) -> np.ndarray:
    """Linear in tau between the levels (as the CRPS scorer); beyond the end levels, linear in the
    normal score z = Phi^-1(tau) with the outermost segment's z-slope (exact for Gaussian tails;
    pre-registration amendment 1). Floored at ``floor``. pred_q [N, Q] -> [N, len(taus)]."""
    pred_q, levels, taus = np.asarray(pred_q, float), np.asarray(levels, float), np.asarray(taus, float)
    inner = quantile_function(pred_q, levels, taus)
    zl = _z(levels)
    lo_slope = (pred_q[:, 1] - pred_q[:, 0]) / (zl[1] - zl[0])
    hi_slope = (pred_q[:, -1] - pred_q[:, -2]) / (zl[-1] - zl[-2])
    below, above = taus < levels[0], taus > levels[-1]
    out = inner.copy()
    out[:, below] = pred_q[:, :1] + lo_slope[:, None] * (_z(taus[below]) - zl[0])[None, :]
    out[:, above] = pred_q[:, -1:] + hi_slope[:, None] * (_z(taus[above]) - zl[-1])[None, :]
    return np.maximum(out, floor)


def recal_fit(pred_q, y, levels: Sequence[float], grid_step: float = 0.001, chunk: int = 4096) -> dict:
    """Monotone calibration map nominal level -> forecast level, from calibration data."""
    grid = np.round(np.arange(grid_step, 1.0, grid_step), 6)
    counts = np.zeros(len(grid))
    y = np.asarray(y, float)
    for s in range(0, len(y), chunk):
        qf = extended_quantile_function(pred_q[s:s + chunk], levels, grid)
        counts += (y[s:s + chunk, None] <= qf).sum(axis=0)
    f_hat = np.maximum.accumulate(counts / len(y))
    target = np.asarray(levels, float)
    idx = np.minimum(np.searchsorted(f_hat, target, side="left"), len(grid) - 1)
    mapped = grid[idx]
    # level remapping cannot widen beyond the extrapolated support: flag levels pinned at the grid edges
    saturated = (mapped <= grid[0]) | (mapped >= grid[-1])
    return {"grid": grid, "f_hat": f_hat, "levels": target, "mapped_levels": mapped,
            "saturated_levels": target[saturated].tolist()}


def recal_apply(pred_q, levels: Sequence[float], cal: dict) -> np.ndarray:
    out = extended_quantile_function(pred_q, levels, cal["mapped_levels"])
    if (np.diff(out, axis=1) < -1e-9).any():
        raise RuntimeError("STOP: recalibrated quantiles cross")
    return out
