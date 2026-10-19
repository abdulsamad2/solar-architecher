"""Deterministic and probabilistic scores, plus a day-level paired bootstrap."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def rmse(pred, target) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(target)) ** 2)))


def mae(pred, target) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(target))))


def skill_score(model_score: float, reference_score: float) -> float:
    """1 - model/reference (e.g. RMSE skill vs smart persistence)."""
    return 1.0 - model_score / reference_score


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: Sequence[float]) -> torch.Tensor:
    """pred [..., Q], target [...]; mean over all elements and quantiles."""
    q = torch.as_tensor(quantiles, device=pred.device, dtype=pred.dtype)
    err = target.unsqueeze(-1) - pred
    return torch.maximum(q * err, (q - 1) * err).mean()


def crps_ensemble(samples, obs) -> np.ndarray:
    """Exact CRPS of the empirical distribution: E|X-y| - 0.5 E|X-X'|.

    samples [N, M], obs [N] -> [N]. A single member (M=1) gives |x - y|.
    Uses the sorted-sample identity, O(M log M) per row.
    """
    x = np.sort(np.asarray(samples, dtype=float), axis=-1)
    y = np.asarray(obs, dtype=float)[..., None]
    m = x.shape[-1]
    term1 = np.abs(x - y).mean(-1)
    w = 2 * np.arange(1, m + 1) - m - 1
    term2 = (w * x).sum(-1) / (m * m)
    return term1 - term2


def crps_from_quantiles(pred_q, obs, quantiles: Sequence[float]) -> np.ndarray:
    """Approximate CRPS as 2 * mean pinball over a discrete quantile set.

    This is a documented approximation (quadrature over q); it is not identical to
    exact CRPS and should be reported as such.
    """
    q = np.asarray(quantiles)
    err = np.asarray(obs)[..., None] - np.asarray(pred_q)
    return 2 * np.maximum(q * err, (q - 1) * err).mean(-1)


def quantile_function(pred_q, levels: Sequence[float], taus) -> np.ndarray:
    """Piecewise-linear quantile function through (levels, pred_q), constant beyond the end levels.

    pred_q [N, Q] (non-decreasing along Q) -> [N, len(taus)].
    """
    pred_q = np.asarray(pred_q, float)
    levels = np.asarray(levels, float)
    taus = np.clip(np.asarray(taus, float), levels[0], levels[-1])
    j = np.clip(np.searchsorted(levels, taus, side="right") - 1, 0, len(levels) - 2)
    w = (taus - levels[j]) / (levels[j + 1] - levels[j])
    return pred_q[:, j] * (1 - w) + pred_q[:, j + 1] * w


def crps_quantile_interp(pred_q, obs, levels: Sequence[float], n_grid: int = 200) -> np.ndarray:
    """CRPS of the distribution whose quantile function is ``quantile_function(pred_q, levels)``.

    CRPS = 2 * int_0^1 pinball_tau(y, Q(tau)) dtau, by the midpoint rule on ``n_grid`` levels.
    A point forecast (all quantiles equal) gives exactly |y - x|, so deterministic models are scored on
    the same scale. Tails beyond the outermost levels are held constant (a documented approximation).
    """
    taus = (np.arange(n_grid) + 0.5) / n_grid
    qf = quantile_function(pred_q, levels, taus)
    err = np.asarray(obs, float)[:, None] - qf
    return 2 * np.maximum(taus * err, (taus - 1) * err).mean(axis=1)


def quantile_reliability(pred_q, obs) -> np.ndarray:
    """Observed frequency of y <= Q_tau for each predicted level (ideal: equals tau)."""
    return (np.asarray(obs, float)[:, None] <= np.asarray(pred_q, float)).mean(axis=0)


def interval_coverage(lower, upper, obs) -> float:
    obs = np.asarray(obs)
    return float(np.mean((obs >= np.asarray(lower)) & (obs <= np.asarray(upper))))


def interval_width(lower, upper) -> float:
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def day_bootstrap_diff(
    loss_a, loss_b, day, strata=None, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Paired day-block bootstrap CI for mean(loss_a - loss_b).

    Per-sample losses are averaged per day first; days (not minutes) are resampled,
    within strata (e.g. sunny/cloudy) when ``strata`` is given (one label per sample).
    Returns (point_estimate, lower, upper).
    """
    diff = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    day = np.asarray(day)
    days, inv = np.unique(day, return_inverse=True)
    per_day = np.bincount(inv, weights=diff) / np.bincount(inv)
    if strata is None:
        day_strata = np.zeros(len(days), dtype=int)
    else:
        strata = np.asarray(strata)
        day_strata = np.array([strata[inv == k][0] for k in range(len(days))])

    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(day_strata == s) for s in np.unique(day_strata)]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        boots[b] = per_day[idx].mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(per_day.mean()), float(lo), float(hi)
