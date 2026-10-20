"""E7-BLEND: post-hoc gated mixture of experts (reports/e7_blend_prereg.md). CPU only, no retraining.

Ramp expert = E4-B-fit (direct K quantiles); calm expert = E6-S (residual event mixture). Quantile averaging
Q = w*Q_ramp + (1-w)*Q_calm, with w fitted on CALIBRATION days only:
  B0 constant w, B1 w = sigmoid(a + b*logit(g)) with b >= 0, B2 hard switch at g >= theta,
where g = calm expert's quantile-implied P(|dK15| >= tau). Selection uses a per-sample CRPS table on a
0.01 w-grid (linear interpolation in w); the 5 best candidates are then re-scored with exact CRPS.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.events import implied_event_probability
from src.metrics import crps_quantile_interp

W_GRID = np.round(np.arange(0, 1.0001, 0.01), 2)
A_GRID = np.round(np.arange(-8.0, 4.0001, 0.25), 2)
B_GRID = np.round(np.arange(0.0, 3.0001, 0.1), 1)
THETA_PCTS = np.arange(50, 100, 5).tolist() + [99]
C_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def gate_feature(p: dict, tau: float) -> np.ndarray:
    """Calm expert's quantile-implied P(|dK15| >= tau), in its own K space (forecast-time information only)."""
    qk = p["pred_q"] / np.maximum(p["pclear_target_kw"], 1.0)[:, None]
    kn = p["pv_now_kw"] / np.maximum(p["pclear_now_kw"], 1.0)
    out = np.empty(len(kn))
    for s in range(0, len(kn), 8192):
        out[s:s + 8192] = implied_event_probability(qk[s:s + 8192], p["quantile_levels"], kn[s:s + 8192], tau)
    return out


def blend(q_ramp, q_calm, w):
    w = np.asarray(w, float)
    w = w[:, None] if w.ndim == 1 else w
    return w * q_ramp + (1 - w) * q_calm


def gated_w(g, a, b):
    lg = np.log(np.clip(g, 1e-4, 1 - 1e-4) / (1 - np.clip(g, 1e-4, 1 - 1e-4)))
    return 1 / (1 + np.exp(-(a + b * lg)))


def crps_table(q_ramp, q_calm, y, levels) -> np.ndarray:
    """[N, len(W_GRID)] per-sample CRPS of the constant-w blend at each grid weight."""
    return np.stack([crps_quantile_interp(blend(q_ramp, q_calm, np.full(len(y), w)), y, levels) for w in W_GRID], 1)


def table_score(table, w) -> float:
    """Mean CRPS for per-sample weights w by linear interpolation in the w-grid."""
    pos = np.clip(w, 0, 1) * (len(W_GRID) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(W_GRID) - 1)
    f = pos - lo
    rows = np.arange(len(w))
    return float(((1 - f) * table[rows, lo] + f * table[rows, hi]).mean())


def fit(cal_ramp, cal_calm, g, y, levels) -> dict:
    t0 = time.perf_counter()
    table = crps_table(cal_ramp, cal_calm, y, levels)
    exact = lambda w: float(crps_quantile_interp(blend(cal_ramp, cal_calm, w), y, levels).mean())
    # B0: constant weight (grid points are exact table columns)
    col = {round(c, 2): int(round(c * 100)) for c in C_GRID}
    b0_scores = {c: float(table[:, col[c]].mean()) for c in C_GRID}
    c_best = min(C_GRID, key=lambda c: (round(b0_scores[c], 12), c))
    # B1: gated (table search over the full grid, then exact re-score of the 5 best)
    cand = sorted(((table_score(table, gated_w(g, a, b)), b, a) for a in A_GRID for b in B_GRID))[:5]
    rescored = sorted((exact(gated_w(g, a, b)), b, a) for _, b, a in cand)
    _, b_best, a_best = rescored[0]
    # B2: hard switch at calibration-day percentiles of g (w in {0,1}: exact table columns)
    thetas = {p: float(np.percentile(g, p)) for p in THETA_PCTS}
    b2_scores = {p: float(np.where(g >= th, table[:, -1], table[:, 0]).mean()) for p, th in thetas.items()}
    p_best = min(THETA_PCTS, key=lambda p: (round(b2_scores[p], 12), -p))
    return {"B0_w": float(c_best), "B1_a": float(a_best), "B1_b": float(b_best),
            "B2_theta": thetas[p_best], "B2_pct": int(p_best),
            "cal_crps": {"B0": b0_scores[c_best], "B1": rescored[0][0], "B2": b2_scores[p_best],
                         "ramp_expert": float(table[:, -1].mean()), "calm_expert": float(table[:, 0].mean())},
            "fit_seconds": time.perf_counter() - t0}


def apply(params, q_ramp, q_calm, g) -> dict:
    return {"B0": blend(q_ramp, q_calm, np.full(len(g), params["B0_w"])),
            "B1": blend(q_ramp, q_calm, gated_w(g, params["B1_a"], params["B1_b"])),
            "B2": blend(q_ramp, q_calm, (g >= params["B2_theta"]).astype(float))}
