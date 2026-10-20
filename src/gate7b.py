"""E7b: enhanced gate features for routing between the direct (E4-B-fit) and residual (E6-S) experts.

Protocol: reports/e7b_gate_prereg.md. Logistic gate w = sigmoid(b0 + beta.z), fitted by minimising the exact
blended CRPS (+ L2 on beta) on calibration days only, lambda chosen on a chronological calibration split.
"""
from __future__ import annotations

import numpy as np
import torch

from src.events import implied_event_probability
from src.metrics import quantile_function

N_GRID = 200
TAUS = (np.arange(N_GRID) + 0.5) / N_GRID
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1)
GATES = {"G0": ("g",), "G1": ("g", "s_D"), "G2": ("g", "s_D", "s_R"), "G3": ("g", "d"), "G4": ("g", "s_D", "s_R", "d")}


def k_quantiles(p) -> np.ndarray:
    return p["pred_q"] / np.maximum(p["pclear_target_kw"], 1.0)[:, None]


def raw_features(direct: dict, residual: dict, tau: float) -> dict:
    """Untransformed forecast-time features (both experts on the same samples)."""
    lv = direct["quantile_levels"]
    qd, qr = k_quantiles(direct), k_quantiles(residual)
    kn = residual["pv_now_kw"] / np.maximum(residual["pclear_now_kw"], 1.0)
    g = np.concatenate([implied_event_probability(qr[s:s + 8192], lv, kn[s:s + 8192], tau) for s in range(0, len(kn), 8192)])
    i10, i90 = list(lv).index(0.10), list(lv).index(0.90)
    mean_d = quantile_function(qd, lv, TAUS).mean(axis=1)
    mean_r = quantile_function(qr, lv, TAUS).mean(axis=1)
    return {"g": g, "s_D": qd[:, i90] - qd[:, i10], "s_R": qr[:, i90] - qr[:, i10], "d": np.abs(mean_d - mean_r)}


SIGNED = {"inflow"}  # E8 G5: signed features are standardised without a log transform


def transform(raw: dict, names) -> np.ndarray:
    cols = []
    for n in names:
        x = raw[n]
        if n == "g":
            cols.append(np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4))))
        elif n in SIGNED:
            cols.append(np.asarray(x, float))
        else:
            cols.append(np.log(np.maximum(x, 0) + 1e-3))
    return np.stack(cols, 1)


def grid_quantiles(q, levels) -> np.ndarray:
    """Piecewise-linear quantile function on the CRPS midpoint grid (constant tails), [N, N_GRID]."""
    return quantile_function(q, levels, TAUS)


class BlendCRPS:
    """Exact blended CRPS: blended grid quantiles are linear in w (w*QD_grid + (1-w)*QR_grid)."""

    def __init__(self, qd_grid, qr_grid, y):
        self.qd = torch.as_tensor(qd_grid, dtype=torch.float64)
        self.qr = torch.as_tensor(qr_grid, dtype=torch.float64)
        self.y = torch.as_tensor(y, dtype=torch.float64)[:, None]
        self.t = torch.as_tensor(TAUS, dtype=torch.float64)[None, :]

    def per_sample(self, w):
        qb = w[:, None] * self.qd + (1 - w[:, None]) * self.qr
        err = self.y - qb
        return 2 * torch.maximum(self.t * err, (self.t - 1) * err).mean(dim=1)


def fit_gate(z, crps: BlendCRPS, lam: float, max_iter: int = 200):
    """Minimise mean CRPS + lam * ||beta||^2 over (b0, beta) with L-BFGS from zero (deterministic)."""
    zt = torch.as_tensor(z, dtype=torch.float64)
    b0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    beta = torch.zeros(zt.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([b0, beta], max_iter=max_iter, tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        w = torch.sigmoid(b0 + zt @ beta)
        loss = crps.per_sample(w).mean() + lam * (beta ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return float(b0.item()), beta.detach().numpy().copy()


def gate_weights(z, b0, beta) -> np.ndarray:
    return 1 / (1 + np.exp(-(b0 + z @ beta)))


def standardize(train_z, *others):
    mu, sd = train_z.mean(0), train_z.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return [(x - mu) / sd for x in (train_z,) + others], (mu, sd)


def select_and_fit(raw_cal: dict, names, qd_cal, qr_cal, y_cal, levels, order) -> dict:
    """Lambda on a chronological half split of calibration days, then refit on all calibration days."""
    half = len(order) // 2
    a, b = order[:half], order[half:]
    zc = transform(raw_cal, names)
    gd, gr = grid_quantiles(qd_cal, levels), grid_quantiles(qr_cal, levels)
    scores = {}
    for lam in LAMBDAS:
        (za, zb), _ = standardize(zc[a], zc[b])
        b0, beta = fit_gate(za, BlendCRPS(gd[a], gr[a], y_cal[a]), lam)
        w = torch.as_tensor(gate_weights(zb, b0, beta))
        scores[lam] = float(BlendCRPS(gd[b], gr[b], y_cal[b]).per_sample(w).mean())
    best = min(scores.values())
    lam = max(l for l in LAMBDAS if scores[l] <= best + 1e-12)
    (zall,), (mu, sd) = standardize(zc)
    full = BlendCRPS(gd, gr, y_cal)
    b0, beta = fit_gate(zall, full, lam)
    return {"names": list(names), "lambda": lam, "lambda_scores": scores, "b0": b0, "beta": beta.tolist(),
            "mu": mu.tolist(), "sd": sd.tolist()}


def apply_gate(params, raw: dict) -> np.ndarray:
    z = (transform(raw, params["names"]) - np.array(params["mu"])) / np.array(params["sd"])
    return gate_weights(z, params["b0"], np.array(params["beta"]))


def average_precision(score, label) -> float:
    order = np.argsort(-score, kind="stable")
    lab = np.asarray(label, bool)[order]
    tp = np.cumsum(lab)
    precision = tp / np.arange(1, len(lab) + 1)
    return float(precision[lab].sum() / max(lab.sum(), 1))
