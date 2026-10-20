"""Synthetic scoring tests for E7b (run before any real E7b result)."""
from statistics import NormalDist

import numpy as np
import torch

from src.gate7b import (BlendCRPS, apply_gate, average_precision, fit_gate, grid_quantiles, select_and_fit,
                        standardize, transform)
from src.metrics import crps_quantile_interp

LV = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
Z = np.array([NormalDist().inv_cdf(q) for q in LV])


def world(n, seed, spread_informative=True):
    """Ramps: direct expert right; calm: residual expert right. g is a weak gate; s_D carries extra signal."""
    rng = np.random.default_rng(seed)
    ramp = rng.random(n) < 0.12
    y = np.where(ramp, rng.normal(-4, 1.5, n), rng.normal(0, 0.3, n))
    qd = np.where(ramp[:, None], -4 + 1.5 * Z, 0.0 + 1.2 * Z)
    qr = np.where(ramp[:, None], 0.0 + 0.3 * Z, 0.0 + 0.3 * Z)
    g = np.clip(np.where(ramp, 0.45, 0.2) + rng.normal(0, 0.12, n), 0.01, 0.99)  # weak
    s_d = np.where(ramp, 2.0, 1.0) if spread_informative else np.full(n, 1.5)
    s_d = s_d * np.exp(rng.normal(0, 0.1, n))
    raw = {"g": g, "s_D": s_d, "s_R": np.full(n, 0.6) * np.exp(rng.normal(0, 0.1, n)), "d": np.abs(rng.normal(0, 1, n))}
    return raw, qd, qr, y, ramp


def test_torch_crps_matches_reference_scorer():
    raw, qd, qr, y, _ = world(2000, 0)
    w = np.random.default_rng(1).random(2000)
    c = BlendCRPS(grid_quantiles(qd, LV), grid_quantiles(qr, LV), y)
    torch_v = c.per_sample(torch.as_tensor(w)).numpy()
    ref = crps_quantile_interp(w[:, None] * qd + (1 - w[:, None]) * qr, y, LV)
    assert np.abs(torch_v - ref).max() < 1e-8


def test_informative_spread_feature_improves_ramp_crps_over_g_only():
    raw, qd, qr, y, ramp = world(8000, 2)
    order = np.arange(len(y))
    p0 = select_and_fit(raw, ("g",), qd, qr, y, LV, order)
    p1 = select_and_fit(raw, ("g", "s_D"), qd, qr, y, LV, order)
    raw2, qd2, qr2, y2, ramp2 = world(8000, 3)
    c = lambda p: crps_quantile_interp(apply_gate(p, raw2)[:, None] * qd2 + (1 - apply_gate(p, raw2)[:, None]) * qr2, y2, LV)
    c0, c1 = c(p0), c(p1)
    assert c1[ramp2].mean() < c0[ramp2].mean() and c1.mean() < c0.mean()


def test_uninformative_feature_does_not_hurt_much():
    raw, qd, qr, y, _ = world(8000, 4, spread_informative=False)
    order = np.arange(len(y))
    p0 = select_and_fit(raw, ("g",), qd, qr, y, LV, order)
    p1 = select_and_fit(raw, ("g", "s_D"), qd, qr, y, LV, order)
    raw2, qd2, qr2, y2, _ = world(8000, 5, spread_informative=False)
    c = lambda p: crps_quantile_interp(apply_gate(p, raw2)[:, None] * qd2 + (1 - apply_gate(p, raw2)[:, None]) * qr2, y2, LV).mean()
    assert c(p1) <= c(p0) * 1.01


def test_standardization_uses_fit_set_only_and_fit_is_deterministic():
    a = np.array([[1.0], [3.0]])
    (za, zb), (mu, sd) = standardize(a, np.array([[100.0]]))
    assert mu[0] == 2.0 and sd[0] == 1.0 and zb[0, 0] == 98.0
    raw, qd, qr, y, _ = world(3000, 6)
    z = transform(raw, ("g", "s_D"))
    c = BlendCRPS(grid_quantiles(qd, LV), grid_quantiles(qr, LV), y)
    assert fit_gate(z, c, 1e-3)[0] == fit_gate(z, c, 1e-3)[0]


def test_average_precision():
    assert average_precision(np.array([0.9, 0.8, 0.1]), np.array([1, 0, 1])) == (1.0 + 2 / 3) / 2
    assert average_precision(np.array([0.9, 0.1]), np.array([1, 0])) == 1.0
