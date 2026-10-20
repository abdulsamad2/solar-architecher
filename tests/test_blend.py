from statistics import NormalDist

import numpy as np

from src.blend import apply, blend, crps_table, fit, gated_w, table_score
from src.metrics import crps_quantile_interp

LV = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
Z = np.array([NormalDist().inv_cdf(q) for q in LV])


def regimes(n, seed):
    """Calm samples: calm expert exact; ramp samples: ramp expert exact. g is an informative gate."""
    rng = np.random.default_rng(seed)
    ramp = rng.random(n) < 0.15
    y = np.where(ramp, rng.normal(-5, 2, n), rng.normal(0, 0.3, n))
    q_calm = np.where(ramp[:, None], 0.0 + 0.3 * Z, 0.0 + 0.3 * Z)               # always predicts calm
    q_ramp = np.where(ramp[:, None], -5 + 2 * Z, 0.0 + 1.5 * Z)                  # good on ramps, wide when calm
    g = np.clip(np.where(ramp, 0.8, 0.1) + rng.normal(0, 0.05, n), 0.01, 0.99)
    return q_ramp, q_calm, y, g, ramp


def test_gated_blend_beats_both_experts_and_constant_blend():
    qr, qc, y, g, _ = regimes(6000, 0)
    p = fit(qr, qc, g, y, LV)
    assert p["B1_b"] > 0
    qr2, qc2, y2, g2, ramp2 = regimes(6000, 1)
    out = apply(p, qr2, qc2, g2)
    c = lambda q, m=slice(None): crps_quantile_interp(q[m], y2[m], LV).mean()
    assert c(out["B1"]) < min(c(qr2), c(qc2), c(out["B0"]))
    assert c(out["B1"], ramp2) < c(out["B0"], ramp2)
    for q in out.values():
        assert np.isfinite(q).all() and (np.diff(q, axis=1) >= -1e-12).all()


def test_table_interpolation_is_close_to_exact():
    qr, qc, y, g, _ = regimes(3000, 2)
    t = crps_table(qr, qc, y, LV)
    w = gated_w(g, -1.0, 1.3)
    exact = crps_quantile_interp(blend(qr, qc, w), y, LV).mean()
    assert abs(table_score(t, w) - exact) < 0.01 * exact


def test_b1_with_zero_slope_equals_constant_blend():
    w = gated_w(np.array([0.1, 0.5, 0.9]), 0.0, 0.0)
    assert np.allclose(w, 0.5)
