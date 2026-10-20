from statistics import NormalDist

import numpy as np

from src.calibrate import base_interval, cqr_apply, cqr_fit, extended_quantile_function, recal_apply, recal_fit
from src.metrics import crps_quantile_interp, quantile_reliability

LV = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
Z = np.array([NormalDist().inv_cdf(q) for q in LV])


def overconfident(n, seed, forecast_sd=0.8):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(5, 25, n)
    y = mu + rng.normal(0, 2.0, n)                   # truth sd 2
    return mu[:, None] + forecast_sd * Z[None, :], y  # forecast sd < 2 -> over-confident


def test_cqr_restores_marginal_coverage():
    q_cal, y_cal = overconfident(20000, 0)
    q_val, y_val = overconfident(20000, 1)
    for name, nominal in (("50", 0.5), ("80", 0.8), ("90", 0.9)):
        lo, hi = base_interval(q_cal, LV, name)
        q_hat = cqr_fit(lo, hi, y_cal, nominal)
        lo_v, hi_v = cqr_apply(*base_interval(q_val, LV, name), q_hat)
        cov = np.mean((y_val >= lo_v) & (y_val <= hi_v))
        assert abs(cov - nominal) < 0.015, (name, cov)


def test_recalibration_fixes_reliability_improves_crps_and_never_crosses():
    # mild over-confidence (forecast sd 1.6 vs truth 2.0), comparable to E4-B's 85% at nominal 90%
    q_cal, y_cal = overconfident(20000, 2, forecast_sd=1.6)
    q_val, y_val = overconfident(20000, 3, forecast_sd=1.6)
    cal = recal_fit(q_cal, y_cal, LV)
    assert (np.diff(cal["mapped_levels"]) >= 0).all() and not cal["saturated_levels"]
    q_new = recal_apply(q_val, LV, cal)
    assert (np.diff(q_new, axis=1) >= 0).all()
    assert np.allclose(quantile_reliability(q_new, y_val), LV, atol=0.015)
    assert crps_quantile_interp(q_new, y_val, LV).mean() < crps_quantile_interp(q_val, y_val, LV).mean()


def test_recalibration_saturates_on_severe_overconfidence_and_says_so():
    # forecast sd 0.8 vs truth 2.0: tails cannot be widened by level remapping alone
    q_cal, y_cal = overconfident(20000, 4)
    cal = recal_fit(q_cal, y_cal, LV)
    assert 0.05 in cal["saturated_levels"] and 0.95 in cal["saturated_levels"]
    assert (np.diff(recal_apply(q_cal, LV, cal), axis=1) >= 0).all()  # still never crosses


def test_extended_quantile_function_is_exact_for_gaussian_tails_and_floors():
    q = 10.0 + 2.0 * Z[None, :]
    taus = [0.001, 0.01, 0.05, 0.99, 0.999]
    out = extended_quantile_function(q, LV, taus)
    assert np.allclose(out[0], [10.0 + 2.0 * NormalDist().inv_cdf(t) for t in taus], atol=1e-9)
    low = extended_quantile_function(0.1 + 2.0 * Z[None, :] * 0.01, LV, [0.0001])
    assert low[0, 0] >= 0.0
