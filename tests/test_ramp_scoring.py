import numpy as np

from src.metrics import crps_quantile_interp
from src.ramp_scoring import draw_outcomes, ramp_band, ramp_weighted_crps, selection_null

LEVELS = np.array([.05, .10, .25, .50, .75, .90, .95])
Z = np.array([-1.6449, -1.2816, -0.6745, 0.0, 0.6745, 1.2816, 1.6449])


def _forecasts(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = rng.uniform(5, 25, n), rng.uniform(0.5, 3.0, n)
    q = mu[:, None] + sd[:, None] * Z
    y = draw_outcomes(q, LEVELS, rng.random((n, 1)))[:, 0]
    lo, hi = ramp_band(mu / 30.0, 0.10, np.full(n, 30.0))
    return q, y, lo, hi


def test_parts_add_up_to_the_crps():
    q, y, lo, hi = _forecasts()
    crps, ramp, calm = ramp_weighted_crps(q, LEVELS, y, lo, hi)
    assert np.abs(crps - ramp - calm).max() < 1e-9
    assert np.abs(crps - crps_quantile_interp(q, y, LEVELS)).max() < 1e-9
    assert (ramp >= -1e-12).all() and (calm >= -1e-12).all()


def test_correct_forecaster_has_no_gap_but_low_ramp_coverage():
    q, y, lo, hi = _forecasts(20_000)
    r = selection_null(q, LEVELS, q[:, 0], q[:, -1], lo, hi, y=y, m=400, seed=1)
    assert r["null_ramp_coverage"] < 0.8  # far below the nominal 0.90 although the forecast is exactly correct
    assert abs(r["gap"]) < 0.03
