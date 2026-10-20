import numpy as np
import pytest

from src.cal2 import conformal_q, interval_score, replay, select_param, widen

MIN = 60_000_000


def stream(n, sd_shift_at=None, seed=0):
    rng = np.random.default_rng(seed)
    origin = np.arange(n, dtype=np.int64) * MIN
    target = origin + 15 * MIN
    sd = np.where(np.arange(n) >= (sd_shift_at or n), 3.0, 1.0)
    y = rng.normal(0, sd)
    lo, hi = np.full(n, -1.645), np.full(n, 1.645)  # nominal 90% for sd 1
    return origin, target, lo, hi, y


def test_updates_are_causal_and_delayed_by_the_horizon():
    o, t, lo, hi, y = stream(200)
    r = replay(o, t, lo, hi, y, 0.1, "aci", 0.05, np.random.default_rng(1).normal(0, 1, 500))
    # forecast 0's outcome is observable at minute 15: forecasts 0..14 see no update, forecast 15 sees one
    assert (r["n_updates"][:15] == 0).all() and r["n_updates"][15] == 1 and r["n_updates"][16] == 2
    assert (r["last_obs_us"] <= o).all()
    r2 = replay(o, t, lo, hi, y, 0.1, "rolling", 100, np.random.default_rng(1).normal(0, 1, 100))
    assert (r2["n_updates"][:15] == 0).all()


def test_adaptation_recovers_coverage_after_shift_and_fixed_does_not():
    o, t, lo, hi, y = stream(20000, sd_shift_at=5000)
    init = np.maximum(-1.645 - np.random.default_rng(2).normal(0, 1, 2000),
                      np.random.default_rng(2).normal(0, 1, 2000) - 1.645)
    fixed = replay(o, t, lo, hi, y, 0.1, "aci", 0.0, init)
    aci = replay(o, t, lo, hi, y, 0.1, "aci", 0.01, init)
    roll = replay(o, t, lo, hi, y, 0.1, "rolling", 990, init)
    after = slice(8000, None)
    cov = lambda r: 1 - r["miss"][after].mean()
    assert cov(fixed) < 0.6
    # ACI is bounded by the calibration score range (clamped alpha, no windup) but still improves;
    # the rolling window absorbs the new scores and recovers nominal coverage
    assert cov(aci) > cov(fixed) + 0.1 and 0.0 <= np.nanmin(aci["alpha_t"]) and np.nanmax(aci["alpha_t"]) <= 1.0
    assert abs(cov(roll) - 0.9) < 0.03


def test_aci_recovers_after_shift_reverses_no_windup():
    o, t, lo, hi, y = stream(30000, sd_shift_at=5000)
    y = y.copy()
    rng = np.random.default_rng(7)
    y[15000:] = rng.normal(0, 1, 15000)  # shift ends
    init = np.maximum(-1.645 - rng.normal(0, 1, 2000), rng.normal(0, 1, 2000) - 1.645)
    aci = replay(o, t, lo, hi, y, 0.1, "aci", 0.01, init)
    assert abs((1 - aci["miss"][20000:].mean()) - 0.9) < 0.03


def test_width_floor_prevents_inversion_and_quantile_rank():
    a, b, deg = widen(0.0, 1.0, -5.0)
    assert deg and a == b == 0.5
    s = np.sort(np.arange(1, 11, dtype=float))
    assert conformal_q(s, 0.9) == 10.0 and conformal_q(s, 0.5) == 6.0 and conformal_q(s, -1) == 1.0


def test_selection_uses_only_calibration_inputs_and_prefers_adaptation_under_shift():
    o, t, lo, hi, y = stream(8000, sd_shift_at=6000)
    chosen, scores = select_param("aci", o, t, {"90": (lo, hi)}, y)
    assert chosen > 0 and scores[chosen] < scores[0.0]
    o2, t2, lo2, hi2, y2 = stream(8000)  # no shift: least adaptive should be competitive
    chosen2, s2 = select_param("aci", o2, t2, {"90": (lo2, hi2)}, y2)
    assert s2[chosen2] <= s2[0.0]


def test_interval_score_penalises_misses():
    assert interval_score([0.0], [1.0], [0.5], 0.1)[0] == pytest.approx(1.0)
    assert interval_score([0.0], [1.0], [2.0], 0.1)[0] == pytest.approx(1.0 + 20.0)
