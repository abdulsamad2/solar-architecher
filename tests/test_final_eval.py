"""Synthetic fixtures for the final evaluation pipeline (no real test data)."""
import numpy as np
import pytest

from src.final_eval import day_bootstrap, ensemble, nearest_init, rolling_conformal_daily, run_test

MIN = 60_000_000


def test_ensemble_is_levelwise_mean_and_monotone():
    a = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8.]])
    b = a + 2
    assert np.allclose(ensemble([a, b]), a + 1)
    with pytest.raises(AssertionError):
        ensemble([a[:, ::-1]])


def test_rolling_daily_reset_causality_and_no_carryover():
    rng = np.random.default_rng(0)
    n_day = 120
    day = np.repeat(["d1", "d2"], n_day)
    origin = np.concatenate([np.arange(n_day) * MIN, 10_000 * MIN + np.arange(n_day) * MIN]).astype(np.int64)
    target = origin + 15 * MIN
    y = rng.normal(0, 1, 2 * n_day)
    y[n_day:] *= 5  # day 2 is much more variable
    lo, hi = np.full(2 * n_day, -1.6), np.full(2 * n_day, 1.6)
    inits = {"d1": np.zeros(50), "d2": np.zeros(50)}
    clo, chi = rolling_conformal_daily(origin, target, day, lo, hi, y, 0.1, 50, inits)
    # first 15 forecasts of each day see only the initial window (no immature outcomes) -> identical widths
    assert np.allclose(chi[:15] - clo[:15], 3.2) and np.allclose(chi[n_day:n_day + 15] - clo[n_day:n_day + 15], 3.2)
    # adaptation within day 2 widens intervals later in the day
    assert (chi[-1] - clo[-1]) > 3.2
    # no carry-over: day 2 starts exactly from its own initial window despite day 1's history
    assert np.isclose(chi[n_day] - clo[n_day], 3.2)


def test_nearest_init_uses_closest_development_scores_chronologically():
    org = np.array([0, 10, 20, 30, 40], dtype=np.int64)
    sc = np.array([1., 2., 3., 4., 5.])
    assert nearest_init(org, sc, 22, 2).tolist() == [3.0, 4.0] or nearest_init(org, sc, 22, 2).tolist() == [2.0, 3.0]
    assert nearest_init(org, sc, 22, 3).tolist() == [2.0, 3.0, 4.0]


def test_day_bootstrap_ci_contains_point_and_is_deterministic():
    rng = np.random.default_rng(1)
    day = np.repeat(np.arange(20).astype(str), 30)
    sunny = np.repeat(np.arange(20) < 10, 30)
    diff = rng.normal(-0.05, 0.1, len(day))
    r1 = day_bootstrap(diff, day, sunny, n_boot=2000)
    assert r1 == day_bootstrap(diff, day, sunny, n_boot=2000) and r1[1] <= r1[0] <= r1[2]


def test_one_shot_runner_refuses_without_recorded_approval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(AssertionError, match="explicit approval"):
        run_test("anything")
