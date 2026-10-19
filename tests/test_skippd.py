import datetime as dt

import numpy as np
import pytest

from src.skippd import official_forecast_samples

T0 = dt.datetime(2018, 5, 1, 9, 0, 0)


def minutes(n, start=T0, phase_s=0):
    return [start + dt.timedelta(minutes=k, seconds=phase_s) for k in range(n)]


def test_exact_rule_stack_target_and_two_minute_gate():
    times = minutes(60)
    s = official_forecast_samples(times)
    # first origin needs 15 minutes of history; last needs 15 minutes of future
    assert s.origin[0] == 15 and s.origin[-1] <= 44
    assert (np.diff(s.origin) == 2).all()
    assert s.history.shape[1] == 16
    assert (s.history[:, 0] == s.origin).all()               # newest first
    assert (s.history == s.origin[:, None] - np.arange(16)).all()
    assert (s.target == s.origin + 15).all()


def test_phase_shift_breaks_exact_but_not_tolerant_matching():
    # second half of the timeline drifts by 20 s, like the benchmark's second-phase changes
    times = minutes(30) + minutes(40, start=T0 + dt.timedelta(minutes=30), phase_s=20)
    exact = official_forecast_samples(times)
    tol = official_forecast_samples(times, tolerance_s=30)
    assert len(tol.origin) > len(exact.origin)
    assert (np.diff(tol.history, axis=1) < 0).all()          # distinct, ordered frames
    assert (tol.target > tol.origin).all()


def test_gaps_drop_samples_and_bad_inputs_rejected():
    times = minutes(60)
    del times[30]
    s = official_forecast_samples(times)
    t = np.array(times)
    needed = np.concatenate([s.history, s.target[:, None]], 1)
    span_ok = [(t[n].max() - t[n].min()) == dt.timedelta(minutes=30) for n in needed]
    assert all(span_ok)
    with pytest.raises(ValueError):
        official_forecast_samples(times[::-1])
    with pytest.raises(ValueError):
        official_forecast_samples(times, tolerance_s=30.5)
