import numpy as np
import pandas as pd
import pytest

from src.clear_sky import ClearSkyPV, clear_sky_poa


def _times(months):
    days = [pd.Timestamp(2018, m, 15) for m in months]
    return np.array([d + pd.Timedelta(minutes=k) for d in days for k in range(7 * 60, 17 * 60, 5)],
                    dtype="datetime64[us]")


def test_recovers_constant_scale_on_synthetic_clear_days():
    t = _times(range(1, 13))
    pv = 0.026 * clear_sky_poa(t)
    cs = ClearSkyPV().fit(t, pv)
    assert cs.seasonal
    assert cs.coef[0] == pytest.approx(0.026, rel=1e-3) and np.allclose(cs.coef[1:], 0, atol=1e-5)
    assert np.allclose(cs(t), pv, rtol=1e-3, atol=1e-6)


def test_narrow_season_falls_back_to_constant_and_stays_positive():
    t = _times([3])
    pv = 0.026 * clear_sky_poa(t)
    cs = ClearSkyPV().fit(t, pv)
    assert not cs.seasonal and np.allclose(cs.coef[1:], 0)
    assert (cs(_times(range(1, 13))) >= 0).all()
