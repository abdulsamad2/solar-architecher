"""Clear-sky helpers and smart persistence.

NOTE: ``clear_sky_ghi`` returns irradiance (W/m^2), NOT clear-sky PV power. The
calibrated PV envelope (POA transposition, temperature, DC/AC losses, fit on
training data only) is still to be implemented in ``clear_sky_pv_kw``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def clear_sky_ghi(times: pd.DatetimeIndex, latitude: float, longitude: float, altitude: float = 0.0) -> pd.Series:
    from pvlib.location import Location

    loc = Location(latitude=latitude, longitude=longitude, altitude=altitude, tz="UTC")
    return loc.get_clearsky(times, model="ineichen")["ghi"]


def clear_sky_pv_kw(times: pd.DatetimeIndex) -> pd.Series:
    raise NotImplementedError("Calibrated pvlib PV envelope not implemented yet (Phase 2, step 2.4).")


def clear_sky_index(p_kw, p_clear_kw, min_clear_kw: float) -> np.ndarray:
    """K = P / P_clear; NaN where P_clear is below ``min_clear_kw``. Not clipped to <= 1."""
    p, pc = np.asarray(p_kw, float), np.asarray(p_clear_kw, float)
    out = np.full_like(p, np.nan)
    ok = pc >= min_clear_kw
    out[ok] = p[ok] / pc[ok]
    return out


def smart_persistence(p_now, p_clear_now, p_clear_future, min_clear_kw: float) -> np.ndarray:
    """P_now * Pclear(t+h)/Pclear(t); falls back to plain persistence when Pclear(t) is too small."""
    p, c0, c1 = (np.asarray(a, float) for a in (p_now, p_clear_now, p_clear_future))
    out = p.copy()
    ok = c0 >= min_clear_kw
    out[ok] = p[ok] * c1[ok] / c0[ok]
    return out
