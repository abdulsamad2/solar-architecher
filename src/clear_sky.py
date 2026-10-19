"""Clear-sky PV anchor, clear-sky index, and smart persistence.

Clear-sky PV model (v1):
  Ineichen clear-sky irradiance at the site -> Hay-Davies plane-of-array transposition for the
  array geometry -> physical IAM on the beam component -> P_clear = scale * POA_eff.
  ``scale`` (kW per W/m^2) absorbs area, efficiency and DC/AC losses and is fitted on TRAINING
  days only (see ``fit_scale``). No cell-temperature model yet (no temperature data), so clear-sky K
  carries a seasonal residual; quantified in reports/protocol_notes.md.

Site/array constants come from the official Relative_op_func.py; altitude is an assumption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SITE = {"latitude": 37.424107, "longitude": -122.174199, "altitude": 30.0, "tz": "America/Los_Angeles"}
ARRAY = {"surface_tilt": 22.5, "surface_azimuth": 195.0}


def localize(times_naive_local) -> pd.DatetimeIndex:
    """SKIPP'D timestamps are naive local clock time (PST/PDT)."""
    idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(times_naive_local, dtype="datetime64[us]")))
    return idx.tz_localize(SITE["tz"], ambiguous="NaT", nonexistent="NaT")


def clear_sky_poa(times_naive_local, shift_min: float = 0.0) -> np.ndarray:
    """Effective clear-sky plane-of-array irradiance (W/m^2) at each time (+ optional shift)."""
    import pvlib

    idx = localize(times_naive_local)
    if shift_min:
        idx = idx + pd.Timedelta(minutes=shift_min)
    loc = pvlib.location.Location(SITE["latitude"], SITE["longitude"], tz=SITE["tz"], altitude=SITE["altitude"])
    valid = ~idx.isna()
    out = np.zeros(len(idx))
    vidx = idx[valid]
    sp = loc.get_solarposition(vidx)
    cs = loc.get_clearsky(vidx, model="ineichen", solar_position=sp)
    poa = pvlib.irradiance.get_total_irradiance(
        ARRAY["surface_tilt"], ARRAY["surface_azimuth"], sp["apparent_zenith"], sp["azimuth"],
        cs["dni"], cs["ghi"], cs["dhi"], dni_extra=pvlib.irradiance.get_extra_radiation(vidx), model="haydavies",
    )
    aoi = pvlib.irradiance.aoi(ARRAY["surface_tilt"], ARRAY["surface_azimuth"], sp["apparent_zenith"], sp["azimuth"])
    eff = poa["poa_direct"] * pvlib.iam.physical(aoi) + poa["poa_diffuse"]
    out[valid] = np.clip(np.nan_to_num(eff.to_numpy()), 0, None)
    return out


def fit_scale(pv_kw, poa, day, min_poa: float = 300.0, day_quantile: float = 0.95) -> float:
    """kW per W/m^2 from TRAINING records only.

    Per day: the ``day_quantile`` of PV/POA over records with POA >= ``min_poa`` (an upper envelope
    that is near the clear-sky ratio on clear and partly cloudy days). Scale = median over days.
    """
    pv_kw, poa, day = (np.asarray(a) for a in (pv_kw, poa, day))
    m = poa >= min_poa
    ratio = pd.Series(pv_kw[m] / poa[m])
    per_day = ratio.groupby(day[m]).quantile(day_quantile)
    return float(per_day.median())


class ClearSkyPV:
    """P_clear(t) = scale(doy) * POA_eff(t), scale(doy) = a + b cos(w doy) + c sin(w doy).

    The annual harmonic is an empirical stand-in for the missing cell-temperature model (clear-day
    K drifts ~0.85 in Aug to ~1.0 in Feb with a constant scale). Fit on training days only.
    """

    def __init__(self, coef=(0.0, 0.0, 0.0)):
        self.coef = np.asarray(coef, dtype=float)

    @staticmethod
    def _design(doy) -> np.ndarray:
        w = 2 * np.pi * np.asarray(doy, float) / 365.25
        return np.stack([np.ones_like(w), np.cos(w), np.sin(w)], axis=-1)

    @staticmethod
    def _doy(times_naive_local) -> np.ndarray:
        return pd.DatetimeIndex(pd.to_datetime(np.asarray(times_naive_local, dtype="datetime64[us]"))).dayofyear.to_numpy()

    def fit(self, times_naive_local, pv_kw, min_poa: float = 300.0, day_quantile: float = 0.5,
            cloudy_cut: float = 0.08, n_iter: int = 3) -> "ClearSkyPV":
        poa = clear_sky_poa(times_naive_local)
        pv_kw = np.asarray(pv_kw, float)
        day = np.asarray(times_naive_local, dtype="datetime64[D]")
        m = poa >= min_poa
        per_day = pd.Series(pv_kw[m] / poa[m]).groupby(day[m]).quantile(day_quantile)
        doy_index = pd.DatetimeIndex(per_day.index)
        X = self._design(doy_index.dayofyear)
        # the annual harmonic needs (nearly) all seasons; otherwise fit a constant scale
        self.seasonal = doy_index.month.nunique() >= 10
        if not self.seasonal:
            X = X[:, :1]
        y = per_day.to_numpy()
        keep = np.ones(len(y), bool)
        for _ in range(n_iter):  # drop days far below the envelope (overcast) and refit
            coef = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
            keep = (y - X @ coef) / (X @ coef) > -cloudy_cut
        self.coef = np.pad(coef, (0, 3 - len(coef)))
        self.n_days_fit = int(keep.sum())
        scale_all_year = self._design(np.arange(1, 367)) @ self.coef
        if not (scale_all_year > 0).all():
            raise ValueError(f"clear-sky scale not positive for all days of year: coef={self.coef}")
        return self

    def __call__(self, times_naive_local, shift_min: float = 0.0) -> np.ndarray:
        scale = self._design(self._doy(times_naive_local)) @ self.coef
        return scale * clear_sky_poa(times_naive_local, shift_min=shift_min)


def clear_sky_index(p_kw, p_clear_kw, min_clear_kw: float) -> np.ndarray:
    """K = P / max(P_clear, min_clear_kw). Invertible everywhere; not clipped to <= 1 (cloud enhancement)."""
    return np.asarray(p_kw, float) / np.maximum(np.asarray(p_clear_kw, float), min_clear_kw)


def from_clear_sky_index(k, p_clear_kw, min_clear_kw: float) -> np.ndarray:
    return np.asarray(k, float) * np.maximum(np.asarray(p_clear_kw, float), min_clear_kw)


def smart_persistence(p_now, p_clear_now, p_clear_future, min_clear_kw: float) -> np.ndarray:
    """P_now * Pclear(t+h)/Pclear(t); falls back to plain persistence when Pclear(t) is too small."""
    p, c0, c1 = (np.asarray(a, float) for a in (p_now, p_clear_now, p_clear_future))
    out = p.copy()
    ok = c0 >= min_clear_kw
    out[ok] = p[ok] * c1[ok] / c0[ok]
    return out
