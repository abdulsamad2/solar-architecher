"""Gatton clear-sky PV model with the ClearSkyPV interface (prereg E-2/E-4; reports/phaseE_gatton_clearsky.json).

P_clear(t) = min( s * [ r * POA_fixed(t) + (1 - r) * POA_tracker(t) ], C )
Geometry (fixed component, tracker, r) and the AC cap C are frozen from the development setup; the scale s is refitted
with the ClearSkyPV rule (per-day median PV/shape for shape >= 300 W/m^2, iterative 8% cloudy-day rejection,
constant scale) on whatever training records ``fit`` receives (the fold's fit days in training).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = Path("reports/phaseE_gatton_clearsky.json")


def load_params() -> dict:
    return json.loads(PARAMS.read_text())


class GattonClearSky:
    def __init__(self, scale: float | None = None):
        self.p = load_params()
        self.scale = self.p["scale"] if scale is None else scale
        self.coef = np.array([self.scale, 0.0, 0.0])  # recorded in metrics like ClearSkyPV.coef
        self.seasonal = False

    @classmethod
    def from_dev_fit(cls) -> "GattonClearSky":
        return cls()

    def shape(self, times) -> np.ndarray:
        from src.gatton_setup import poa_clear
        idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(times, dtype="datetime64[us]"))).tz_localize(self.p["site"]["tz"])
        r = self.p["r_fixed"]
        return r * poa_clear(idx, self.p["fixed_component"]) + (1 - r) * poa_clear(idx, {"type": "tracker"})

    def fit(self, times, pv_kw) -> "GattonClearSky":
        shape = self.shape(times)
        y = np.asarray(pv_kw, float)
        day = np.asarray(times, dtype="datetime64[D]")
        m = (shape >= self.p["min_poa"]) & np.isfinite(y)
        v = pd.Series(y[m] / shape[m]).groupby(day[m]).quantile(0.5).to_numpy()
        keep = np.ones(len(v), bool)
        for _ in range(3):
            sc = float(v[keep].mean())
            keep = (v - sc) / sc > -0.08
        self.scale, self.coef, self.n_days_fit = sc, np.array([sc, 0.0, 0.0]), int(keep.sum())
        return self

    def __call__(self, times, shift_min: float = 0.0) -> np.ndarray:
        t = np.asarray(times, dtype="datetime64[us]")
        if shift_min:
            t = t + np.timedelta64(int(shift_min * 60), "s")
        return np.minimum(self.scale * self.shape(t), self.p["cap_kw"])
