"""Sky-camera geometry: map solar (zenith, azimuth) to 64x64 image pixels, fitted on clear-day sun detections.

Model: image angle = s * azimuth + phi (s = +1 or -1 handles mirroring), radius = f * R(zenith) with lens law
R in {equidistant: theta, equisolid: 2 sin(theta/2), orthographic: sin(theta)}; centre (cx, cy).
Pixel convention: x = column, y = row (downwards); centre of pixel (i, j) at (j, i).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LENS = {"equidistant": lambda z: z, "equisolid": lambda z: 2 * np.sin(z / 2), "orthographic": np.sin}


def solar_position(times_naive_local) -> pd.DataFrame:
    import pvlib

    from src.clear_sky import SITE, localize

    idx = localize(times_naive_local)
    loc = pvlib.location.Location(SITE["latitude"], SITE["longitude"], tz=SITE["tz"], altitude=SITE["altitude"])
    sp = loc.get_solarposition(idx)
    return pd.DataFrame({"zenith_deg": sp["apparent_zenith"].to_numpy(), "azimuth_deg": sp["azimuth"].to_numpy()})


def detect_sun(images: np.ndarray, thresh: int = 250, min_px: int = 2, max_px: int = 40):
    """Centroid (x, y) of saturated pixels per frame [N,64,64,3]; NaN when no compact sun blob is found."""
    sat = (images.min(axis=-1) >= thresh)
    n = sat.sum(axis=(1, 2))
    ys, xs = np.mgrid[0:images.shape[1], 0:images.shape[2]]
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = (sat * xs).sum(axis=(1, 2)) / n
        cy = (sat * ys).sum(axis=(1, 2)) / n
        spread = np.sqrt(((sat * ((xs - cx[:, None, None]) ** 2 + (ys - cy[:, None, None]) ** 2)).sum(axis=(1, 2))) / n)
    ok = (n >= min_px) & (n <= max_px) & (spread < 3.0)
    return np.where(ok, cx, np.nan), np.where(ok, cy, np.nan)


def project(params: dict, zenith_deg, azimuth_deg):
    z = np.radians(np.asarray(zenith_deg, float))
    a = np.radians(np.asarray(azimuth_deg, float))
    r = params["f"] * LENS[params["lens"]](z)
    ang = params["s"] * a + params["phi"]
    return params["cx"] + r * np.cos(ang), params["cy"] + r * np.sin(ang)


def fit_geometry(x, y, zenith_deg, azimuth_deg) -> dict:
    """Least-squares fit of (cx, cy, f, phi) for each lens law and handedness; returns the best by RMS error."""
    from scipy.optimize import least_squares

    best = None
    for lens in LENS:
        for s in (1, -1):
            def resid(p):
                px, py = project({"cx": p[0], "cy": p[1], "f": p[2], "phi": p[3], "s": s, "lens": lens}, zenith_deg, azimuth_deg)
                return np.concatenate([px - x, py - y])
            starts = [np.array([31.5, 31.5, 20.0, ph]) for ph in np.linspace(-np.pi, np.pi, 8, endpoint=False)]
            for p0 in starts:
                r = least_squares(resid, p0, loss="soft_l1", f_scale=1.0)
                rms = float(np.sqrt(np.mean(r.fun ** 2)))
                if best is None or rms < best["rms_px"]:
                    best = {"cx": float(r.x[0]), "cy": float(r.x[1]), "f": float(r.x[2]),
                            "phi": float(np.arctan2(np.sin(r.x[3]), np.cos(r.x[3]))), "s": s, "lens": lens, "rms_px": rms}
    return best


def sun_pixel(params: dict, times_naive_local) -> tuple[np.ndarray, np.ndarray]:
    sp = solar_position(times_naive_local)
    return project(params, sp["zenith_deg"].to_numpy(), sp["azimuth_deg"].to_numpy())
