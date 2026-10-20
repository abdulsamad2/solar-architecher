"""E8 solar-relative and motion features (spec: reports/e8_prereg_DRAFT.md section 3).

Per frame of a 16-frame history window (oldest -> newest), 11 scalars:
  P (solar position, 5): cos(zenith), sin(azimuth), cos(azimuth), sun_x / 64, sun_y / 64
  M (motion, 6), between frame k and the previous frame in the window (zero for the oldest frame):
     annular mean |dI| in sun-centred rings r in [0,4), [4,8), [8,12) px  (grey level / 255)
     global shift (dx, dy) of the 24x24 sun-centred crop by phase correlation, in px per minute
     radial inflow: mean over four quadrant sub-crops of the shift component pointing TOWARD the sun, px/min
All inputs are frames inside the window (all at or before the origin): causal by construction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.sky_geometry import project, solar_position

CROP = 24
RINGS = ((0, 4), (4, 8), (8, 12))
N_P, N_M = 5, 6


def load_geometry(path="reports/e8_geometry_frozen.json") -> dict:
    return json.loads(Path(path).read_text())


def sun_xy(times_naive_local, geometry: dict):
    """Sun pixel per time using the regime geometry (pre/post camera change)."""
    t = np.asarray(times_naive_local, dtype="datetime64[us]")
    sp = solar_position(t)
    zen, az = sp["zenith_deg"].to_numpy(), sp["azimuth_deg"].to_numpy()
    post = t.astype("datetime64[D]") >= np.datetime64(geometry["change_date"])
    x, y = np.empty(len(t)), np.empty(len(t))
    for flag, key in ((False, "pre"), (True, "post")):
        m = post == flag
        if m.any():
            x[m], y[m] = project(geometry[key], zen[m], az[m])
    return x, y, zen, az


def solar_scalars(zen, az, x, y) -> np.ndarray:
    z, a = np.radians(zen), np.radians(az)
    return np.stack([np.cos(z), np.sin(a), np.cos(a), x / 64.0, y / 64.0], -1).astype(np.float32)


def sun_crop(gray: np.ndarray, x: float, y: float, size: int = CROP) -> np.ndarray:
    """size x size crop of a [H,W] grey image centred on (x, y), zero-padded outside the image."""
    h = size // 2
    cx, cy = int(round(x)), int(round(y))
    pad = np.pad(gray, size, mode="constant")
    return pad[cy + size - h:cy + size + h, cx + size - h:cx + size + h]


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Shift (dx, dy) that maps a onto b (b(x) ~ a(x - d)), integer peak + parabolic sub-pixel refinement."""
    a = (a - a.mean()) * np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    b = (b - b.mean()) * np.hanning(b.shape[0])[:, None] * np.hanning(b.shape[1])[None, :]
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    r = fb * np.conj(fa)
    r /= np.maximum(np.abs(r), 1e-9)
    c = np.real(np.fft.ifft2(r))
    py, px = np.unravel_index(np.argmax(c), c.shape)

    def sub(v_m, v_0, v_p):
        d = v_m - 2 * v_0 + v_p
        return 0.0 if abs(d) < 1e-12 else 0.5 * (v_m - v_p) / d

    n0, n1 = c.shape
    dy = py + sub(c[(py - 1) % n0, px], c[py, px], c[(py + 1) % n0, px])
    dx = px + sub(c[py, (px - 1) % n1], c[py, px], c[py, (px + 1) % n1])
    dy = dy - n0 if dy > n0 / 2 else dy
    dx = dx - n1 if dx > n1 / 2 else dx
    return float(dx), float(dy)


def ring_masks(size: int = CROP):
    yy, xx = np.mgrid[0:size, 0:size] - (size / 2 - 0.5)
    r = np.hypot(xx, yy)
    return [(r >= lo) & (r < hi) for lo, hi in RINGS]


def radial_inflow(prev: np.ndarray, cur: np.ndarray) -> float:
    """Mean over the four 12x12 quadrants of the shift component pointing toward the crop centre (the sun)."""
    h = prev.shape[0] // 2
    vals = []
    for (r0, c0), (uy, ux) in (((0, 0), (1, 1)), ((0, h), (1, -1)), ((h, 0), (-1, 1)), ((h, h), (-1, -1))):
        dx, dy = phase_shift(prev[r0:r0 + h, c0:c0 + h], cur[r0:r0 + h, c0:c0 + h])
        vals.append((dx * ux + dy * uy) / np.sqrt(2))  # unit vector from quadrant centre toward crop centre
    return float(np.mean(vals))


_MASKS = None


def pair_motion(prev_gray: np.ndarray, cur_gray: np.ndarray, x: float, y: float, minutes: float) -> np.ndarray:
    """6 motion features between two grey frames, both cropped at the CURRENT sun position (x, y), so solar
    motion is not counted as cloud motion. Shifts in px per minute."""
    global _MASKS
    if _MASKS is None:
        _MASKS = ring_masks()
    prev, cur = sun_crop(prev_gray, x, y), sun_crop(cur_gray, x, y)
    diff = np.abs(cur - prev)
    dx, dy = phase_shift(prev, cur)
    return np.array([diff[m].mean() for m in _MASKS] + [dx / minutes, dy / minutes, radial_inflow(prev, cur) / minutes],
                    np.float32)


def motion_features(gray_frames: np.ndarray, xs, ys, minutes_per_step: float) -> np.ndarray:
    """[16, 6] motion between consecutive frames of a chronological window (draft definition; zero for the oldest).
    The frozen E8 features use pair_motion on 1-min record pairs instead (Amendment A)."""
    out = np.zeros((len(gray_frames), N_M), np.float32)
    for k in range(1, len(gray_frames)):
        out[k] = pair_motion(gray_frames[k - 1], gray_frames[k], xs[k], ys[k], minutes_per_step)
    return out
