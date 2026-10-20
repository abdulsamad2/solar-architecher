"""Unit tests for E8 solar-relative and motion features (synthetic, CPU)."""
import numpy as np
import pytest

from src.sky_geometry import detect_sun, fit_geometry, project
from src.solar_features import (CROP, motion_features, phase_shift, radial_inflow, ring_masks, solar_scalars,
                                sun_crop, sun_xy)

GEO = {"change_date": "2019-05-22",
       "pre": {"cx": 30.5, "cy": 28.9, "f": 17.9, "phi": -1.28, "s": -1, "lens": "equidistant"},
       "post": {"cx": 31.7, "cy": 31.0, "f": 18.0, "phi": -1.25, "s": -1, "lens": "equidistant"}}


def texture(n=64, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((n, n))
    for _ in range(3):  # smooth a little so phase correlation has structure at several scales
        x = (x + np.roll(x, 1, 0) + np.roll(x, 1, 1)) / 3
    return x


def test_geometry_fit_recovers_known_camera():
    rng = np.random.default_rng(0)
    zen, az = rng.uniform(10, 70, 400), rng.uniform(60, 300, 400)
    x, y = project(GEO["pre"], zen, az)
    g = fit_geometry(x + rng.normal(0, 0.3, 400), y + rng.normal(0, 0.3, 400), zen, az)
    px, py = project(g, zen, az)
    assert np.median(np.hypot(px - x, py - y)) < 0.3 and g["lens"] == "equidistant" and g["s"] == -1


def test_sun_detection_finds_synthetic_sun():
    img = np.full((3, 64, 64, 3), 120, np.uint8)
    img[0, 20:23, 40:43] = 255
    img[1, 50:52, 10:12] = 255
    sx, sy = detect_sun(img)
    assert abs(sx[0] - 41) < 0.01 and abs(sy[0] - 21) < 0.01 and abs(sx[1] - 10.5) < 0.01 and np.isnan(sx[2])


def test_regime_switch_and_solar_scalars():
    t = np.array(["2019-05-21T12:00", "2019-05-22T12:00"], dtype="datetime64[us]")
    x, y, zen, az = sun_xy(t, GEO)
    assert abs(x[1] - x[0]) > 0.5  # different geometry across the change date (same solar position +- 1 day)
    s = solar_scalars(zen, az, x, y)
    assert s.shape == (2, 5) and (np.abs(s[:, :3]) <= 1).all() and (s[:, 0] > 0).all()


def test_sun_crop_centres_and_pads():
    g = np.zeros((64, 64))
    g[10, 50] = 1.0
    c = sun_crop(g, 50, 10)
    assert c.shape == (CROP, CROP) and c[CROP // 2, CROP // 2] == 1.0
    edge = sun_crop(g, 1, 1)
    assert edge.shape == (CROP, CROP) and edge[:8, :8].sum() == 0  # zero padding outside the image


@pytest.mark.parametrize("dx,dy", [(2, 0), (0, -3), (1, 2)])
def test_phase_correlation_recovers_shift(dx, dy):
    a = texture(32, 1)
    b = np.roll(np.roll(a, dy, 0), dx, 1)
    ex, ey = phase_shift(a, b)
    assert abs(ex - dx) < 0.3 and abs(ey - dy) < 0.3


def test_radial_inflow_sign():
    a = np.zeros((24, 24))
    a[2:6, 2:6] = 1.0; a[2:6, 18:22] = 1.0; a[18:22, 2:6] = 1.0; a[18:22, 18:22] = 1.0  # one blob per quadrant
    toward = np.zeros_like(a)
    toward[3:7, 3:7] = 1.0; toward[3:7, 17:21] = 1.0; toward[17:21, 3:7] = 1.0; toward[17:21, 17:21] = 1.0
    assert radial_inflow(a, toward) > 0.5
    assert radial_inflow(toward, a) < -0.5


def test_motion_features_static_is_zero_and_change_localised():
    frames = np.repeat(texture(64, 2)[None], 4, 0)
    xs, ys = np.full(4, 32.0), np.full(4, 32.0)
    m = motion_features(frames, xs, ys, 1.0)
    assert m.shape == (4, 6) and np.allclose(m[:, :3], 0) and (m[0] == 0).all()
    f2 = frames.copy()
    f2[3, 30:34, 30:34] += 0.5  # change only in the inner ring at the last step
    m2 = motion_features(f2, xs, ys, 1.0)
    assert m2[3, 0] > 0 and m2[3, 2] == 0 and np.allclose(m2[1:3, :3], 0)


def test_motion_scales_with_stride_minutes():
    a = texture(64, 3)
    frames = np.stack([a, np.roll(a, 2, 1)])
    one = motion_features(frames, [32, 32], [32, 32], 1.0)[1, 3]
    four = motion_features(frames, [32, 32], [32, 32], 4.0)[1, 3]
    assert abs(one - 2.0) < 0.4 and abs(four - 0.5) < 0.1


def test_ring_masks_partition_inner_disc():
    ms = ring_masks()
    assert not (ms[0] & ms[1]).any() and not (ms[1] & ms[2]).any() and all(m.any() for m in ms)
