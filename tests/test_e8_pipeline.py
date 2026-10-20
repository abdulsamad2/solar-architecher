"""E8 real-data pipeline tests (opt-in: pytest -m realdata). Window builder, timestamps, causality, feature store."""
import json
from pathlib import Path

import numpy as np
import pytest

FEAT = Path("data/features")
pytestmark = [pytest.mark.realdata,
              pytest.mark.skipif(not (FEAT / "e8_record_features_v1.npz").exists(), reason="E8 feature store not built")]

STRIDE = {"H15": 1, "H30": 2, "H60": 4}
MIN = 60_000_000


@pytest.fixture(scope="module")
def ctx():
    m = np.load("data/manifests/forecast_tol30_v2.npz")
    w = np.load(FEAT / "e8_windows_v1.npz")
    r = np.load(FEAT / "e8_record_features_v1.npz")
    t = np.asarray(np.load("data/raw/times_trainval.npy", allow_pickle=True), dtype="datetime64[us]").astype(np.int64)
    dev = ~m["is_test"]
    origin_us = m["times_us"][m["origin"][dev]]
    return {"m": m, "w": {k: w[k] for k in w.files}, "r": {k: r[k] for k in r.files}, "t": t, "origin_us": origin_us}


def test_frozen_hashes_match():
    from src.e8_features import sha256, verify_frozen

    verify_frozen()
    meta = json.loads((FEAT / "e8_features_v1.json").read_text())
    assert sha256(FEAT / "e8_record_features_v1.npz") == meta["record_features_sha256"]
    assert sha256(FEAT / "e8_windows_v1.npz") == meta["windows_sha256"]


def test_window_counts_and_common_population(ctx):
    w = ctx["w"]
    counts = {k: int((w[k] >= 0).all(1).sum()) for k in STRIDE}
    assert counts == {"H15": 164362, "H30": 159414, "H60": 150758}
    common = (w["H15"] >= 0).all(1) & (w["H30"] >= 0).all(1) & (w["H60"] >= 0).all(1)
    assert int(common.sum()) == 150618 and np.array_equal(common, w["eligible_common"])


@pytest.mark.parametrize("name", list(STRIDE))
def test_window_timestamps_causal_same_day_and_16_frames(ctx, name):
    rows = ctx["w"][name]
    ok = (rows >= 0).all(1)
    rows, o = rows[ok], ctx["origin_us"][ok]
    assert rows.shape[1] == 16
    tt = ctx["t"][rows]
    nominal = o[:, None] - STRIDE[name] * np.arange(16)[None, :] * MIN
    assert (np.abs(tt - nominal) <= 30_000_000).all()
    assert (tt <= o[:, None]).all() and (np.diff(tt, axis=1) < 0).all()
    day = tt.astype("datetime64[us]").astype("datetime64[D]")
    assert (day == day[:, :1]).all()


def test_h15_window_equals_existing_e4_history(ctx):
    from src.skippd import SkippdForecastDataset

    ds = SkippdForecastDataset("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz",
                               "dev", preload=False, newest_first=True)
    assert np.array_equal(ctx["w"]["H15"], ds.hist_rows)


def test_one_minute_pairs_separation_causality_and_missing_policy(ctx):
    r, t = ctx["r"], ctx["t"]
    p = r["prev_1min_row"]
    has = p >= 0
    assert np.array_equal(has, r["has_motion"])
    dt = (t[has] - t[p[has]]) / MIN
    assert (dt >= 0.5).all() and (dt <= 1.5).all() and (p[has] < np.flatnonzero(has)).all()
    assert np.allclose(r["M"][~has], 0.0)  # missing-pair policy: zero motion
    assert 0.99 < has.mean()


def test_feature_store_matches_on_the_fly(ctx):
    import h5py

    from src.solar_features import load_geometry, pair_motion, solar_scalars, sun_xy

    r, t = ctx["r"], ctx["t"]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(np.flatnonzero(r["has_motion"]), 200, replace=False))
    g = np.load(FEAT / "gray_u8_trainval.npy", mmap_mode="r")
    x, y, zen, az = sun_xy(t[idx].astype("datetime64[us]"), load_geometry())
    assert np.allclose(solar_scalars(zen, az, x, y), r["P"][idx], atol=1e-5)
    for k, i in enumerate(idx):
        p = r["prev_1min_row"][i]
        mf = pair_motion(g[p].astype(np.float32) / 255, g[i].astype(np.float32) / 255, x[k], y[k], (t[i] - t[p]) / MIN)
        assert np.allclose(mf, r["M"][i], atol=1e-5)
    with h5py.File("data/raw/2017_2019_images_pv_processed.hdf5", "r") as f:
        gray = f["trainval/images_log"][idx[:20]].astype(np.float32).mean(-1)
    assert np.array_equal(np.round(gray).astype(np.uint8), g[idx[:20]])


def test_coordinate_convention_against_detected_sun(ctx):
    """Predicted sun pixel (x = column, y = row) agrees with the brightest blurred spot on bright frames."""
    import torch
    import torch.nn.functional as F

    r = ctx["r"]
    g = np.load(FEAT / "gray_u8_trainval.npy", mmap_mode="r")
    rng = np.random.default_rng(1)
    idx = np.sort(rng.choice(len(g), 4000, replace=False))
    blur = F.avg_pool2d(torch.from_numpy(g[idx].astype(np.float32))[:, None], 5, 1, 2)[:, 0].numpy().reshape(len(idx), -1)
    am, peak = blur.argmax(1), blur.max(1)
    zen_ok = r["P"][idx, 0] > np.cos(np.radians(70))
    sel = (peak >= 240) & zen_ok
    err = np.hypot(am[sel] % 64 - r["sun_x"][idx][sel], am[sel] // 64 - r["sun_y"][idx][sel])
    assert sel.sum() > 300 and np.median(err) < 1.5
