"""E8 feature store (reports/e8_prereg.md section 3 + Amendment A).

Per development image record r (HDF5 'trainval' row order):
  P[r] (5): solar scalars at record time (frozen regime geometry)
  M[r] (6): pair_motion(record r', record r) with r' = nearest record to t_r - 1 min (+-30 s, same day, r' < r);
            zero and has_motion = False if missing.
Per window (H15/H30/H60) and development sample: 16 history rows (newest first), from src.e8_audit.history_rows,
mapped to HDF5 rows; eligibility must equal the frozen eligibility manifest (STOP otherwise).
"""
from __future__ import annotations

import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

from src.e8_audit import WINDOWS, history_rows
from src.skippd import _nearest
from src.solar_features import N_M, load_geometry, pair_motion, solar_scalars, sun_xy

GEOMETRY_SHA = "308807C6817393E897241A8E2FE4BEA5966E6EB728398FDAFB2063D8F0DC41C0"
ELIGIBILITY_SHA = "2F4CDB9AD672C816A46AD49D1B3237A151DCFF41DFEEC615DAB2A16E5658BB0B"
FEAT = Path("data/features")
GRAY = FEAT / "gray_u8_trainval.npy"
MIN_US, TOL_US = 60_000_000, 30_000_000


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def verify_frozen():
    assert sha256("reports/e8_geometry_frozen.json") == GEOMETRY_SHA, "STOP: geometry file changed"
    assert sha256("data/manifests/e8_eligibility_v1.npz") == ELIGIBILITY_SHA, "STOP: eligibility manifest changed"


def record_times() -> np.ndarray:
    return np.asarray(np.load("data/raw/times_trainval.npy", allow_pickle=True), dtype="datetime64[us]")


def predecessor_1min(t_us: np.ndarray) -> np.ndarray:
    """Row of the record nearest t - 1 min (+-30 s), same day and earlier; -1 if none. t_us sorted ascending."""
    p = _nearest(t_us, t_us - MIN_US, TOL_US)
    day = t_us.astype("datetime64[us]").astype("datetime64[D]")
    ok = (p >= 0) & (p < np.arange(len(t_us)))
    ok &= np.where(ok, day[np.maximum(p, 0)] == day, False)
    return np.where(ok, p, -1)


_G = None


def _work(args):
    global _G
    if _G is None:
        _G = np.load(GRAY, mmap_mode="r")
    rows, prev, xs, ys, dts = args
    out = np.zeros((len(rows), N_M), np.float32)
    for i, (r, p, x, y, dt) in enumerate(zip(rows, prev, xs, ys, dts)):
        if p >= 0:
            out[i] = pair_motion(_G[p].astype(np.float32) / 255.0, _G[r].astype(np.float32) / 255.0, x, y, dt)
    return out


def build_record_table(workers: int = 32) -> dict:
    verify_frozen()
    FEAT.mkdir(parents=True, exist_ok=True)
    if not GRAY.exists():
        with h5py.File("data/raw/2017_2019_images_pv_processed.hdf5", "r") as f:
            ds = f["trainval/images_log"]
            g = np.empty(ds.shape[:3], np.uint8)
            for s in range(0, ds.shape[0], 20000):
                g[s:s + 20000] = np.round(ds[s:s + 20000].astype(np.float32).mean(-1)).astype(np.uint8)
        np.save(GRAY, g)
    t = record_times()
    t_us = t.astype(np.int64)
    assert (np.diff(t_us) > 0).all(), "trainval records must be sorted"
    geo = load_geometry()
    x, y, zen, az = sun_xy(t, geo)
    P = solar_scalars(zen, az, x, y)
    prev = predecessor_1min(t_us)
    dts = np.where(prev >= 0, (t_us - t_us[np.maximum(prev, 0)]) / MIN_US, 1.0)
    idx = np.arange(len(t))
    chunks = [(idx[s:s + 2000], prev[s:s + 2000], x[s:s + 2000], y[s:s + 2000], dts[s:s + 2000])
              for s in range(0, len(t), 2000)]
    with Pool(workers) as pool:
        M = np.concatenate(pool.map(_work, chunks))
    assert np.isfinite(P).all() and np.isfinite(M).all(), "STOP: non-finite features"
    out = {"P": P, "M": M, "prev_1min_row": prev, "has_motion": prev >= 0, "sun_x": x.astype(np.float32),
           "sun_y": y.astype(np.float32)}
    np.savez_compressed(FEAT / "e8_record_features_v1.npz", **out)
    return out


def build_windows() -> dict:
    """Per window: [N_dev, 16] HDF5 trainval rows (newest first; -1 if ineligible) + eligibility check."""
    verify_frozen()
    FEAT.mkdir(parents=True, exist_ok=True)
    m = np.load("data/manifests/forecast_tol30_v2.npz")
    dev = ~m["is_test"]
    origin = m["origin"][dev]
    elig = np.load("data/manifests/e8_eligibility_v1.npz")
    out = {"sample_ids": elig["sample_ids"]}
    for name, stride in WINDOWS.items():
        tl = history_rows(m["times_us"], origin, stride)
        ok = (tl >= 0).all(1)
        assert np.array_equal(ok, elig[f"eligible_{name}"]), f"STOP: {name} eligibility differs from frozen manifest"
        assert (m["hdf5_group"][np.maximum(tl, 0)][ok] == 0).all(), "STOP: dev window uses a test record"
        out[name] = np.where(tl >= 0, m["hdf5_row"][np.maximum(tl, 0)], -1).astype(np.int64)
    out["eligible_common"] = elig["eligible_common"]
    np.savez_compressed(FEAT / "e8_windows_v1.npz", **out)
    return out


if __name__ == "__main__":
    import time

    t0 = time.time()
    w = build_windows()
    print("windows built:", {k: int((v >= 0).all(1).sum()) for k, v in w.items() if k.startswith("H")}, f"{time.time() - t0:.0f}s")
    t0 = time.time()
    r = build_record_table()
    print(f"record table: {len(r['P']):,} records, motion available {r['has_motion'].mean():.4f}, {time.time() - t0:.0f}s")
    meta = {"geometry_sha256": GEOMETRY_SHA, "eligibility_sha256": ELIGIBILITY_SHA,
            "record_features_sha256": sha256(FEAT / "e8_record_features_v1.npz"),
            "windows_sha256": sha256(FEAT / "e8_windows_v1.npz")}
    (FEAT / "e8_features_v1.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
