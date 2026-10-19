"""Port of the official SKIPP'D forecast sample construction, plus a frozen manifest builder.

Reference: reference-skippd/data_processing/data_preprocess_forecast.ipynb (cell 6).
Rules reproduced:
  * stack of ``stack_height + 1`` frames at t - j minutes, j = 0..stack_height (newest first)
  * target PV at t + forecast_horizon minutes
  * greedy origin thinning: candidate i is used only if
        times[i] - times[last_valid] > sampling_interval - 1 s
    where last_valid starts at index 0 and updates only on valid samples
  * the test split is the 20 official days, taken from samples built on the full timeline

Documented deviation (protocol decision, see reports/protocol_notes.md):
the official build used the unreleased ~10 s image stream, where exact t - j minute matches
always exist. The benchmark HDF5 is thinned to ~1 min with drifting second-phase, so exact
matching drops ~19 % of samples. ``tolerance_s`` matches each required time to the nearest
record within that tolerance; ``tolerance_s=0`` is the strict official rule.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

SUNNY_DAYS = [(2017, 9, 15), (2017, 10, 6), (2017, 10, 22), (2018, 2, 16), (2018, 6, 12),
              (2018, 6, 23), (2019, 1, 25), (2019, 6, 23), (2019, 7, 14), (2019, 10, 14)]
CLOUDY_DAYS = [(2017, 6, 24), (2017, 9, 20), (2017, 10, 11), (2018, 1, 25), (2018, 3, 9),
               (2018, 10, 4), (2019, 5, 27), (2019, 6, 28), (2019, 8, 10), (2019, 10, 19)]
SUNNY_DATES = {dt.date(*d) for d in SUNNY_DAYS}
TEST_DATES = SUNNY_DATES | {dt.date(*d) for d in CLOUDY_DAYS}


@dataclass(frozen=True)
class ForecastSamples:
    origin: np.ndarray   # [N] row index of t in the sorted timeline
    history: np.ndarray  # [N, stack_height+1] row indices, newest first (column 0 is t)
    target: np.ndarray   # [N] row index matched to t + horizon
    is_test: np.ndarray  # [N] bool, official test day


def _to_us(times) -> np.ndarray:
    return np.asarray(times, dtype="datetime64[us]").astype(np.int64)


def _nearest(sorted_us: np.ndarray, query_us: np.ndarray, tol_us: int) -> np.ndarray:
    k = np.clip(np.searchsorted(sorted_us, query_us), 1, len(sorted_us) - 1)
    k = np.where(np.abs(sorted_us[k - 1] - query_us) <= np.abs(sorted_us[k] - query_us), k - 1, k)
    return np.where(np.abs(sorted_us[k] - query_us) <= tol_us, k, -1)


def official_forecast_samples(
    times,
    stack_height: int = 15,
    forecast_horizon_min: int = 15,
    sampling_interval_min: int = 2,
    tolerance_s: float = 0.0,
) -> ForecastSamples:
    """``times``: strictly increasing naive local datetimes of the paired image/PV records."""
    us = _to_us(times)
    if len(us) < 2 or (np.diff(us) <= 0).any():
        raise ValueError("times must be strictly increasing; sort the combined timeline first")
    minute, tol = 60_000_000, int(round(tolerance_s * 1_000_000))
    if tol > minute // 2:
        raise ValueError("tolerance must be at most 30 s (half the frame spacing)")

    hist = np.stack([_nearest(us, us - j * minute, tol) for j in range(stack_height + 1)], axis=1)
    tgt = _nearest(us, us + forecast_horizon_min * minute, tol)
    valid = (hist >= 0).all(1) & (tgt >= 0) & (np.diff(hist, axis=1) < 0).all(1)
    valid &= tgt > np.arange(len(us))  # target strictly after origin

    gate = sampling_interval_min * minute - 1_000_000
    keep, last = [], 0
    for i in np.flatnonzero(valid):
        if us[i] - us[last] > gate:
            keep.append(i)
            last = i
    origin = np.asarray(keep, dtype=np.int64)
    dates = np.asarray(times, dtype="datetime64[D]")[origin].astype(object)
    return ForecastSamples(
        origin=origin,
        history=hist[origin],
        target=tgt[origin],
        is_test=np.array([d in TEST_DATES for d in dates], dtype=bool),
    )


def load_benchmark_timeline(raw_dir: str | Path):
    """Combined, sorted timeline and the HDF5 (group, row) location of each record."""
    raw_dir = Path(raw_dir)
    tv = np.load(raw_dir / "times_trainval.npy", allow_pickle=True)
    te = np.load(raw_dir / "times_test.npy", allow_pickle=True)
    times = np.concatenate([tv, te])
    group = np.concatenate([np.zeros(len(tv), np.int8), np.ones(len(te), np.int8)])  # 0 trainval, 1 test
    row = np.concatenate([np.arange(len(tv)), np.arange(len(te))])
    order = np.argsort(_to_us(times), kind="stable")
    return times[order], group[order], row[order]


def build_forecast_manifest(
    raw_dir: str | Path,
    out_path: str | Path,
    tolerance_s: float = 30.0,
    dev_sampling_min: int = 2,
    test_sampling_min: int = 1,
) -> dict:
    """Freeze the forecast sample index to ``out_path`` (.npz) with a JSON sidecar of counts and hash.

    The official SUNSET run used 2-min origins for dev (149,680) but every valid minute for the
    test days (12,248), hence separate cadences.
    """
    times, group, row = load_benchmark_timeline(raw_dir)
    dev_s = official_forecast_samples(times, sampling_interval_min=dev_sampling_min, tolerance_s=tolerance_s)
    test_s = official_forecast_samples(times, sampling_interval_min=test_sampling_min, tolerance_s=tolerance_s)
    keep_dev, keep_test = ~dev_s.is_test, test_s.is_test
    order = np.argsort(np.concatenate([dev_s.origin[keep_dev], test_s.origin[keep_test]]), kind="stable")
    s = ForecastSamples(*(np.concatenate([getattr(dev_s, f)[keep_dev], getattr(test_s, f)[keep_test]])[order]
                          for f in ("origin", "history", "target", "is_test")))
    dates = np.asarray(times, dtype="datetime64[D]")[s.origin].astype(object)
    is_sunny = np.array([d in SUNNY_DATES for d in dates], dtype=bool)

    # a test-day origin must only draw on test-group records, and vice versa
    groups_used = group[np.concatenate([s.history, s.target[:, None]], axis=1)]
    assert (groups_used == s.is_test[:, None].astype(np.int8)).all(), "split leakage across HDF5 groups"

    arrays = {
        "times_us": _to_us(times),
        "hdf5_group": group,
        "hdf5_row": row,
        "origin": s.origin,
        "history": s.history,
        "target": s.target,
        "is_test": s.is_test,
        "is_sunny": is_sunny,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    digest = hashlib.sha256()
    for k in sorted(arrays):
        digest.update(k.encode())
        digest.update(np.ascontiguousarray(arrays[k]).tobytes())
    info = {
        "tolerance_s": tolerance_s,
        "stack_frames": int(s.history.shape[1]),
        "horizon_min": 15,
        "dev_sampling_min": dev_sampling_min,
        "test_sampling_min": test_sampling_min,
        "n_records": int(len(times)),
        "n_dev": int((~s.is_test).sum()),
        "n_test": int(s.is_test.sum()),
        "n_test_sunny": int((s.is_test & is_sunny).sum()),
        "n_test_cloudy": int((s.is_test & ~is_sunny).sum()),
        "n_dev_days": int(len(set(dates[~s.is_test]))),
        "n_test_days": int(len(set(dates[s.is_test]))),
        "official_reference": {"roadmap_dev": 161928, "sunset_notebook_dev": 149680,
                               "sunset_notebook_test": 12248, "sunset_notebook_test_sunny": 6197},
        "content_sha256": digest.hexdigest(),
    }
    out_path.with_suffix(".json").write_text(json.dumps(info, indent=2))
    return info


class SkippdForecastDataset(Dataset):
    """Forecast samples from the frozen manifest, read from the benchmark HDF5.

    Returns (images [T,3,64,64] float in [0,1], pv_history [T] kW, target [1] kW).
    Frames are chronological (oldest first) unless ``newest_first`` (the official storage order).
    Per-sample metadata for evaluation: ``origin_time``, ``target_time``, ``day``, ``is_sunny``.
    """

    def __init__(
        self,
        h5_path: str | Path,
        manifest_path: str | Path,
        split: str = "dev",
        sample_ids: np.ndarray | None = None,
        preload: bool = True,
        newest_first: bool = False,
    ):
        if split not in ("dev", "test"):
            raise ValueError("split must be 'dev' or 'test'")
        m = np.load(manifest_path)
        in_split = m["is_test"] if split == "test" else ~m["is_test"]
        ids = np.flatnonzero(in_split) if sample_ids is None else np.asarray(sample_ids, dtype=np.int64)
        if not in_split[ids].all():
            raise ValueError(f"sample_ids include samples outside the {split} split")

        self.sample_ids = ids
        self.group_name = "test" if split == "test" else "trainval"
        rows = m["hdf5_row"]
        hist = m["history"][ids]
        if not newest_first:
            hist = hist[:, ::-1]
        self.hist_rows = np.ascontiguousarray(rows[hist])
        self.target_rows = rows[m["target"][ids]]
        times = m["times_us"].astype("datetime64[us]")
        self.origin_time = times[m["origin"][ids]]
        self.target_time = times[m["target"][ids]]
        self.day = self.origin_time.astype("datetime64[D]")
        self.is_sunny = m["is_sunny"][ids]

        self.h5_path = str(h5_path)
        with h5py.File(self.h5_path, "r") as f:
            g = f[self.group_name]
            self.pv = torch.from_numpy(g["pv_log"][...].astype(np.float32))
            self.images = torch.from_numpy(g["images_log"][...]) if preload else None
        self._h5 = None

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _read_frames(self, rows: np.ndarray) -> torch.Tensor:
        if self.images is not None:
            return self.images[torch.from_numpy(rows)]
        if self._h5 is None:  # opened lazily so each DataLoader worker gets its own handle
            self._h5 = h5py.File(self.h5_path, "r")
        uniq, inv = np.unique(rows, return_inverse=True)
        return torch.from_numpy(self._h5[self.group_name]["images_log"][uniq][inv])

    def __getitem__(self, i: int):
        rows = self.hist_rows[i]
        frames = self._read_frames(rows).permute(0, 3, 1, 2).float().div_(255.0)
        return frames, self.pv[torch.from_numpy(rows)], self.pv[self.target_rows[i]].reshape(1)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build the frozen SKIPP'D forecast sample manifest.")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out", default="data/manifests/forecast_tol30_v2.npz")
    p.add_argument("--tolerance-s", type=float, default=30.0)
    p.add_argument("--dev-sampling-min", type=int, default=2)
    p.add_argument("--test-sampling-min", type=int, default=1)
    a = p.parse_args()
    print(json.dumps(build_forecast_manifest(a.raw_dir, a.out, a.tolerance_s, a.dev_sampling_min,
                                             a.test_sampling_min), indent=2))
