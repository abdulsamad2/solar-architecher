"""Canonical manifest validation, causal window construction, and a PyTorch Dataset.

The canonical manifest has one row per paired observation:
    timestamp_utc, image_path, pv_kw, split
plus an optional ``day`` column (local calendar day). ``day`` matters because
Stanford daytime crosses UTC midnight (17:00 PDT == 00:00 UTC), so UTC dates
must not be used as day keys.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

REQUIRED_COLUMNS = ("timestamp_utc", "image_path", "pv_kw", "split")
VALID_SPLITS = {"train", "val", "test"}


class ManifestError(ValueError):
    pass


def load_manifest(path: str | Path, local_tz: str = "America/Los_Angeles") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    if "day" not in df.columns:
        df["day"] = df["timestamp_utc"].dt.tz_convert(local_tz).dt.date.astype(str)
    return df


def validate_manifest(df: pd.DataFrame, check_images: bool = False, root: str | Path = ".") -> list[str]:
    """Return a list of problems (empty == valid). Never mutates ``df``."""
    problems: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]

    ts = df["timestamp_utc"]
    if not pd.api.types.is_datetime64_any_dtype(ts) or ts.dt.tz is None:
        problems.append("timestamp_utc must be timezone-aware datetime (UTC)")
    elif str(ts.dt.tz) != "UTC":
        problems.append(f"timestamp_utc timezone is {ts.dt.tz}, expected UTC")

    n_dup = int(ts.duplicated().sum())
    if n_dup:
        problems.append(f"{n_dup} duplicated timestamps")

    pv = pd.to_numeric(df["pv_kw"], errors="coerce").to_numpy(dtype=float)
    n_bad = int((~np.isfinite(pv)).sum())
    if n_bad:
        problems.append(f"{n_bad} non-finite pv_kw values")

    bad_split = set(df["split"].unique()) - VALID_SPLITS
    if bad_split:
        problems.append(f"unknown split labels: {sorted(bad_split)}")

    if "day" in df.columns:
        splits_per_day = df.groupby("day")["split"].nunique()
        leaky = splits_per_day[splits_per_day > 1].index.tolist()
        if leaky:
            problems.append(f"{len(leaky)} days appear in more than one split, e.g. {leaky[:3]}")

    if check_images:
        root = Path(root)
        n_missing = sum(not (root / p).is_file() for p in df["image_path"])
        if n_missing:
            problems.append(f"{n_missing} image paths do not resolve")
    return problems


@dataclass(frozen=True)
class Windows:
    history: np.ndarray  # [N, history_len] row indices, oldest first, last == origin
    target: np.ndarray   # [N, n_horizons] row indices strictly after origin


def window_indices(
    timestamps: Sequence | pd.Series,
    history_len: int = 15,
    step: pd.Timedelta = pd.Timedelta(minutes=1),
    horizons_minutes: Sequence[int] = (15,),
    day: Sequence | None = None,
) -> Windows:
    """Build windows by exact timestamp joins; any missing frame drops the window.

    Prototype convention: history = origin - (history_len-1)*step ... origin,
    targets = origin + h minutes. This must be reconciled with the official
    SKIPP'D forecasting rule before claiming benchmark parity.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if ts.has_duplicates:
        raise ManifestError("duplicated timestamps; deduplicate before windowing")
    pos = pd.Series(np.arange(len(ts)), index=ts)
    day_arr = None if day is None else np.asarray(day)

    hist_offsets = [step * k for k in range(-(history_len - 1), 1)]
    tgt_offsets = [pd.Timedelta(minutes=int(h)) for h in horizons_minutes]
    if any(o <= pd.Timedelta(0) for o in tgt_offsets):
        raise ValueError("horizons must be strictly positive")

    hist_idx = np.stack([pos.reindex(ts + o).to_numpy(dtype=float) for o in hist_offsets], axis=1)
    tgt_idx = np.stack([pos.reindex(ts + o).to_numpy(dtype=float) for o in tgt_offsets], axis=1)
    ok = ~np.isnan(hist_idx).any(axis=1) & ~np.isnan(tgt_idx).any(axis=1)
    hist_idx = np.nan_to_num(hist_idx, nan=-1).astype(np.int64)
    tgt_idx = np.nan_to_num(tgt_idx, nan=-1).astype(np.int64)

    if day_arr is not None:
        same_day = (day_arr[hist_idx] == day_arr[hist_idx[:, -1:]]).all(axis=1) & (
            day_arr[tgt_idx] == day_arr[hist_idx[:, -1:]]
        ).all(axis=1)
        ok &= same_day
    return Windows(history=hist_idx[ok], target=tgt_idx[ok])


def _default_image_loader(path: Path) -> torch.Tensor:
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    with Image.open(path) as im:
        return to_tensor(im.convert("RGB"))


class SkyPVDataset(Dataset):
    """Returns (images [T,3,H,W], pv_history [T], target [n_horizons]) in kW."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        windows: Windows,
        root: str | Path = ".",
        transform: Callable | None = None,
        image_loader: Callable[[Path], torch.Tensor] = _default_image_loader,
    ):
        self.paths = [Path(root) / p for p in manifest["image_path"]]
        self.pv = torch.tensor(manifest["pv_kw"].to_numpy(dtype=np.float32))
        self.windows = windows
        self.transform = transform
        self.image_loader = image_loader

    def __len__(self) -> int:
        return len(self.windows.history)

    def __getitem__(self, i: int):
        hist = self.windows.history[i]
        frames = []
        for j in hist:
            img = self.image_loader(self.paths[j])
            frames.append(self.transform(img) if self.transform else img)
        return torch.stack(frames), self.pv[hist], self.pv[self.windows.target[i]]
