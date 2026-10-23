"""Phase E preprocessing for the Gatton dataset (reports/phaseE_gatton_prereg.md, section 3).

Steps (images are inputs only; PV of sealed test days is never loaded here):
  circle   : fisheye centre/radius from the mean of 200 random noon images on DEVELOPMENT days (seed 20261023)
  index    : per PV minute (GMT+10), the nearest 10-s image within +-30 s (ties -> earlier)
  images   : crop [centre +- radius], zero outside the circle, BOX-resize to 64x64 RGB uint8 (all days, frozen params)
  pv       : development-day PV only (kW, duplicates -> first, negatives -> 0)
Outputs on D: (never in the repository): D:/solar_data/gatton/processed/{circle.json, records.npz, images_64.npy, pv_dev.npz}
"""
from __future__ import annotations

import json
import os
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"D:/solar_data/gatton")
ZIP = ROOT / "bag" / "data" / "2020.zip"
PV_CSV = ROOT / "bag" / "data" / "pv_2020.csv"
OUT = ROOT / "processed"
DEV_END = np.datetime64("2020-04-15")       # last development day (inclusive)
TEST_START = np.datetime64("2020-04-16")    # sealed test: 2020-04-16 .. 2020-05-06
DRAFT = (512, 384)                          # JPEG draft decode at 1/4 scale (2048x1536 -> 512x384)
SIZE = 64


def image_times(names) -> np.ndarray:
    """'2020-01-23/2020-01-23_13-32-10.jpg' -> datetime64[s] (local GMT+10)."""
    stem = [n.rsplit("/", 1)[-1][:-4] for n in names]
    return np.array([f"{s[:10]}T{s[11:13]}:{s[14:16]}:{s[17:19]}" for s in stem], dtype="datetime64[s]")


def _decode(data: bytes) -> np.ndarray:
    im = Image.open(__import__("io").BytesIO(data))
    im.draft("RGB", DRAFT)
    return np.asarray(im.convert("RGB"))


def fit_circle(z: zipfile.ZipFile, names: np.ndarray, times: np.ndarray) -> dict:
    day = times.astype("datetime64[D]")
    hour = (times - day).astype(int) / 3600
    cand = np.flatnonzero((day <= DEV_END) & (np.abs(hour - 12) <= 1))
    pick = np.random.default_rng(20261023).choice(cand, 200, replace=False)
    mean = np.mean([_decode(z.read(names[i])).astype(np.float32).mean(-1) for i in pick], axis=0)
    mask = mean > 0.1 * mean.max()
    # drop the corner text overlays: keep the largest blob by restricting to pixels far from the image border
    yy, xx = np.nonzero(mask)
    cy, cx = np.median(yy), np.median(xx)
    r = np.sqrt(mask.sum() / np.pi)
    keep = (yy - cy) ** 2 + (xx - cx) ** 2 <= (1.2 * r) ** 2
    cy, cx = yy[keep].mean(), xx[keep].mean()
    r = float(np.sqrt(keep.sum() / np.pi))
    return {"cx": float(cx), "cy": float(cy), "r": r, "draft_size": list(DRAFT), "n_images": 200, "seed": 20261023,
            "threshold": "mean > 0.1 * max", "dev_end": str(DEV_END)}


_Z, _C = None, None


def _init(circle):
    global _Z, _C
    _Z, _C = zipfile.ZipFile(ZIP), circle


def _process(name: str) -> np.ndarray:
    a = _decode(_Z.read(name)).astype(np.float32)
    cx, cy, r = _C["cx"], _C["cy"], _C["r"]
    x0, y0 = int(round(cx - r)), int(round(cy - r))
    s = int(round(2 * r))
    pad = np.zeros((s, s, 3), np.float32)
    ys, xs = slice(max(y0, 0), min(y0 + s, a.shape[0])), slice(max(x0, 0), min(x0 + s, a.shape[1]))
    pad[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = a[ys, xs]
    yy, xx = np.mgrid[0:s, 0:s]
    pad[(yy + 0.5 - s / 2) ** 2 + (xx + 0.5 - s / 2) ** 2 > (s / 2) ** 2] = 0.0
    return np.asarray(Image.fromarray(pad.round().clip(0, 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BOX))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ZIP)
    names = np.array(sorted(n for n in z.namelist() if n.endswith(".jpg")))
    t_img = image_times(names)
    assert (np.diff(t_img.astype(np.int64)) > 0).all(), "image times not strictly increasing"
    circle = fit_circle(z, names, t_img)
    (OUT / "circle.json").write_text(json.dumps(circle, indent=1))
    print("circle", circle, flush=True)
    # PV minute timeline (timestamps only for every day; values only for development days)
    import pandas as pd
    t_all = pd.to_datetime(pd.read_csv(PV_CSV, usecols=["DateTime"])["DateTime"],
                           format="%d/%m/%Y %I:%M:%S %p").to_numpy().astype("datetime64[s]")
    first = np.r_[True, t_all[1:] != t_all[:-1]]
    t_pv = t_all[first]
    day_pv = t_pv.astype("datetime64[D]")
    in_img = (day_pv >= t_img.min().astype("datetime64[D]")) & (day_pv <= t_img.max().astype("datetime64[D]"))
    t_rec = t_pv[in_img]
    # nearest image within +-30 s (ties -> earlier)
    ti, tr = t_img.astype(np.int64), t_rec.astype(np.int64)
    k = np.clip(np.searchsorted(ti, tr), 1, len(ti) - 1)
    k = np.where(np.abs(ti[k - 1] - tr) <= np.abs(ti[k] - tr), k - 1, k)
    ok = np.abs(ti[k] - tr) <= 30
    t_rec, img_idx = t_rec[ok], k[ok]
    print(f"paired 1-min records: {ok.sum():,} of {len(ok):,} PV minutes in the image period", flush=True)
    rec_day = t_rec.astype("datetime64[D]")
    is_test = rec_day >= TEST_START
    np.savez_compressed(OUT / "records.npz", times=t_rec.astype("datetime64[s]").astype(np.int64), image_name=names[img_idx],
                        image_offset_s=(ti[img_idx] - tr[ok]), is_test=is_test)
    # development PV values only: CSV rows of sealed test days (and later) are skipped while reading, never loaded
    dev_row = np.flatnonzero(t_all.astype("datetime64[D]") <= DEV_END)
    keep_rows = set((dev_row + 1).tolist()) | {0}  # +1 for the header line
    dfv = pd.read_csv(PV_CSV, skiprows=lambda i: i not in keep_rows, na_values=["(null)"])  # amendment E-1
    dfv["Power(kW)"] = pd.to_numeric(dfv["Power(kW)"], errors="coerce")
    t_dev = pd.to_datetime(dfv["DateTime"], format="%d/%m/%Y %I:%M:%S %p").to_numpy().astype("datetime64[s]")
    assert t_dev.max() < TEST_START, "STOP: a sealed test row was loaded"
    first_dev = np.r_[True, t_dev[1:] != t_dev[:-1]]
    t_dev, pv_dev = t_dev[first_dev], dfv["Power(kW)"].to_numpy(float)[first_dev]
    n_nan = int(np.isnan(pv_dev).sum())
    np.savez_compressed(OUT / "pv_dev.npz", times=t_dev.astype(np.int64), pv_kw=np.where(np.isnan(pv_dev), np.nan, np.maximum(pv_dev, 0.0)),
                        n_negative_clipped=int((pv_dev < 0).sum()), n_missing=n_nan)
    print(f"development PV rows (incl. PV-only January days): {len(t_dev):,}; negatives clipped {int((pv_dev < 0).sum())}; "
          f"missing {n_nan}", flush=True)
    # images for all paired records (inputs only), frozen parameters
    with Pool(16, initializer=_init, initargs=(circle,)) as pool:
        imgs = np.stack(pool.map(_process, names[img_idx], chunksize=256))
    np.save(OUT / "images_64.npy", imgs)
    print("images", imgs.shape, imgs.dtype, "| dev records", int((~is_test).sum()), "| test records", int(is_test.sum()), flush=True)


if __name__ == "__main__":
    main()
