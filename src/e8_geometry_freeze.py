"""Fit and freeze the per-regime sky-camera geometry from images only (no PV, no day labels).

Frames: development records, every 3rd, solar zenith < 75 deg; sun = brightest 5x5-blurred spot with peak >= 240.
Regimes: before / from 2019-05-22 (camera change; daily R/B ratio 1.30 -> 2.29). Robust (soft-L1) fit.
Canonical form: f > 0, phi in (-pi, pi].
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from src.sky_geometry import fit_geometry, project, solar_position

CHANGE_DATE = "2019-05-22"


def canonical(g):
    if g["f"] < 0:
        g["f"], g["phi"] = -g["f"], float(np.arctan2(np.sin(g["phi"] + np.pi), np.cos(g["phi"] + np.pi)))
    return g


def brightest_spot(images):
    g = torch.from_numpy(images.astype(np.float32).mean(-1))[:, None]
    blur = F.avg_pool2d(g, 5, 1, 2)[:, 0].numpy().reshape(len(images), -1)
    am = blur.argmax(1)
    return (am % 64).astype(float), (am // 64).astype(float), blur.max(1)


def main():
    t = np.asarray(np.load("data/raw/times_trainval.npy", allow_pickle=True), dtype="datetime64[us]")
    sel = np.arange(0, len(t), 3)
    sp = solar_position(t[sel])
    zen, az = sp["zenith_deg"].to_numpy(), sp["azimuth_deg"].to_numpy()
    xs, ys, peak = [], [], []
    with h5py.File("data/raw/2017_2019_images_pv_processed.hdf5", "r") as f:
        ds = f["trainval/images_log"]
        for s in range(0, len(sel), 20000):
            x, y, p = brightest_spot(ds[sel[s:s + 20000]])
            xs.append(x); ys.append(y); peak.append(p)
    x, y, peak = np.concatenate(xs), np.concatenate(ys), np.concatenate(peak)
    post = t[sel].astype("datetime64[D]") >= np.datetime64(CHANGE_DATE)
    ok = (peak >= 240) & (zen < 75)
    out = {"change_date": CHANGE_DATE, "selection": "image-only: brightest 5x5-blurred spot >= 240, zenith < 75 deg, every 3rd dev record"}
    for name, m in (("pre", ok & ~post), ("post", ok & post)):
        g = canonical(fit_geometry(x[m], y[m], zen[m], az[m]))
        px, py = project(g, zen[m], az[m])
        err = np.hypot(px - x[m], py - y[m])
        out[name] = {**g, "n": int(m.sum()), "median_err_px": float(np.median(err)), "p90_err_px": float(np.percentile(err, 90)),
                     "share_err_gt_3px": float((err > 3).mean())}
    Path("reports/e8_geometry_frozen.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
