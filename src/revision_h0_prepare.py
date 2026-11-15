"""Revision R-H0 (pilot): derive high-resolution inputs from the raw 2048 x 2048 SKIPP'D frames
(protocol revision/protocols/R-H0_resolution_pilot.md, amendment 1). CPU only; no model, no PV outcome.

For every benchmark development record of the pilot months, the latest raw frame at or before the record time (within
60 s; per-day clock offset of 0 or +-1 h chosen by image agreement) gives
  full256  the full frame area-averaged to 256 x 256,
  sun_hi   a 512 x 512 raw window centred on the sun (solar geometry + fisheye model) as 64 x 64,
  ring_hi  the sky ring at zenith 55-88 deg unwrapped to 32 x 256 (azimuth from the sun's azimuth),
and the controls sun_ctl / ring_ctl, the same samplings of the same raw frame after area-averaging to 64 x 64.
Raw frames live outside the repository (default D:/skippd_raw); the output is one HDF5 file (default D:/skippd_hires).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pvlib
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

LAT, LON = 37.427, -122.174  # Stanford; raw frame names are local wall-clock time (checked per day against the sun)
PILOT_MONTHS = ("2017-05", "2017-09", "2017-10", "2017-11", "2017-12", "2018-01", "2018-02", "2018-03", "2018-11", "2019-01", "2019-02", "2019-05")
RING_ZEN, RING_SHAPE, SUN_WIN = (55.0, 88.0), (32, 256), 512
OFFSETS = (0, 3600, -3600)
BASE_UNTIL = "2018-03"  # last month of the first camera-orientation epoch (sun residuals rise afterwards)


def raw_files(root, day):
    y, m, d = day.split("-")
    fs = sorted(glob.glob(os.path.join(root, y, m, d, "*.jpg")))
    t = np.array([int(f[-10:-8]) * 3600 + int(f[-8:-6]) * 60 + int(f[-6:-4]) for f in fs])
    return fs, t


def solar(day, sec, shift_s=0):
    """Solar zenith and azimuth for local wall-clock seconds of a day (plus a clock correction in seconds)."""
    t = pd.Timestamp(str(day)).tz_localize("America/Los_Angeles") + pd.to_timedelta(np.atleast_1d(sec) + shift_s, unit="s")
    sp = pvlib.solarposition.get_solarposition(t, LAT, LON)
    return sp.apparent_zenith.to_numpy(), sp.azimuth.to_numpy()


def project(cal, zen, az):
    """Sky direction (zenith, azimuth in degrees) -> raw pixel (x, y): tilted camera, r = f*theta + k*theta^3."""
    f, k, yaw, tx, ty = cal["p"]
    th, a = np.radians(zen), np.radians(az)
    d = np.stack([np.sin(th) * np.sin(a), np.sin(th) * np.cos(a), np.cos(th)], -1)
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    c = d @ (ry @ rx @ rz).T
    thc, phc = np.arccos(np.clip(c[..., 2], -1, 1)), np.arctan2(c[..., 0], c[..., 1])
    r = f * thc + k * thc ** 3
    return cal["cx"] + cal["mirror"] * r * np.sin(phc), cal["cy"] - r * np.cos(phc)


def for_day(cal, day):
    """Calibration valid on a day: the base epoch, or the orientation of the nearest later month that was refitted."""
    mth = str(day)[:7]
    pm = cal.get("p_by_month", {})
    if mth <= BASE_UNTIL or not pm:
        return cal
    near = min(pm, key=lambda k: abs((pd.Timestamp(k + "-15") - pd.Timestamp(mth + "-15")).days))
    return cal | {"p": pm[near]}


def _sun_obs(args):
    root, day = args
    fs, t = raw_files(root, day)
    out = []
    for f, s in list(zip(fs, t))[::10]:
        a = np.asarray(Image.open(f).convert("L").resize((512, 512), Image.BILINEAR))
        sat = a >= 250
        if 20 < sat.sum() < 6000:
            yy, xx = np.nonzero(sat)
            if xx.std() < 25 and yy.std() < 25:
                out.append((day, s, xx.mean() * 4.0, yy.mean() * 4.0))
    mx = np.max([np.asarray(Image.open(f).convert("L").resize((256, 256), Image.BILINEAR)) for f in fs[len(fs) // 4:3 * len(fs) // 4:40]], 0) if len(fs) > 200 else None
    return out, mx


def calibrate(root, days, workers):
    """Image circle from daytime maxima; tilt, yaw and radial law from compact sun blobs of all days (robust fit)."""
    with Pool(workers) as p:
        res = p.map(_sun_obs, [(root, d) for d in days])
    mx = np.max([m for _, m in res if m is not None], 0) > 40
    ys, xs = np.nonzero(mx)
    cal = {"cx": float(xs.mean() * 8), "cy": float(ys.mean() * 8), "radius": float(np.sqrt(mx.sum() / np.pi) * 8)}
    obs = [o for r, _ in res for o in r]
    oday = np.array([o[0] for o in obs])
    osec = np.array([o[1] for o in obs])
    x, y = np.array([o[2] for o in obs]), np.array([o[3] for o in obs])

    def angles(shift):  # per-observation clock correction (seconds)
        zen, az = np.empty(len(obs)), np.empty(len(obs))
        for d in np.unique(oday):
            m = oday == d
            for sh in np.unique(shift[m]):
                mm = m & (shift == sh)
                zen[mm], az[mm] = solar(d, osec[mm], int(sh))
        return zen, az

    def fit(zen, az):
        keep, best = zen < 75, None
        for mirror in (1, -1):
            for yaw0 in np.radians([0, 90, 180, 270]):
                c = cal | {"mirror": mirror}
                fun = lambda p: np.concatenate([project(c | {"p": p}, zen[keep], az[keep])[0] - x[keep], project(c | {"p": p}, zen[keep], az[keep])[1] - y[keep]])
                r = least_squares(fun, [cal["radius"] / (np.pi / 2), 0.0, yaw0, 0.0, 0.0], loss="soft_l1", f_scale=15)
                e = np.hypot(*(np.array(project(c | {"p": r.x}, zen, az)) - np.array([x, y])))
                if best is None or np.median(e[keep]) < best[0]:
                    best = (float(np.median(e[keep])), mirror, r.x.tolist())
        return best

    shift = np.zeros(len(obs), int)
    for _ in range(3):  # alternate: fit the lens, then pick each day's clock correction (0 or +-1 h) by the sun error
        zen, az = angles(shift)
        err, mirror, p = fit(zen, az)
        c = cal | {"mirror": mirror, "p": p}
        new = shift.copy()
        for d in np.unique(oday):
            m = oday == d
            if m.sum() < 5:
                continue
            es = {}
            for sh in OFFSETS:
                z2, a2 = solar(d, osec[m], sh)
                es[sh] = np.median(np.hypot(*(np.array(project(c, z2, a2)) - np.array([x[m], y[m]]))))
            new[m] = min(es, key=es.get)
        if np.array_equal(new, shift):
            break
        shift = new
    zen, az = angles(shift)
    keep = zen < 75
    month = np.array([d[:7] for d in oday])
    # the camera orientation changed over the years: lens law (f, k) and mirror from the first epoch (<= BASE_UNTIL),
    # orientation (yaw, two tilts) refitted for every later month with enough sun observations
    base = month <= BASE_UNTIL
    xa, ya, keep_all = x, y, keep
    x, y = xa[base], ya[base]
    err, mirror, p = fit(zen[base], az[base])
    x, y = xa, ya
    c = cal | {"mirror": mirror}
    by_month = {}
    for mth in sorted(set(month)):
        mm = (month == mth) & keep
        if mth <= BASE_UNTIL or mm.sum() < 30:
            continue
        fun = lambda o: np.concatenate([project(c | {"p": [p[0], p[1], *o]}, zen[mm], az[mm])[0] - x[mm], project(c | {"p": [p[0], p[1], *o]}, zen[mm], az[mm])[1] - y[mm]])
        by_month[str(mth)] = [p[0], p[1], *least_squares(fun, p[2:], loss="soft_l1", f_scale=15).x.tolist()]
    cal |= {"mirror": mirror, "p": p, "p_by_month": by_month}
    e = np.empty(len(x))
    for mth in set(month):
        mm = month == mth
        e[mm] = np.hypot(*(np.array(project(for_day(cal, mth + "-15"), zen[mm], az[mm])) - np.array([x[mm], y[mm]])))
    cal |= {"n_sun_obs": int(keep.sum()), "median_err_px": float(np.median(e[keep])),
            "median_err_px_by_month": {str(m): float(np.median(e[(month == m) & keep])) for m in sorted(set(month)) if ((month == m) & keep).sum() >= 5},
            "day_clock_shift_s": {str(d): int(shift[oday == d][0]) for d in np.unique(oday) if (oday == d).sum() >= 5}}
    return cal


def ring_grid(cal, zen_s, az_s, scale=1.0):
    zz = np.linspace(RING_ZEN[1], RING_ZEN[0], RING_SHAPE[0])  # horizon at the top row
    aa = az_s + (np.arange(RING_SHAPE[1]) + 0.5) / RING_SHAPE[1] * 360.0 - 180.0  # the sun's azimuth in the centre column
    x, y = project(cal, *np.meshgrid(zz, aa, indexing="ij"))
    return x * scale, y * scale


def sample(img, x, y):
    c = [map_coordinates(img[..., k], [y, x], order=1, mode="constant", cval=0.0) for k in range(3)]
    return np.clip(np.stack(c, -1), 0, 255).astype(np.uint8)


def _day(args):
    root, day, rows, tb, bench, cal = args
    cal = for_day(cal, day)
    fs, tr = raw_files(root, day)
    if not len(fs):
        return day, None
    gray = lambda a: a.astype(float).mean(-1)
    # clock offset of the benchmark relative to the raw names: best image agreement on a few frames
    score = {}
    probe = np.linspace(0, len(tb) - 1, min(12, len(tb))).astype(int)
    for off in OFFSETS:
        cc = []
        for i in probe:
            j = np.searchsorted(tr, tb[i] + off, side="right") - 1
            if j >= 0 and tb[i] + off - tr[j] <= 60:
                a = gray(np.asarray(Image.open(fs[j]).resize((64, 64), Image.BOX)))
                b = gray(bench[i])
                if a.std() > 1 and b.std() > 1:
                    cc.append(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        score[off] = float(np.mean(cc)) if len(cc) >= 3 else -1.0
    off = max(score, key=score.get)
    out = {k: [] for k in ("row", "raw_sec", "lag_s", "full256", "sun_hi", "sun_ctl", "ring_hi", "ring_ctl", "zen", "az")}
    j_all = np.searchsorted(tr, tb + off, side="right") - 1
    ok = (j_all >= 0) & (tb + off - tr[np.maximum(j_all, 0)] <= 60)
    known = cal["day_clock_shift_s"]
    near = min(known, key=lambda d: abs((pd.Timestamp(d) - pd.Timestamp(str(day))).days)) if known else None
    clock = known.get(str(day), known[near] if near else 0)
    zen, az = solar(day, tr[np.maximum(j_all, 0)], clock)
    half = SUN_WIN // 2
    for i in np.flatnonzero(ok):
        im = Image.open(fs[j_all[i]])
        raw = np.asarray(im).astype(np.float32)
        low = np.asarray(im.resize((64, 64), Image.BOX)).astype(np.float32)  # control: the same frame at benchmark resolution
        sx, sy = project(cal, zen[i], az[i])
        gx, gy = np.meshgrid(sx - half + (np.arange(64) + 0.5) * SUN_WIN / 64, sy - half + (np.arange(64) + 0.5) * SUN_WIN / 64)
        x0, y0 = int(round(sx)) - half, int(round(sy)) - half
        win = np.zeros((SUN_WIN, SUN_WIN, 3), np.float32)
        xa, xb, ya, yb = max(x0, 0), min(x0 + SUN_WIN, 2048), max(y0, 0), min(y0 + SUN_WIN, 2048)
        if xb > xa and yb > ya:
            win[ya - y0:yb - y0, xa - x0:xb - x0] = raw[ya:yb, xa:xb]
        out["sun_hi"].append(win.reshape(64, 8, 64, 8, 3).mean((1, 3)).astype(np.uint8))
        out["sun_ctl"].append(sample(low, gx / 32 - 0.5, gy / 32 - 0.5))
        rx, ry = ring_grid(cal, zen[i], az[i])
        out["ring_hi"].append(sample(raw, rx, ry))
        out["ring_ctl"].append(sample(low, rx / 32 - 0.5, ry / 32 - 0.5))
        out["full256"].append(np.asarray(im.resize((256, 256), Image.BOX)))
        out["row"].append(rows[i]); out["raw_sec"].append(tr[j_all[i]]); out["lag_s"].append(tb[i] + off - tr[j_all[i]])
        out["zen"].append(zen[i]); out["az"].append(az[i])
    return day, {"offset_s": off, "offset_scores": score, "n_bench": int(len(tb)), "n_matched": int(ok.sum()), "clock_shift_s": int(clock),
                 "arrays": {k: np.asarray(v) for k, v in out.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="D:/skippd_raw")
    ap.add_argument("--out", default="D:/skippd_hires/pilot_v1.h5")
    ap.add_argument("--months", nargs="*", default=list(PILOT_MONTHS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--calibration", default="revision/results/h0_fisheye_calibration.json")
    a = ap.parse_args()
    times = np.load("data/raw/times_trainval.npy", allow_pickle=True)
    day = np.array([t.strftime("%Y-%m-%d") for t in times])
    sec = np.array([t.hour * 3600 + t.minute * 60 + t.second for t in times])
    days = [d for d in sorted(set(day[np.isin([d[:7] for d in day], a.months)])) if raw_files(a.raw, d)[0]]
    print(len(days), "benchmark days with raw frames in", a.months, flush=True)
    cp = Path(a.calibration)
    if cp.exists():
        cal = json.loads(cp.read_text())
    else:
        cal = calibrate(a.raw, days, a.workers)
        cp.write_text(json.dumps(cal, indent=1))
    sh = np.array(list(cal["day_clock_shift_s"].values()))
    print("fisheye calibration: median sun error", round(cal["median_err_px"], 1), "px of 2048;", cal["n_sun_obs"], "sun observations; by month",
          {k: round(v, 1) for k, v in cal["median_err_px_by_month"].items()}, "| days with clock shift 0 / +1 h / -1 h:",
          int((sh == 0).sum()), int((sh == 3600).sum()), int((sh == -3600).sum()), flush=True)
    h5 = h5py.File("data/raw/2017_2019_images_pv_processed.hdf5", "r")["trainval/images_log"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    shapes = {"full256": (256, 256, 3), "sun_hi": (64, 64, 3), "sun_ctl": (64, 64, 3), "ring_hi": (*RING_SHAPE, 3), "ring_ctl": (*RING_SHAPE, 3)}
    report = {}
    with h5py.File(a.out, "w") as o, Pool(a.workers) as pool:
        ds = {k: o.create_dataset(k, (0, *s), maxshape=(None, *s), dtype="u1", chunks=(64, *s)) for k, s in shapes.items()}
        meta = {k: [] for k in ("row", "raw_sec", "lag_s", "zen", "az")}
        tasks = []
        for d in days:
            rows = np.flatnonzero(day == d)
            tasks.append((a.raw, d, rows, sec[rows], h5[rows[0]:rows[-1] + 1][rows - rows[0]], cal))
        for d, r in pool.imap(_day, tasks, chunksize=1):
            if r is None:
                continue
            arr = r.pop("arrays")
            report[d] = r
            n = len(arr["row"])
            for k in shapes:
                ds[k].resize(ds[k].shape[0] + n, 0)
                if n:
                    ds[k][-n:] = arr[k]
            for k in meta:
                meta[k].append(arr[k])
            print(d, "offset", r["offset_s"], "agreement", round(r["offset_scores"][r["offset_s"]], 3), "matched", r["n_matched"], "of", r["n_bench"], flush=True)
        for k in meta:
            o.create_dataset(k, data=np.concatenate(meta[k]))
        o.attrs["calibration"] = json.dumps(cal)
    tot = sum(r["n_matched"] for r in report.values())
    Path(a.out).with_suffix(".json").write_text(json.dumps({"n_records": tot, "days": report}, indent=1))
    print("records written", tot, "of", sum(r["n_bench"] for r in report.values()), "->", a.out)


if __name__ == "__main__":
    main()
