"""Phase E development-only setup for Gatton (prereg sections 3.4-3.8; amendments E-1, E-2).

1. time alignment check (image brightness vs PV first differences, lags -5..+5 min)
2. clear-sky geometry selection (fixed-tilt grid vs N-S single-axis tracker) and scale fit
3. day labels; 5 blocked folds with calibration split (development days only)
Uses ONLY development records (<= 2020-04-15). Output: reports/phaseE_gatton_setup.json, D:/.../processed/clearsky.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from src.folds import blocked_folds, calibration_split

P = Path(r"D:/solar_data/gatton/processed")
SITE = dict(latitude=-27.559, longitude=152.343, altitude=90.0, tz="Etc/GMT-10")
CAP_RATIO = 3275 / 30.1
MIN_POA = 300.0


def poa_clear(times: pd.DatetimeIndex, geom: dict) -> np.ndarray:
    loc = pvlib.location.Location(**SITE)
    sp = loc.get_solarposition(times)
    cs = loc.get_clearsky(times, model="ineichen", solar_position=sp)
    if geom["type"] == "tracker":
        tr = pvlib.tracking.singleaxis(sp["apparent_zenith"], sp["azimuth"], axis_tilt=0, axis_azimuth=0,
                                       max_angle=60, backtrack=False)
        tilt, az = tr["surface_tilt"].fillna(0), tr["surface_azimuth"].fillna(0)
    else:
        tilt, az = geom["tilt"], geom["azimuth"]
    poa = pvlib.irradiance.get_total_irradiance(tilt, az, sp["apparent_zenith"], sp["azimuth"], cs["dni"], cs["ghi"],
                                                cs["dhi"], dni_extra=pvlib.irradiance.get_extra_radiation(times),
                                                model="haydavies")
    aoi = pvlib.irradiance.aoi(tilt, az, sp["apparent_zenith"], sp["azimuth"])
    eff = poa["poa_direct"] * pvlib.iam.physical(aoi) + poa["poa_diffuse"]
    return np.clip(np.nan_to_num(eff.to_numpy()), 0, None)


def main():
    pv = np.load(P / "pv_dev.npz")
    t = pv["times"].astype("datetime64[s]")
    y = pv["pv_kw"]
    assert t.max() < np.datetime64("2020-04-16"), "STOP: non-development PV present"
    idx = pd.DatetimeIndex(t).tz_localize(SITE["tz"])
    day = t.astype("datetime64[D]")
    out = {"n_dev_pv_rows": int(len(t)), "n_missing": int(np.isnan(y).sum())}

    # 1. time alignment on development image records
    rec = np.load(P / "records.npz")
    imgs = np.load(P / "images_64.npy", mmap_mode="r")
    dev = np.flatnonzero(~rec["is_test"])
    rt = rec["times"][dev]
    bright = np.array([imgs[i].mean() for i in dev])
    pv_at = dict(zip(t.astype(np.int64), y))
    corr = {}
    for L in range(-5, 6):
        yb = np.array([pv_at.get(int(s) + 60 * L, np.nan) for s in rt])
        db, dy = np.diff(bright), np.diff(yb)
        consecutive = (np.diff(rt) == 60) & (np.diff(rt.astype("datetime64[s]").astype("datetime64[D]").astype(np.int64)) == 0)
        m = consecutive & np.isfinite(dy)
        corr[L] = float(np.corrcoef(db[m], dy[m])[0, 1])
    best_lag = max(corr, key=corr.get)
    out["alignment"] = {"corr_by_lag_min": corr, "best_lag_min": int(best_lag)}
    print("alignment corr by lag:", {k: round(v, 3) for k, v in corr.items()}, "-> best", best_lag, flush=True)

    # 2. geometry selection (amendment E-2)
    cands = [{"type": "fixed", "tilt": tl, "azimuth": az} for tl in range(0, 45, 5) for az in (0, 15, 345, 30, 330)]
    cands.append({"type": "tracker"})
    ok_days = [d for d in np.unique(day) if (np.isfinite(y[day == d]).sum() >= 300)]
    scores = []
    for g in cands:
        poa = poa_clear(idx, g)
        dc = []
        for d in ok_days:
            s = (day == d) & np.isfinite(y) & (poa > 0)
            if s.sum() >= 300:
                dc.append(np.corrcoef(y[s], poa[s])[0, 1])
        dc = np.array(dc)
        dc = np.sort(dc[np.isfinite(dc)])[::-1]  # undefined correlations (constant series) are dropped
        scores.append(float(dc[: max(1, int(0.2 * len(dc)))].mean()))
    best = int(np.argmax(np.round(scores, 6)))
    geom = cands[best]
    out["geometry"] = {"selected": geom, "score": scores[best],
                       "top5": sorted(zip(scores, map(json.dumps, cands)), reverse=True)[:5]}
    print("geometry:", geom, round(scores[best], 5), flush=True)
    # amendment E-4: mixed fixed/tracker shape with AC cap
    poa_f = poa_clear(idx, geom) if geom["type"] == "fixed" else poa_clear(idx, {"type": "fixed", "tilt": 0, "azimuth": 0})
    poa_t = poa_clear(idx, {"type": "tracker"})
    cap = float(np.nanquantile(y, 0.999))

    def fit_scale(shape):
        m = (shape >= MIN_POA) & np.isfinite(y)
        v = pd.Series(y[m] / shape[m]).groupby(day[m]).quantile(0.5).to_numpy()
        keep = np.ones(len(v), bool)
        for _ in range(3):
            sc = float(v[keep].mean())  # least-squares constant scale (ClearSkyPV rule, < 10 months)
            keep = (v - sc) / sc > -0.08
        return sc, int(keep.sum())

    mix = {}
    for r in np.round(np.arange(0, 1.01, 0.1), 1):
        shape = r * poa_f + (1 - r) * poa_t
        sc, nk = fit_scale(shape)
        pc = np.minimum(sc * shape, cap)
        dc = []
        for d in ok_days:
            s = (day == d) & np.isfinite(y) & (pc > 0)
            if s.sum() >= 300:
                dc.append(np.corrcoef(y[s], pc[s])[0, 1])
        dc = np.array(dc)
        dc = np.sort(dc[np.isfinite(dc)])[::-1]
        mix[float(r)] = {"score": float(dc[: max(1, int(0.2 * len(dc)))].mean()), "scale": sc, "n_days": nk}
    best_r = max(mix, key=lambda r: (round(mix[r]["score"], 6), r))
    sc = mix[best_r]["scale"]
    poa = best_r * poa_f + (1 - best_r) * poa_t
    out["mixture"] = {"by_r": mix, "selected_r": best_r, "cap_kw": cap, "fixed_component": geom}
    out["clear_sky_scale_kw_per_wm2"] = sc
    print("mixture r:", best_r, "score", round(mix[best_r]["score"], 5), "scale", round(sc, 4), "cap", round(cap, 1),
          {r: round(v["score"], 4) for r, v in mix.items()}, flush=True)

    # 3. day labels and folds (image days <= 2020-04-15)
    img_days = sorted({str(d) for d in rt.astype("datetime64[s]").astype("datetime64[D]")})
    pclear = np.minimum(sc * poa, cap)
    labels = {}
    for d in img_days:
        s = (day == np.datetime64(d)) & np.isfinite(y)
        c = np.corrcoef(y[s], pclear[s])[0, 1] if s.sum() > 2 else np.nan
        smooth = np.abs(np.diff(y[s])).mean() if s.sum() > 1 else np.nan
        labels[d] = bool(c > 0.995 and smooth < 0.15 * CAP_RATIO)
    folds = blocked_folds([np.datetime64(d).astype(object) for d in img_days], 5, 1)
    for fd in folds:
        for k in ("val_days", "train_days", "purged_days"):
            fd[k] = [str(x) for x in fd[k]]
        fit, cal, pur = calibration_split(fd["train_days"])
        fd["fit_days"], fd["cal_days"], fd["cal_purged_days"] = fit, cal, pur
    out["n_image_dev_days"] = len(img_days)
    out["n_sunny_dev_days"] = int(sum(labels.values()))
    out["folds"] = [{k: (len(v) if isinstance(v, list) else v) for k, v in fd.items()} for fd in folds]
    (P / "clearsky.json").write_text(json.dumps({"site": SITE, "fixed_component": geom, "tracker": {"axis_azimuth": 0, "max_angle": 60},
                                                  "r_fixed": best_r, "scale": sc, "cap_kw": cap, "min_poa": MIN_POA,
                                                  "c_min_kw": 1.0 * CAP_RATIO}, indent=1))
    Path("data/manifests").mkdir(exist_ok=True)
    Path("data/manifests/gatton_folds_v1.json").write_text(json.dumps({"day_labels": labels, "folds": folds}, indent=1))
    Path("reports/phaseE_gatton_setup.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("n_image_dev_days", "n_sunny_dev_days", "folds")}, indent=1))


if __name__ == "__main__":
    main()
