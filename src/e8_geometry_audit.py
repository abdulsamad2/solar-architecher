"""E8 CPU audit, part 2: sky-image orientation and fisheye projection (development clear days only)."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from src.sky_geometry import detect_sun, fit_geometry, project, solar_position


def main():
    times = np.load("data/raw/times_trainval.npy", allow_pickle=True)
    t = np.asarray(times, dtype="datetime64[us]")
    labels = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())["day_labels"]
    day = t.astype("datetime64[D]").astype(str)
    clear = np.array([labels.get(d, False) for d in day])
    sel = np.flatnonzero(clear)[::5]  # every 5th record on clear days
    with h5py.File("data/raw/2017_2019_images_pv_processed.hdf5", "r") as f:
        imgs = f["trainval/images_log"][np.sort(sel)]
    sel = np.sort(sel)
    sx, sy = detect_sun(imgs)
    sp = solar_position(t[sel])
    zen, az = sp["zenith_deg"].to_numpy(), sp["azimuth_deg"].to_numpy()
    ok = np.isfinite(sx) & (zen < 75)
    year = t[sel].astype("datetime64[Y]").astype(int) + 1970
    month = t[sel].astype("datetime64[M]").astype(str)
    shift = month >= "2019-05"
    rep = {"n_frames_examined": int(len(sel)), "n_sun_detections": int(ok.sum())}
    fits = {}
    for name, mask in (("all_pre_2019-05", ok & ~shift), ("2017", ok & (year == 2017)), ("2018", ok & (year == 2018)),
                       ("2019_pre_05", ok & (year == 2019) & ~shift), ("2019_post_05", ok & shift)):
        if mask.sum() < 50:
            fits[name] = {"n": int(mask.sum())}
            continue
        g = fit_geometry(sx[mask], sy[mask], zen[mask], az[mask])
        px, py = project(g, zen[mask], az[mask])
        err = np.hypot(px - sx[mask], py - sy[mask])
        fits[name] = {**g, "n": int(mask.sum()), "median_err_px": float(np.median(err)), "p90_err_px": float(np.percentile(err, 90))}
    rep["fits"] = fits
    # cross-period transfer: pre-shift geometry applied to post-shift detections
    if "cx" in fits.get("all_pre_2019-05", {}) and fits["2019_post_05"].get("n", 0) >= 50:
        m = ok & shift
        px, py = project(fits["all_pre_2019-05"], zen[m], az[m])
        rep["pre_geometry_on_post_shift_median_err_px"] = float(np.median(np.hypot(px - sx[m], py - sy[m])))
    # orientation sanity: azimuth of sun in image at morning vs afternoon (pre-shift fit)
    g = fits["all_pre_2019-05"]
    for label, az_deg in (("east_az90", 90), ("south_az180", 180), ("west_az270", 270)):
        x, y = project(g, 45.0, az_deg)
        rep[f"pixel_at_zenith45_{label}"] = [round(float(x), 1), round(float(y), 1)]
    Path("reports/e8_geometry_audit.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
