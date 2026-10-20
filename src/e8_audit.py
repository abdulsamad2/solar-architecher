"""E8 CPU audit, part 1: history-window eligibility and the common evaluation population.

Windows keep 16 frames and stretch the stride: H15 = 1-min stride, H30 = 2-min, H60 = 4-min.
A sample is eligible for a window iff every frame t - stride*j (j = 0..15) has a record within +-30 s on the
same day, frames distinct and strictly ordered. Development samples only; the official test set is not read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.skippd import _nearest

WINDOWS = {"H15": 1, "H30": 2, "H60": 4}
N_FRAMES = 16
MIN_US, TOL_US = 60_000_000, 30_000_000


def history_rows(times_us: np.ndarray, origin_idx: np.ndarray, stride_min: int) -> np.ndarray:
    """[N, 16] timeline indices of frames t - stride*j (newest first); -1 if missing/other day/duplicate."""
    t0 = times_us[origin_idx]
    rows = np.stack([_nearest(times_us, t0 - stride_min * j * MIN_US, TOL_US) for j in range(N_FRAMES)], 1)
    day = times_us.astype("datetime64[us]").astype("datetime64[D]")
    ok = (rows >= 0)
    same_day = np.where(ok, day[np.maximum(rows, 0)] == day[origin_idx][:, None], False)
    ordered = np.concatenate([np.ones((len(rows), 1), bool), np.diff(rows, axis=1) < 0], 1)
    return np.where(ok & same_day & ordered, rows, -1)


def eligibility(manifest_path: str) -> dict:
    m = np.load(manifest_path)
    t = m["times_us"]
    dev = ~m["is_test"]
    origin = m["origin"][dev]
    out = {}
    for name, stride in WINDOWS.items():
        rows = history_rows(t, origin, stride)
        out[name] = (rows >= 0).all(1)
    return {"dev_index": np.flatnonzero(dev), "eligible": out, "origin_us": t[origin]}


def main():
    man = "data/manifests/forecast_tol30_v2.npz"
    folds = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    e = eligibility(man)
    origin_us = e["origin_us"]
    day = origin_us.astype("datetime64[us]").astype("datetime64[D]").astype(str)
    hour = ((origin_us.astype("datetime64[us]") - origin_us.astype("datetime64[us]").astype("datetime64[D]"))
            .astype("timedelta64[m]").astype(int)) / 60
    fold_of_day = {d: fd["fold"] for fd in folds["folds"] for d in fd["val_days"]}
    fold = np.array([fold_of_day.get(d, -1) for d in day])
    clear = np.array([folds["day_labels"][d] for d in day])
    # ramp stratum (Gate 2 definition) for validation samples
    from src.gate2_report import ramp_labels
    from src.skippd import SkippdForecastDataset
    ramp = ramp_labels("data/raw/2017_2019_images_pv_processed.hdf5", man, folds)["is_ramp"]
    ds = SkippdForecastDataset("data/raw/2017_2019_images_pv_processed.hdf5", man, "dev", preload=False)
    is_ramp = np.array([ramp.get(s, False) for s in ds.sample_ids])
    assert len(ds.sample_ids) == len(origin_us)

    el = e["eligible"]
    common = el["H15"] & el["H30"] & el["H60"]
    n = len(common)
    first_rec_hour = {}
    for d in np.unique(day):
        first_rec_hour[d] = hour[day == d].min()
    since_start = hour - np.array([first_rec_hour[d] for d in day])
    rep = {"n_dev_samples": int(n)}
    for name, mask in list(el.items()) + [("common", common)]:
        rep[name] = {"eligible": int(mask.sum()), "share": round(float(mask.mean()), 4),
                     "per_fold_val": {k: [int((mask & (fold == k)).sum()), int((fold == k).sum())] for k in range(5)},
                     "sunny_share_val": round(float(clear[mask & (fold >= 0)].mean()), 4),
                     "ramp_share_val": round(float(is_ramp[mask & (fold >= 0)].mean()), 4)}
    lost = ~common
    rep["lost_vs_H15"] = {"n": int((el["H15"] & ~common).sum()),
                          "share_of_H15": round(float((el["H15"] & ~common).sum() / el["H15"].sum()), 4),
                          "median_minutes_since_first_record": float(np.median(since_start[el["H15"] & ~common]) * 60),
                          "share_within_first_75min_of_day": round(float((since_start[el["H15"] & ~common] < 1.25).mean()), 4),
                          "ramp_share_lost_val": round(float(is_ramp[(el["H15"] & ~common) & (fold >= 0)].mean()), 4),
                          "hour_hist_lost": np.histogram(hour[el["H15"] & ~common], bins=np.arange(5, 21))[0].tolist()}
    rep["ramp_share_val_all_H15"] = round(float(is_ramp[el["H15"] & (fold >= 0)].mean()), 4)
    Path("reports/e8_eligibility_audit.json").write_text(json.dumps(rep, indent=1))
    np.savez_compressed("data/manifests/e8_eligibility_v1.npz", sample_ids=ds.sample_ids,
                        **{f"eligible_{k}": v for k, v in el.items()}, eligible_common=common)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
