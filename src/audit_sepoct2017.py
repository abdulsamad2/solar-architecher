"""Descriptive CPU-only audit of the Sep-Oct 2017 coverage collapse (fold 1). Changes no Phase 6 rule.

Compares Sep-Oct 2017 with Sep-Oct 2018 and 2019 (and other months) on development data only:
image statistics (11-14 h local), PV / clear-sky index, weather mix, and the direction of E4-B-fit misses.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.clear_sky import ClearSkyPV

RAW = Path("data/raw")


def image_stats(h5_path) -> pd.DataFrame:
    with h5py.File(h5_path, "r") as f:
        ds = f["trainval/images_log"]
        rows = []
        for s in range(0, ds.shape[0], 20000):
            x = ds[s:s + 20000].astype(np.float32)  # [n,64,64,3]
            disk = x[:, 8:56, 8:56, :]               # inside the fisheye circle, away from the horizon mask
            r, g, b = (disk[..., c].mean(axis=(1, 2)) for c in range(3))
            sat = (disk.max(axis=-1) >= 250).mean(axis=(1, 2))
            rows.append(np.stack([r, g, b, sat], 1))
    return pd.DataFrame(np.concatenate(rows), columns=["R", "G", "B", "sat_frac"])


def main():
    times = np.load(RAW / "times_trainval.npy", allow_pickle=True)
    t = pd.to_datetime(np.asarray(times, dtype="datetime64[s]"))
    with h5py.File(RAW / "2017_2019_images_pv_processed.hdf5", "r") as f:
        pv = f["trainval/pv_log"][...].astype(float)
    folds = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    labels = folds["day_labels"]
    # anchor fitted on all dev days EXCEPT Sep-Oct of any year (descriptive reference, not a model input)
    ref = ~t.month.isin([9, 10])
    pc = ClearSkyPV().fit(t[ref].values, pv[ref])(t.values)
    k = pv / np.maximum(pc, 1.0)

    df = pd.DataFrame({"t": t, "pv": pv, "pc": pc, "k": k}).join(image_stats(RAW / "2017_2019_images_pv_processed.hdf5"))
    df["day"] = df.t.dt.date.astype(str)
    df["ym"] = df.t.dt.strftime("%Y-%m")
    df["clear_day"] = df.day.map(labels)
    df["rb"] = df.R / df.B.clip(lower=1)
    mid = df[(df.t.dt.hour >= 11) & (df.t.dt.hour < 14)]

    def summarise(g):
        return pd.Series({
            "days": g.day.nunique(),
            "clear_day_share": g.drop_duplicates("day").clear_day.mean(),
            "midday_K_median": g.k.median(),
            "midday_K_clear_days": g[g.clear_day].k.median(),
            "midday_R/B": g.rb.median(),
            "midday_R/B_clear_days": g[g.clear_day].rb.median(),
            "midday_brightness": g[["R", "G", "B"]].mean(axis=1).median(),
            "midday_saturated_frac": g.sat_frac.median(),
        })

    sel = mid[mid.t.dt.month.isin([9, 10])]
    table = sel.groupby("ym").apply(summarise).round(3)
    allm = mid.groupby("ym").apply(summarise).round(3)

    # daily view for Oct 2017 (North Bay fires: Oct 8-31, 2017) vs Oct 2018/2019
    daily = mid[mid.t.dt.month == 10].groupby("day").agg(K=("k", "median"), RB=("rb", "median"),
                                                         clear=("clear_day", "first"))

    # direction of E4-B-fit misses (seeds 42/123/456) by month, fold-validation samples only
    miss = []
    for s in (42, 123, 456):
        for kf in range(5):
            z = np.load(f"experiments/E4_calib/v1/csi/seed{s}/fold_{kf}/preds_val.npz")
            ym = pd.to_datetime(z["origin_time_us"].astype("datetime64[us]")).strftime("%Y-%m")
            q = z["pred_q"]
            miss.append(pd.DataFrame({"ym": ym, "below_q05": z["target"] < q[:, 0], "above_q95": z["target"] > q[:, -1],
                                      "err_median": z["target"] - q[:, 4], "width90": q[:, -1] - q[:, 0],
                                      "pclear": z["pclear_target_kw"], "seed": s}))
    miss = pd.concat(miss)
    mm = miss.groupby("ym").agg(below_q05=("below_q05", "mean"), above_q95=("above_q95", "mean"),
                                median_error_kw=("err_median", "median"), width90_kw=("width90", "median"),
                                n=("seed", "size")).round(3)
    out = {"sep_oct_by_year": table.to_dict("index"), "all_months_midday": allm.to_dict("index"),
           "oct_daily": daily.round(3).to_dict("index"), "e4bfit_miss_direction_by_month": mm.to_dict("index")}
    Path("reports/sepoct2017_audit.json").write_text(json.dumps(out, indent=1, default=float))
    pd.set_option("display.width", 200)
    print(table.to_string())
    print()
    print(mm.loc[[i for i in mm.index if i[5:] in ("08", "09", "10", "11", "12")]].to_string())
    print()
    print(daily.loc[[d for d in daily.index if d.startswith("2017-10")]].round(3).to_string())
    print()
    print("Oct 2018/2019 daily R/B median (clear days):",
          round(float(daily.loc[[d for d in daily.index if not d.startswith("2017") and daily.loc[d, "clear"]]].RB.median()), 3))


if __name__ == "__main__":
    main()
