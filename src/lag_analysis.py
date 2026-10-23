"""Post-hoc descriptive lag analysis on committed predictions (SKIPP'D 20-day test; SkyGPT external test).

For each cloudy day, the forecast series (median, aligned at target time t+15) is cross-correlated with the observed
series after first-differencing; the lag (minutes) maximising the correlation is the forecast's effective delay.
Smart persistence has a 15-min delay by construction. Output: reports/lag_analysis.json + reports/figures/F7_lag.*
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from src.baselines import official_persistence
from src.final_eval import H5, MAN
from src.skippd import SUNNY_DATES, SkippdForecastDataset

LAGS = np.arange(-5, 21)  # minutes; positive = forecast lags the observation


def best_lag(t_min, f, y, step=1):
    """t_min: integer minutes of the target time within a day; series on a `step`-minute grid (gaps allowed).
    Returns (best lag in minutes, corr at lag 0, corr at lag 15)."""
    keep = (t_min - t_min.min()) % step == 0
    t_min, f, y = t_min[keep], f[keep], y[keep]
    n = (t_min.max() - t_min.min()) // step + 1
    F, Y = np.full(n, np.nan), np.full(n, np.nan)
    idx = (t_min - t_min.min()) // step
    F[idx], Y[idx] = f, y
    dF, dY = np.diff(F), np.diff(Y)
    lags = [L for L in LAGS if L % step == 0]
    corr = {}
    for L in lags:
        k = L // step
        a, b = (dF[k:], dY[:len(dY) - k]) if k >= 0 else (dF[:k], dY[-k:])
        m = np.isfinite(a) & np.isfinite(b)
        corr[L] = np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 30 else np.nan
    best = max((L for L in corr if np.isfinite(corr[L])), key=lambda L: corr[L])
    c15 = corr.get(15, corr.get(14, np.nan))
    return int(best), float(corr[0]), float(c15)


def per_day(day, t, series, y, step=1):
    res = {}
    for d in np.unique(day):
        s = day == d
        tm = ((t[s] - t[s].astype("datetime64[D]")).astype("timedelta64[m]").astype(int))
        o = np.argsort(tm)
        res[d] = {name: best_lag(tm[o], f[s][o], y[s][o], step) for name, f in series.items()}
    return res


def summary(r, names):
    return {k: {"median_best_lag_min": float(np.median([r[d][k][0] for d in r])),
                "mean_corr_lag0": float(np.mean([r[d][k][1] for d in r])),
                "mean_corr_lag15": float(np.mean([r[d][k][2] for d in r]))} for k in names}


def main():
    out = {}
    p = np.load("experiments/final/test_predictions_v1.npz")
    ds = SkippdForecastDataset(H5, MAN, "test", preload=False)
    pv = ds.pv.numpy().astype(float)
    y = pv[ds.target_rows]
    day = ds.day.astype(str)
    cloudy = np.array([np.datetime64(d).astype(object) not in SUNNY_DATES for d in day])
    pers = official_persistence(ds.origin_time.astype(object), pv[ds.hist_rows[:, -1]])
    series = {"G3''-F": p["q|G3''-F"][:, 4], "E4-B-fit": p["q|E4-B-fit"][:, 4], "E6-S": p["q|E6-S"][:, 4],
              "SUNSET port": p["sunset"], "smart persistence": pers}
    t = ds.target_time.astype("datetime64[m]")
    r = per_day(day[cloudy], t[cloudy], {k: v[cloudy] for k, v in series.items()}, y[cloudy])
    out["SKIPPD test (10 cloudy days)"] = {"per_day": r, "summary": summary(r, series)}
    a = np.load("experiments/phaseA/a2_predictions.npz")
    with h5py.File("data/external/skygpt/test_set_2019nov_dec.hdf5", "r") as f:
        ye = f["test/pv_pred"][:, 14].astype(float)
    te = (a["origin_time"].astype("datetime64[m]") + np.timedelta64(15, "m"))
    de = te.astype("datetime64[D]").astype(str)
    se = {"G3''-F": a["q|G3''-F"][:, 4], "E4-B-fit": a["q|E4-B-fit"][:, 4], "E6-S": a["q|E6-S"][:, 4],
          "SUNSET port": a["sunset_members"].mean(0), "smart persistence": a["persistence"]}
    re_ = per_day(de, te, se, ye, step=2)
    out["SkyGPT external test (5 cloudy days; 2-min sampling)"] = {"per_day": re_, "summary": summary(re_, se)}
    Path("reports/lag_analysis.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print("==", k)
        for m, s in v["summary"].items():
            print(f"  {m:18s} best lag {s['median_best_lag_min']:5.1f} min | corr@0 {s['mean_corr_lag0']:+.3f} | corr@15 {s['mean_corr_lag15']:+.3f}")


if __name__ == "__main__":
    main()
