"""Paper figures (Phase D). Uses committed results and development data only; writes reports/figures/*.png|pdf.

F1 forecast example on one official test day     F2 per-day CRPS on the 20 test days
F3 Phase A2 comparison on SkyGPT's test samples  F4 clear-day PV / clear-sky ratio over time (decline)
F5 camera change: midday sky R/B ratio            F6 interval coverage: dev vs SKIPP'D test vs external
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("reports/figures")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9a9993"
H5 = "data/raw/2017_2019_images_pv_processed.hdf5"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "legend.frameon": False, "lines.linewidth": 1.6})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220)
    plt.close(fig)
    print("wrote", name)


def f1_test_day(day="2019-08-10"):
    from src.baselines import official_persistence
    from src.final_eval import MAN
    from src.skippd import SkippdForecastDataset
    p = np.load("experiments/final/test_predictions_v1.npz")
    ds = SkippdForecastDataset(H5, MAN, "test", preload=False)
    sel = ds.day.astype(str) == day
    pv = ds.pv.numpy().astype(float)
    y = pv[ds.target_rows][sel]
    q = p["q|G3''-F"][sel]
    pers = official_persistence(ds.origin_time[sel].astype(object), pv[ds.hist_rows[sel, -1]])
    t = ds.target_time[sel].astype("datetime64[s]").astype(object)
    o = np.argsort(t)
    t, y, q, pers = np.array(t)[o], y[o], q[o], pers[o]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.fill_between(t, q[:, 0], q[:, 8], color=BLUE, alpha=0.18, lw=0, label="G3″-F 5–95% (raw)")
    ax.fill_between(t, q[:, 2], q[:, 6], color=BLUE, alpha=0.32, lw=0, label="G3″-F 20–80% (raw)")
    ax.plot(t, pers, color=ORANGE, lw=1.1, ls="--", label="Smart persistence")
    ax.plot(t, q[:, 4], color=BLUE, lw=1.4, label="G3″-F median")
    ax.plot(t, y, color=INK, lw=1.0, label="Observed PV")
    ax.set_ylabel("PV power at t + 15 min (kW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title(f"Official test day {day} (cloudy): 15-min-ahead forecast", loc="left", fontsize=10, color=INK)
    ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    save(fig, "F1_test_day_forecast")


def f2_per_day():
    r = json.load(open("reports/test_posthoc_descriptive.json"))["per_day"]
    days = sorted(r, key=lambda d: r[d]["crps|persistence"])
    series = [("G3″-F", "crps|G3''-F", BLUE, "o"), ("E4-B-fit", "crps|E4-B-fit", AQUA, "s"),
              ("SUNSET port", "crps|SUNSET port", YELLOW, "D"), ("Smart persistence", "crps|persistence", ORANGE, "^")]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    yy = np.arange(len(days))
    for j, (name, key, c, mk) in enumerate(series):
        ax.scatter([r[d][key] for d in days], yy + (1.5 - j) * 0.16, s=26, color=c, marker=mk, label=name, zorder=3,
                   edgecolor=SURF, linewidth=0.8)
    ax.set_yticks(yy, [f"{d}  {'sunny' if r[d]['sunny'] else 'cloudy'}" for d in days], fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("Mean CRPS per day (kW, log scale; lower is better)")
    ax.set_title("Per-day CRPS on the 20 official SKIPP'D test days", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7.5, loc="lower right")
    save(fig, "F2_per_day_crps")


def f3_phaseA():
    a2 = json.load(open("reports/phaseA_a2_results.json"))
    rows = [("G3″-F (ours, run once)", a2["crps"]["G3''-F"]["overall"], BLUE),
            ("SkyGPT (published)", 2.81, GREY), ("SUNSET (published)", 3.31, GREY), ("Smart persistence (published)", 3.67, GREY)]
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    yy = np.arange(len(rows))[::-1]
    ax.barh(yy, [r[1] for r in rows], color=[r[2] for r in rows], height=0.55)
    for yv, (name, v, _) in zip(yy, rows):
        ax.text(v + 0.04, yv, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(yy, [r[0] for r in rows])
    ax.set_xlim(0, 4.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("CRPS (kW), lower is better")
    ax.set_title("SkyGPT's external test (5 cloudy days, Nov–Dec 2019)", loc="left", fontsize=10, color=INK)
    save(fig, "F3_phaseA_skygpt_comparison")


def _dev_times_pv():
    t = np.asarray(np.load("data/raw/times_trainval.npy", allow_pickle=True), dtype="datetime64[s]")
    with h5py.File(H5, "r") as f:
        pv = f["trainval/pv_log"][...].astype(float)
    return t, pv


def f4_clear_ratio():
    from src.official.relative_op_func import Relative_output
    t, pv = _dev_times_pv()
    folds = json.load(open("data/manifests/folds_blocked5_v1.json"))
    clear = {d for d, s in folds["day_labels"].items() if s}
    day = t.astype("datetime64[D]").astype(str)
    hour = (t - t.astype("datetime64[D]")).astype(int) / 3600
    sel = np.flatnonzero(np.isin(day, list(clear)) & (hour >= 11) & (hour <= 13))
    ratio = {}
    for i in sel:
        _, p_theo = Relative_output(t[i].astype(object), 0.0)
        if p_theo > 5:
            ratio.setdefault(day[i], []).append(pv[i] / p_theo)
    d = np.array(sorted(ratio), dtype="datetime64[D]")
    r = np.array([np.median(ratio[str(x)]) for x in d])
    years = (d - d[0]).astype(int) / 365.25
    naive_slope, naive_icpt = np.polyfit(years, r, 1)
    # season-adjusted trend: linear term + annual harmonic (clear days are unevenly spread over seasons)
    doy = 2 * np.pi * (d - d.astype("datetime64[Y]")).astype(int) / 365.25
    X = np.column_stack([np.ones_like(years), years, np.cos(doy), np.sin(doy)])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    resid = r - X @ beta
    se = np.sqrt(resid.var(ddof=4) * np.linalg.inv(X.T @ X)[1, 1])
    slope, icpt = beta[1], beta[0]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    dd = d.astype(object)
    ax.scatter(dd, r, s=9, color=BLUE, alpha=0.7, edgecolor="none", label="Clear day (11:00–13:00 median)")
    fit = (X @ beta).astype(float)
    gap = np.r_[False, np.diff(d).astype(int) > 30]  # do not draw the fit across data gaps
    fx, fy = [], []
    for x_, y_, g_ in zip(dd, fit, gap):
        if g_:
            fx.append(None), fy.append(np.nan)
        fx.append(x_), fy.append(y_)
    fx = [x_ if x_ is not None else fx[i - 1] for i, x_ in enumerate(fx)]
    ax.plot(fx, fy, color=GREY, lw=1.0, label="Fit: trend + annual cycle")
    ax.plot([dd[0], dd[-1]], [icpt, icpt + slope * years[-1]], color=ORANGE, lw=1.6,
            label=f"Season-adjusted trend: {100 * slope / icpt:+.1f}% per year")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_ylabel("PV / theoretical clear-sky PV")
    ax.set_title("Clear-day output relative to the fixed clear-sky model, development data 2017–2019", loc="left",
                 fontsize=10, color=INK)
    ax.legend(fontsize=7.5, loc="lower left")
    save(fig, "F4_clear_day_ratio_decline")
    return {"season_adjusted_relative_per_year": float(slope / icpt), "season_adjusted_se_relative": float(se / icpt),
            "naive_linear_relative_per_year": float(naive_slope / naive_icpt), "n_days": int(len(d))}


def f5_camera():
    t, _ = _dev_times_pv()
    day = t.astype("datetime64[D]")
    hour = (t - day).astype(int) / 3600
    idx = np.flatnonzero((hour >= 11.75) & (hour <= 12.25))
    rb = {}
    with h5py.File(H5, "r") as f:
        imgs = f["trainval/images_log"]
        for s in range(0, len(idx), 4000):
            chunk = idx[s:s + 4000]
            x = imgs[chunk].astype(np.float32)
            inside = x.sum(-1) > 30
            r_ = np.where(inside, x[..., 0], np.nan)
            b_ = np.where(inside, x[..., 2], np.nan)
            vals = np.nanmean(r_, (1, 2)) / np.maximum(np.nanmean(b_, (1, 2)), 1e-3)
            for i, v in zip(chunk, vals):
                rb.setdefault(day[i], []).append(v)
    d = np.array(sorted(rb))
    v = np.array([np.median(rb[x]) for x in d])
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.scatter(d.astype(object), v, s=8, color=BLUE, alpha=0.7, edgecolor="none")
    change = dt.date(2019, 5, 22)
    ax.axvline(change, color=ORANGE, lw=1.4, ls="--")
    ax.text(change, ax.get_ylim()[1], "  camera change 2019-05-22", color=INK, fontsize=8, va="top")
    ax.set_ylabel("Midday sky R/B ratio (daily median)")
    ax.set_title("Camera change in SKIPP'D development images", loc="left", fontsize=10, color=INK)
    save(fig, "F5_camera_change_rb")


def f6_coverage():
    dev = json.load(open("reports/final_rehearsal_dev.json"))["intervals"]["G3''-F"]
    test = json.load(open("reports/test_results.json"))["intervals"]["G3''-F"]
    ext = json.load(open("reports/phaseA_a2_results.json"))["intervals_calibrated"]["G3''-F"]
    nom = np.array([50, 80, 90])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, stratum in zip(axes, ("overall", "ramp")):
        ax.plot([40, 100], [40, 100], color=GREY, lw=1, ls=":")
        for name, src, c, mk in (("Development (out-of-fold)", dev, BLUE, "o"), ("SKIPP'D test", test, AQUA, "s"),
                                 ("SkyGPT external test", ext, ORANGE, "^")):
            cov = [100 * src[str(n)][stratum]["coverage"] for n in nom]
            ax.plot(nom, cov, color=c, marker=mk, ms=6, label=name)
        ax.set_xticks(nom)
        ax.set_xlim(45, 95)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Nominal coverage (%)")
        ax.set_title(f"{stratum.capitalize()} samples", loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("Observed coverage (%)")
    axes[0].legend(fontsize=7.5, loc="upper left")
    save(fig, "F6_interval_coverage")


def main():
    f1_test_day()
    f2_per_day()
    f3_phaseA()
    trend = f4_clear_ratio()
    f5_camera()
    f6_coverage()
    (OUT / "figure_numbers.json").write_text(json.dumps({"F4_trend": trend}, indent=1))
    print(trend)


if __name__ == "__main__":
    main()
