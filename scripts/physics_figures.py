"""Phase F figures (development out-of-fold results). Called by scripts/evaluate_physics_ablation.py."""
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
INK, INK2, GRID, SURF, GREY = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#9a9993"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]  # fixed categorical order
NAMES = {"A0": "Original G3''-F", "A1": "+Physics", "R": "+Ramp", "A3": "+Phys+Ramp", "A4": "+Phys+Ramp+Bounds",
         "A5": "+...+Direction"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "legend.frameon": False})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220)
    plt.close(fig)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    def box(x, y, w, h, text, color=SERIES[0], fill="#eaf2fc"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fill, ec=color, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color=INK)

    def arrow(x0, y0, x1, y1, color=INK2):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, lw=1.1))

    box(0.2, 8.2, 2.6, 1.1, "Sky images\n16 frames, 60 min")
    box(0.2, 6.2, 2.6, 1.1, "3D-CNN + GRU\n(ramp expert A2)")
    box(0.2, 3.9, 2.6, 1.1, "Recent K history\n+ sky images (15 min)")
    box(3.4, 3.9, 2.4, 1.1, "Event-mixture expert\n(E6-S, calm)")
    box(3.4, 6.2, 2.4, 1.1, "Disagreement gate\nw = σ(β·[g, d])")
    box(6.4, 6.2, 2.6, 1.1, "9 K quantiles\nK̂(t+15, q)")
    box(6.4, 4.4, 2.6, 1.1, "× P_clear(t+15)\n(Ineichen, Hay–Davies, IAM)")
    box(6.4, 2.6, 2.6, 1.1, "PV quantiles (kW)\n→ causal conformal")
    arrow(1.5, 8.2, 1.5, 7.3)
    arrow(2.8, 6.75, 3.4, 6.75)
    arrow(1.5, 5.0, 1.5, 6.2, INK2)
    arrow(2.8, 4.45, 3.4, 4.45)
    arrow(4.6, 5.0, 4.6, 6.2)
    arrow(5.8, 6.75, 6.4, 6.75)
    arrow(7.7, 6.2, 7.7, 5.5)
    arrow(7.7, 4.4, 7.7, 3.7)
    box(3.3, 0.5, 6.5, 1.3, "Physics-informed training objective (A2 expert only)\n"
                            "L = L_quantile + λ_phys·L_phys + λ_ramp·L_ramp + λ_bound·L_bound (+ λ_dir·L_dir)",
        color=SERIES[1], fill="#fdeee7")
    arrow(4.5, 1.8, 1.5, 6.2, SERIES[1])
    ax.text(0.2, 9.75, "SHADOW-MoE with a physics-informed objective (architecture unchanged)", fontsize=10, color=INK)
    save(fig, "F_phys1_pipeline")


def fig_crps(res, arms):
    strata = ["overall", "cloudy", "ramp", "sunrise ramp", "rising ramp", "falling ramp"]
    a0 = res["A0"]["crps"]
    others = [a for a in arms if a != "A0"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(strata))
    wd = 0.8 / max(len(others), 1)
    for j, a in enumerate(others):
        d = [100 * (res[a]["crps"][g] / a0[g] - 1) for g in strata]
        ax.bar(x + (j - (len(others) - 1) / 2) * wd, d, wd * 0.9, color=SERIES[(j + 1) % len(SERIES)], label=NAMES[a])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, strata)
    ax.set_ylabel("CRPS change vs original G3''-F (%)\n(negative = better)")
    ax.set_title("Physics-informed ablations: CRPS vs the original model (dev, out-of-fold)",
                 loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7.5, ncol=3)
    save(fig, "F_phys2_ablation_crps")


def fig_coverage(res, arms):
    strata = ["overall", "sunny", "cloudy", "ramp", "sunrise ramp"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(strata))
    wd = 0.8 / len(arms)
    for j, a in enumerate(arms):
        v = [100 * res[a]["intervals"]["90"][g]["coverage"] for g in strata]
        ax.bar(x + (j - (len(arms) - 1) / 2) * wd, v, wd * 0.9, color=SERIES[j % len(SERIES)], label=NAMES[a])
    ax.axhline(90, color=INK2, lw=1, ls="--")
    ax.text(len(strata) - 0.5, 90.5, "nominal 90%", fontsize=7.5, color=INK2, ha="right")
    ax.set_xticks(x, strata)
    ax.set_ylim(40, 100)
    ax.set_ylabel("90% interval coverage (%)")
    ax.set_title("Calibrated 90% coverage by condition (development, out-of-fold)", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7.5, ncol=3, loc="lower left")
    save(fig, "F_phys3_coverage")


def fig_ramp_magnitude(qs, z, ctx, best):
    y = z["y"].astype(float)
    pct, pcn, pvn, _ = ctx
    k_true, k_now = y / np.maximum(pct, 1), pvn / np.maximum(pcn, 1)
    ramp = z["ramp"].astype(bool)
    d_true = (k_true - k_now)[ramp]
    panels = [("A0", NAMES["A0"])] + ([(best, NAMES[best])] if best else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.3), sharey=True, squeeze=False)
    for ax, (a, title) in zip(axes[0], panels):
        d_pred = (qs[a][:, 4] / np.maximum(pct, 1) - k_now)[ramp]
        ax.hexbin(d_true, d_pred, gridsize=45, cmap="Blues", mincnt=1, bins="log", linewidths=0)
        lim = 1.0
        ax.plot([-lim, lim], [-lim, lim], color=SERIES[1], lw=1)
        ax.axhline(0, color=GREY, lw=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        slope = np.polyfit(d_true, d_pred, 1)[0]
        ax.set_title(f"{title}: slope {slope:.2f}", loc="left", fontsize=9, color=INK)
        ax.set_xlabel("observed ΔK (t → t+15)")
    axes[0][0].set_ylabel("predicted median ΔK")
    fig.suptitle("Ramp magnitude on ramp samples\n(1:1 line = perfect; slope < 1 = damped)", x=0.02, ha="left",
                 fontsize=9.5, color=INK)
    save(fig, "F_phys4_ramp_magnitude")


def fig_example(qs, z, ctx, best):
    y, day, org = z["y"].astype(float), z["day"], z["origin_us"]
    ramp, sunny = z["ramp"].astype(bool), z["sunny"].astype(bool)
    d_ramp = {d: ramp[day == d].sum() for d in np.unique(day[~sunny])}
    dsel = max(d_ramp, key=d_ramp.get)
    s = np.flatnonzero(day == dsel)
    s = s[np.argsort(org[s])]
    peak = s[np.argmax(np.convolve(ramp[s].astype(float), np.ones(30), "same"))]
    w = s[np.abs(org[s] - org[peak]) <= 90 * 60_000_000]
    t = (z["target_us"][w].astype("datetime64[us]")).astype("datetime64[m]").astype(object)
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 2.4])
    m = np.load(ROOT / "data/manifests/forecast_tol30_v2.npz")
    rows = m["hdf5_row"][m["origin"][z["sample_ids"][w[np.linspace(0, len(w) - 1, 6).astype(int)]]]]
    with h5py.File(ROOT / "data/raw/2017_2019_images_pv_processed.hdf5", "r") as f:
        imgs = [f["trainval/images_log"][int(r)] for r in rows]
    for j, im in enumerate(imgs):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(im)
        ax.set_axis_off()
    ax = fig.add_subplot(gs[1, :])
    for a, c, lab in [("A0", SERIES[0], NAMES["A0"])] + ([(best, SERIES[1], NAMES[best])] if best else []):
        ax.fill_between(t, qs[a][w, 0], qs[a][w, 8], color=c, alpha=0.18, lw=0, label=f"{lab} 5–95%")
        ax.plot(t, qs[a][w, 4], color=c, lw=1.3, label=f"{lab} median")
    ax.plot(t, y[w], color=INK, lw=1.0, label="observed PV")
    ax.set_ylabel("PV at t+15 (kW)")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    fig.suptitle(f"Difficult ramp sequence, development day {dsel}; sky frames across the window",
                 x=0.02, ha="left", fontsize=10, color=INK)
    save(fig, "F_phys5_ramp_example")


def make_figures(res, qs, z, ctx, arms):
    best = res.get("comparison_vs_A0", {}).get("best_arm")
    fig_pipeline()
    if len(arms) > 1:
        fig_crps(res, arms)
    fig_coverage(res, arms)
    fig_ramp_magnitude(qs, z, ctx, best)
    fig_example(qs, z, ctx, best)
    print("figures written to", OUT)
