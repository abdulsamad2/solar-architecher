"""Run E4-CAL2 (reports/e4_cal2_prereg.md) and write reports/e4_cal2_results.json + fold-1 weekly figure."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from src.cal2 import ALPHA, conformal_q, conformity, interval_score, replay, select_param, widen
from src.calibrate import base_interval
from src.compare import bootstrap_diff
from src.gate2_report import ramp_labels, strata

SEEDS, FOLDS = (42, 123, 456), range(5)
CAL = Path("experiments/E4_calib/v1/csi")
FULL = Path("experiments/E4_quantile/v1/csi")
OUT = Path("experiments/E4_cal2/v1")
METHODS = ("uncalibrated", "fixed CQR-kW", "ACI", "rolling conformal")


def bounds_of(d):
    return {n: base_interval(d["pred_q"], d["quantile_levels"], n) for n in ALPHA}


def run_fold(seed: int, k: int) -> dict:
    run = CAL / f"seed{seed}" / f"fold_{k}"
    out = OUT / f"seed{seed}" / f"fold_{k}"
    out.mkdir(parents=True, exist_ok=True)
    c = dict(np.load(run / "preds_cal.npz"))
    cb = bounds_of(c)

    # ---- 1) hyperparameters from CALIBRATION data only, written before validation is touched ----
    sel = {m: select_param(m, c["origin_time_us"], c["target_time_us"], cb, c["target"]) for m in ("aci", "rolling")}
    params = {"aci_gamma": sel["aci"][0], "rolling_W": int(sel["rolling"][0]),
              "selection_scores": {m: {str(p): s for p, s in sel[m][1].items()} for m in sel},
              "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    (out / "params.json").write_text(json.dumps(params, indent=1))

    # ---- 2) validation replay ----
    v = dict(np.load(run / "preds_val.npz"))
    assert not set(c["sample_ids"]) & set(v["sample_ids"]), "STOP: calibration/validation overlap"
    vb = bounds_of(v)
    y = v["target"]
    val_start = v["origin_time_us"].min()
    near = np.argsort(np.abs(c["origin_time_us"] - val_start), kind="stable")[:params["rolling_W"]]
    near = near[np.argsort(c["origin_time_us"][near], kind="stable")]
    res, trace = {m: {} for m in METHODS}, {}
    for n, a in ALPHA.items():
        cal_scores = conformity(*cb[n], c["target"])
        lo, hi = vb[n]
        res["uncalibrated"][n] = (lo, hi, np.zeros(len(y), bool))
        q = conformal_q(np.sort(cal_scores), 1 - a)
        fx = [widen(lo[i], hi[i], q) for i in range(len(y))]
        res["fixed CQR-kW"][n] = tuple(np.array(z) for z in zip(*fx))
        r_aci = replay(v["origin_time_us"], v["target_time_us"], lo, hi, y, a, "aci", params["aci_gamma"], cal_scores)
        r_rol = replay(v["origin_time_us"], v["target_time_us"], lo, hi, y, a, "rolling", params["rolling_W"],
                       cal_scores[near])
        res["ACI"][n] = (r_aci["lo"], r_aci["hi"], r_aci["degenerate"])
        res["rolling conformal"][n] = (r_rol["lo"], r_rol["hi"], r_rol["degenerate"])
        for m, r in (("aci", r_aci), ("rolling", r_rol)):
            assert (r["last_obs_us"] <= v["origin_time_us"]).all(), "STOP: future outcome used"
            trace.update({f"{m}_{n}_{key}": r[key] for key in ("q", "alpha_t", "n_updates", "last_obs_us", "miss")})
    for m in METHODS:
        for n in ALPHA:
            lo_, hi_, _ = res[m][n]
            assert np.isfinite(lo_).all() and np.isfinite(hi_).all() and (lo_ <= hi_).all(), f"STOP: invalid {m} {n}"
    np.savez_compressed(out / "intervals.npz", sample_ids=v["sample_ids"], origin_time_us=v["origin_time_us"],
                        target_time_us=v["target_time_us"], target=y,
                        **{f"{m}|{n}|{e}": res[m][n][j] for m in METHODS for n in ALPHA for j, e in enumerate(("lo", "hi", "degenerate"))})
    np.savez_compressed(out / "trace.npz", origin_time_us=v["origin_time_us"], target_time_us=v["target_time_us"], **trace)
    return {"v": v, "res": res, "params": params}


def main():
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    ramp = ramp_labels("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz", folds_doc)
    results = {"per_seed": {}, "params": {}}
    weekly = {m: [] for m in METHODS}
    for s in SEEDS:
        folds = [run_fold(s, k) for k in FOLDS]
        results["params"][s] = {k: f["params"] for k, f in zip(FOLDS, folds)}
        cat = lambda key: np.concatenate([f["v"][key] for f in folds])
        v = {key: cat(key) for key in ("sample_ids", "target", "clear_day", "fold", "day", "origin_time_us")}
        ref = np.concatenate([np.load(FULL / f"seed{s}" / f"fold_{k}" / "preds_val.npz")["sample_ids"] for k in FOLDS])
        assert np.array_equal(v["sample_ids"], ref), "STOP: evaluation samples differ from Gate 2"
        st = strata(v, ramp["is_ramp"])
        y = v["target"]
        iv = {m: {n: tuple(np.concatenate([f["res"][m][n][j] for f in folds]) for j in range(3)) for n in ALPHA} for m in METHODS}
        met = {m: {g: {n: {"coverage": float(np.mean((y[mk] >= iv[m][n][0][mk]) & (y[mk] <= iv[m][n][1][mk]))),
                           "width_kw": float(np.mean(iv[m][n][1][mk] - iv[m][n][0][mk])),
                           "interval_score": float(interval_score(iv[m][n][0][mk], iv[m][n][1][mk], y[mk], ALPHA[n]).mean()),
                           "degenerate": int(iv[m][n][2][mk].sum())} for n in ALPHA} for g, mk in st.items()} for m in METHODS}
        dayk = np.char.add(v["fold"].astype(str), np.char.add("_", v["day"].astype(str)))
        diffs = {}
        for m in ("ACI", "rolling conformal"):
            diffs[m] = {}
            for n in ALPHA:
                is_m = interval_score(*iv[m][n][:2], y, ALPHA[n])
                is_f = interval_score(*iv["fixed CQR-kW"][n][:2], y, ALPHA[n])
                miss_m = ((y < iv[m][n][0]) | (y > iv[m][n][1])).astype(float)
                miss_f = ((y < iv["fixed CQR-kW"][n][0]) | (y > iv["fixed CQR-kW"][n][1])).astype(float)
                diffs[m][n] = {g: {"interval_score": bootstrap_diff(is_m[mk], is_f[mk], dayk[mk], v["clear_day"][mk], kind="mean"),
                                   "coverage": tuple(-x for x in bootstrap_diff(miss_m[mk], miss_f[mk], dayk[mk], v["clear_day"][mk], kind="mean"))}
                               for g, mk in (("overall", st["overall"]), ("fold 1", st["fold 1"]), ("ramp", st["ramp"]))}
        # time-resolved 90% coverage: per calendar month per fold, and per 7-day window in fold 1
        month = v["origin_time_us"].astype("datetime64[us]").astype("datetime64[M]").astype(str)
        monthly = {m: {f"fold {k} {mo}": float(np.mean(((y >= iv[m]["90"][0]) & (y <= iv[m]["90"][1]))[(v["fold"] == k) & (month == mo)]))
                       for k in FOLDS for mo in np.unique(month[v["fold"] == k])} for m in METHODS}
        f1 = v["fold"] == 1
        d1 = v["origin_time_us"][f1].astype("datetime64[us]").astype("datetime64[D]")
        week = ((d1 - d1.min()).astype(int) // 7)
        for m in METHODS:
            hit = ((y >= iv[m]["90"][0]) & (y <= iv[m]["90"][1]))[f1]
            weekly[m].append([float(hit[week == w].mean()) if (week == w).any() else np.nan for w in range(week.max() + 1)])
        results["per_seed"][s] = {"metrics": met, "paired_vs_fixed": diffs, "monthly_cov90": monthly}
    results["decision"] = decide(results)
    Path("reports/e4_cal2_results.json").write_text(json.dumps(results, indent=1, default=float))
    plot_weekly(weekly)
    print(json.dumps(results["decision"], indent=1))


def decide(results) -> dict:
    out = {}
    for m in ("ACI", "rolling conformal"):
        ok = []
        for s in SEEDS:
            d = results["per_seed"][s]["paired_vs_fixed"][m]["90"]["overall"]["interval_score"]
            f1m = results["per_seed"][s]["metrics"][m]["fold 1"]["90"]["coverage"]
            f1f = results["per_seed"][s]["metrics"]["fixed CQR-kW"]["fold 1"]["90"]["coverage"]
            ok.append(bool(d[2] < 0 and f1m > f1f))
        out[m] = {"seeds_qualifying": int(sum(ok)), "qualifies": sum(ok) >= 2, "per_seed": ok,
                  "seed_mean_pooled_IS90": float(np.mean([results["per_seed"][s]["metrics"][m]["overall"]["90"]["interval_score"] for s in SEEDS]))}
    q = [m for m in out if out[m]["qualifies"]]
    out["recommended"] = min(q, key=lambda m: out[m]["seed_mean_pooled_IS90"]) if q else None
    return out


def plot_weekly(weekly):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for m, series in weekly.items():
        n = min(len(x) for x in series)
        arr = np.array([x[:n] for x in series])
        ax.plot(np.arange(n), np.nanmean(arr, 0), marker="o", ms=3, lw=1.2, label=m)
    ax.axhline(0.9, color="0.5", ls="--", lw=1)
    ax.set_xlabel("week since start of fold-1 validation (Sep 2017)")
    ax.set_ylabel("90% interval coverage (seed mean)")
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig("reports/e4_cal2_fold1_weekly.png", dpi=110)


if __name__ == "__main__":
    main()
