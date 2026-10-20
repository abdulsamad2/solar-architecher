"""Gate 2 report: E4 replication (Part A), calibration (Part B), computational cost.

Protocol: reports/gate2_calibration_prereg.md. Validation folds only; the official test set is never read.
Stop conditions are asserted: identical evaluation samples across compared models, disjoint calibration and
validation samples, no crossing recalibrated quantiles (raised in src.calibrate).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from src.calibrate import INTERVALS, base_interval, cqr_apply, cqr_fit, recal_apply, recal_fit
from src.clear_sky import ClearSkyPV
from src.compare import bootstrap_diff
from src.metrics import crps_quantile_interp, quantile_reliability
from src.skippd import SkippdForecastDataset

SEEDS = (42, 123, 456)
FOLDS = range(5)
NOMINAL = {"50": 0.5, "80": 0.8, "90": 0.9}
E4 = Path("experiments/E4_quantile/v1")
CAL = Path("experiments/E4_calib/v1/csi")


def load_val(run: Path) -> dict:
    parts = [dict(np.load(run / f"fold_{k}" / "preds_val.npz")) for k in FOLDS]
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0] if key != "quantile_levels"} | {
        "levels": parts[0]["quantile_levels"]}


# ---------- strata: ramp labels from the fold's full-training-day anchor (pre-registered) ----------
def ramp_labels(h5, manifest, folds_doc) -> dict:
    ds = SkippdForecastDataset(h5, manifest, "dev", preload=False)
    m = np.load(manifest)
    g0 = m["hdf5_group"] == 0
    rec_t = m["times_us"][g0][np.argsort(m["hdf5_row"][g0])].astype("datetime64[us]")
    rec_day = rec_t.astype("datetime64[D]").astype(str)
    pv = ds.pv.numpy().astype(float)
    day = ds.day.astype(str)
    out, thresholds = {}, {}
    for fd in folds_doc["folds"]:
        k = fd["fold"]
        pc = ClearSkyPV().fit(rec_t[np.isin(rec_day, fd["train_days"])], pv[np.isin(rec_day, fd["train_days"])])(rec_t)
        k_rec = pv / np.maximum(pc, 1.0)
        dk = np.abs(k_rec[ds.target_rows] - k_rec[ds.hist_rows[:, -1]])
        tau = float(np.quantile(dk[np.isin(day, fd["train_days"])], 0.9))
        thresholds[k] = tau
        va = np.isin(day, fd["val_days"])
        out.update(dict(zip(ds.sample_ids[va], dk[va] >= tau)))
    return {"is_ramp": out, "tau_ramp": thresholds}


def strata(v: dict, ramp: dict) -> dict:
    is_ramp = np.array([ramp[s] for s in v["sample_ids"]])
    st = {"overall": np.ones(len(v["target"]), bool), "sunny": v["clear_day"], "cloudy": ~v["clear_day"],
          "ramp": is_ramp, "non-ramp": ~is_ramp}
    st.update({f"fold {k}": v["fold"] == k for k in FOLDS})
    return st


def interval_metrics(bounds: dict, y, st) -> dict:
    return {g: {n: {"coverage": float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))), "width_kw": float(np.mean(hi[m] - lo[m]))}
                for n, (lo, hi) in bounds.items()} for g, m in st.items()}


def quantile_metrics(q, y, levels, st) -> dict:
    crps = crps_quantile_interp(q, y, levels)
    bounds = {n: base_interval(q, levels, n) for n in INTERVALS}
    out = interval_metrics(bounds, y, st)
    for g, m in st.items():
        out[g]["crps_kw"] = float(crps[m].mean())
        out[g]["reliability"] = quantile_reliability(q[m], y[m]).round(4).tolist()
    return out, crps


# ---------- Part A ----------
def part_a(ramp) -> dict:
    res = {"per_seed": {}}
    for s in SEEDS:
        a, b = load_val(E4 / "kw" / f"seed{s}"), load_val(E4 / "csi" / f"seed{s}")
        assert np.array_equal(a["sample_ids"], b["sample_ids"]), f"STOP: E4-A/E4-B samples differ (seed {s})"
        st = strata(a, ramp)
        y, levels = a["target"], a["levels"]
        ma, ca = quantile_metrics(a["pred_q"], y, levels, st)
        mb, cb = quantile_metrics(b["pred_q"], y, levels, st)
        day = np.char.add(a["fold"].astype(str), np.char.add("_", a["day"].astype(str)))
        diff = {g: bootstrap_diff(cb[st[g]], ca[st[g]], day[st[g]], a["clear_day"][st[g]], kind="mean")
                for g in ("overall", "sunny", "cloudy", "ramp")}
        rmse = {n: {g: float(np.sqrt(np.mean((v["pred"][st[g]] - y[st[g]]) ** 2))) for g in ("overall", "sunny", "cloudy")}
                for n, v in (("E4-A", a), ("E4-B", b))}
        pers = {g: float(np.abs(a["persistence"][st[g]] - y[st[g]]).mean()) for g in st}
        res["per_seed"][s] = {"E4-A": ma, "E4-B": mb, "crps_diff_B_minus_A": diff, "median_rmse": rmse,
                              "persistence_crps": pers, "n": int(len(y))}
    res["across_seeds"] = {
        f"{n} {g} CRPS": _stats([res["per_seed"][s][n][g]["crps_kw"] for s in SEEDS])
        for n in ("E4-A", "E4-B") for g in ("overall", "sunny", "cloudy", "ramp")}
    res["across_seeds"].update({
        f"E4-B - E4-A {g} CRPS": _stats([res["per_seed"][s]["crps_diff_B_minus_A"][g][0] for s in SEEDS])
        for g in ("overall", "sunny", "cloudy", "ramp")})
    res["across_seeds"].update({
        f"{n} 90% coverage": _stats([res["per_seed"][s][n]["overall"]["90"]["coverage"] for s in SEEDS])
        for n in ("E4-A", "E4-B")})
    return res


# ---------- Part B ----------
def part_b(ramp) -> dict:
    res = {"per_seed": {}, "fit_seconds": []}
    for s in SEEDS:
        run = CAL / f"seed{s}"
        full = load_val(E4 / "csi" / f"seed{s}")
        fold_val, fold_cal = [], []
        for k in FOLDS:
            v = dict(np.load(run / f"fold_{k}" / "preds_val.npz"))
            c = dict(np.load(run / f"fold_{k}" / "preds_cal.npz"))
            assert not set(c["sample_ids"]) & set(v["sample_ids"]), "STOP: calibration and validation samples overlap"
            split = json.loads((run / f"fold_{k}" / "calibration_split.json").read_text())
            assert not set(split["fit_days"]) & set(split["cal_days"]), "STOP: fit and calibration days overlap"
            assert not set(c["day"]) - set(split["cal_days"]), "STOP: calibration sample outside calibration days"
            levels = v["quantile_levels"]
            t0 = time.perf_counter()
            cqr_kw = {n: cqr_fit(*base_interval(c["pred_q"], levels, n), c["target"], NOMINAL[n]) for n in INTERVALS}
            sc_c = np.maximum(c["pclear_target_kw"], 1.0)
            cqr_k = {n: cqr_fit(*base_interval(c["pred_q"], levels, n), c["target"], NOMINAL[n], scale=sc_c) for n in INTERVALS}
            rc = recal_fit(c["pred_q"], c["target"], levels)
            q_recal = recal_apply(v["pred_q"], levels, rc)  # raises on crossing (STOP)
            res["fit_seconds"].append(time.perf_counter() - t0)
            sc_v = np.maximum(v["pclear_target_kw"], 1.0)
            v["cqr_kw"] = {n: cqr_apply(*base_interval(v["pred_q"], levels, n), cqr_kw[n]) for n in INTERVALS}
            v["cqr_k"] = {n: cqr_apply(*base_interval(v["pred_q"], levels, n), cqr_k[n], scale=sc_v) for n in INTERVALS}
            v["q_recal"] = q_recal
            v["meta"] = {"q_hat_kw": cqr_kw, "q_hat_k": cqr_k, "recal_mapped_levels": rc["mapped_levels"].tolist(),
                         "recal_saturated_levels": rc["saturated_levels"], "n_cal": int(len(c["target"]))}
            fold_val.append(v)
        cat = lambda key: np.concatenate([f[key] for f in fold_val])
        v = {key: cat(key) for key in ("sample_ids", "target", "pred_q", "clear_day", "fold", "q_recal", "day")}
        assert np.array_equal(v["sample_ids"], full["sample_ids"]), "STOP: calibrated vs E4-B evaluation samples differ"
        st = strata(v, ramp)
        y, levels = v["target"], fold_val[0]["quantile_levels"]
        bounds = {m: {n: (np.concatenate([f[m][n][0] for f in fold_val]), np.concatenate([f[m][n][1] for f in fold_val]))
                      for n in INTERVALS} for m in ("cqr_kw", "cqr_k")}
        out = {
            "E4-B (full train)": quantile_metrics(full["pred_q"], y, levels, st)[0],
            "E4-B-fit uncalibrated": quantile_metrics(v["pred_q"], y, levels, st)[0],
            "E4-B-fit + CQR-kW": interval_metrics(bounds["cqr_kw"], y, st),
            "E4-B-fit + CQR-K": interval_metrics(bounds["cqr_k"], y, st),
            "E4-B-fit + recalibration": quantile_metrics(v["q_recal"], y, levels, st)[0],
        }
        out["inverted_intervals"] = {m: {n: int(np.sum(bounds[m][n][0] > bounds[m][n][1])) for n in INTERVALS}
                                     for m in bounds}
        out["per_fold_meta"] = [f["meta"] for f in fold_val]
        res["per_seed"][s] = out
    res["closing_rule"] = closing_rule(res)
    return res


def closing_rule(res) -> dict:
    ok = []
    detail = {}
    for s in SEEDS:
        r = res["per_seed"][s]["E4-B-fit + CQR-kW"]
        cov = {n: r["overall"][n]["coverage"] for n in INTERVALS}
        fold90 = {k: r[f"fold {k}"]["90"]["coverage"] for k in FOLDS}
        cond = (abs(cov["80"] - 0.8) <= 0.03 and abs(cov["90"] - 0.9) <= 0.03 and abs(cov["50"] - 0.5) <= 0.05
                and min(fold90.values()) >= 0.80)
        ok.append(cond)
        detail[s] = {"pooled_coverage": cov, "min_fold_90": min(fold90.values()), "pass": bool(cond)}
    return {"per_seed": detail, "seeds_passing": int(sum(ok)), "gate2_closed": bool(sum(ok) >= 2)}


# ---------- cost ----------
def cost() -> dict:
    out = {}
    for name, root in (("E4-A", E4 / "kw"), ("E4-B", E4 / "csi"), ("E4-B-fit", CAL)):
        secs, epochs = [], []
        for s in SEEDS:
            log = (root / f"seed{s}" / "train.log").read_text(encoding="utf-8")
            ep = [float(x) for x in re.findall(r"epoch\s+\d+ .*\((\d+)s\)", log)]
            secs.append(sum(ep))
            epochs.append(len(ep))
        out[name] = {"train_wall_s_per_run": _stats(secs), "epochs_per_run": _stats(epochs)}
    return out


def _stats(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "values": v.round(4).tolist()}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="data/raw/2017_2019_images_pv_processed.hdf5")
    p.add_argument("--manifest", default="data/manifests/forecast_tol30_v2.npz")
    p.add_argument("--folds", default="data/manifests/folds_blocked5_v1.json")
    p.add_argument("--out", default="reports/gate2_results.json")
    a = p.parse_args()
    r = ramp_labels(a.h5, a.manifest, json.loads(Path(a.folds).read_text()))
    result = {"tau_ramp": r["tau_ramp"], "part_a": part_a(r["is_ramp"]), "part_b": part_b(r["is_ramp"]), "cost": cost()}
    Path(a.out).write_text(json.dumps(result, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(json.dumps(result["part_b"]["closing_rule"], indent=1))
