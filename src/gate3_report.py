"""Gate 3 analysis (reports/phase6_prereg.md): residual-target, mixture-architecture and event-supervision gains,
raw-distribution CRPS (primary: ramp stratum), rolling-conformal coverage, event and severity diagnostics, cost.

Validation folds only; the official test set is never read. Stop conditions are asserted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.cal2 import ALPHA, conformity, interval_score, replay, select_param
from src.calibrate import base_interval
from src.clear_sky import ClearSkyPV
from src.compare import bootstrap_diff
from src.events import event_labels, future_rows, implied_event_probability
from src.folds import calibration_split
from src.gate2_report import ramp_labels, strata
from src.metrics import crps_quantile_interp
from src.skippd import SkippdForecastDataset

SEEDS, FOLDS = (42, 123, 456), range(5)
ARMS = {"E4-B-fit": Path("experiments/E4_calib/v1/csi"), "E6-QR": Path("experiments/E6/v1/E6-QR/csi"),
        "E6-U": Path("experiments/E6/v1/E6-U/csi"), "E6-S": Path("experiments/E6/v1/E6-S/csi")}
CONTRASTS = {"residual-target gain (E6-QR - E4-B-fit)": ("E6-QR", "E4-B-fit"),
             "mixture-architecture gain (E6-U - E6-QR)": ("E6-U", "E6-QR"),
             "event-supervision gain (E6-S - E6-U)": ("E6-S", "E6-U"),
             "decomposition vs residual quantiles (E6-S - E6-QR)": ("E6-S", "E6-QR")}
DELTA = 0.02
GROUPS = ("overall", "sunny", "cloudy", "ramp", "non-ramp")


def load(arm: str, seed: int, split: str) -> list[dict]:
    return [dict(np.load(ARMS[arm] / f"seed{seed}" / f"fold_{k}" / f"preds_{split}.npz")) for k in FOLDS]


def cat(parts, key):
    return np.concatenate([p[key] for p in parts])


def k_space(p):
    """K(t+15) quantiles, K(t) and dK15 truth in each run's own (fit-day) anchor."""
    sc_t = np.maximum(p["pclear_target_kw"], 1.0)
    return p["pred_q"] / sc_t[:, None], p["pv_now_kw"] / np.maximum(p["pclear_now_kw"], 1.0), p["target"] / sc_t


def training_climatology(h5, manifest, folds_doc) -> dict:
    """Arrival-class frequencies on each fold's FIT-day samples (the climatological reference forecast)."""
    ds = SkippdForecastDataset(h5, manifest, "dev", preload=False)
    m = np.load(manifest)
    g0 = m["hdf5_group"] == 0
    rec_t = m["times_us"][g0][np.argsort(m["hdf5_row"][g0])]
    rec_day = rec_t.astype("datetime64[us]").astype("datetime64[D]").astype(str)
    pv = ds.pv.numpy().astype(float)
    day = ds.day.astype(str)
    rows = future_rows(rec_t, ds.origin_time.astype("datetime64[us]").astype(np.int64))
    frozen = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    clim = {}
    for fd in folds_doc["folds"]:
        fit = calibration_split(fd["train_days"])[0]
        sel = np.isin(rec_day, fit)
        pc = ClearSkyPV().fit(rec_t[sel].astype("datetime64[us]"), pv[sel])(rec_t.astype("datetime64[us]"))
        k = pv / np.maximum(pc, 1.0)
        kf = np.where(rows >= 0, k[np.maximum(rows, 0)], np.nan)
        tau = frozen[str(fd["fold"])]["training_labels_fit_days"]
        lab = event_labels(k[ds.hist_rows[:, -1]], kf, k[ds.target_rows], tau)
        s = np.isin(day, fit) & lab["event_valid"]
        clim[fd["fold"]] = (np.bincount(lab["time_class"][s], minlength=4) / s.sum()).tolist()
    return clim


def conformal(cal_parts, val_parts):
    """E4-CAL2 rolling-window conformal per fold (W chosen on calibration days only)."""
    out = {n: ([], []) for n in ALPHA}
    for c, v in zip(cal_parts, val_parts):
        lv = v["quantile_levels"]
        cb = {n: base_interval(c["pred_q"], lv, n) for n in ALPHA}
        w, _ = select_param("rolling", c["origin_time_us"], c["target_time_us"], cb, c["target"])
        near = np.argsort(np.abs(c["origin_time_us"] - v["origin_time_us"].min()), kind="stable")[:int(w)]
        near = near[np.argsort(c["origin_time_us"][near], kind="stable")]
        for n, a in ALPHA.items():
            lo, hi = base_interval(v["pred_q"], lv, n)
            r = replay(v["origin_time_us"], v["target_time_us"], lo, hi, v["target"], a, "rolling", int(w),
                       conformity(*cb[n], c["target"])[near])
            assert (r["last_obs_us"] <= v["origin_time_us"]).all(), "STOP: future outcome used"
            assert np.isfinite(r["lo"]).all() and (r["lo"] <= r["hi"]).all(), "STOP: invalid interval"
            out[n][0].append(r["lo"])
            out[n][1].append(r["hi"])
    return {n: (np.concatenate(lo), np.concatenate(hi)) for n, (lo, hi) in out.items()}


def main():
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    h5, man = "data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz"
    ramp = ramp_labels(h5, man, folds_doc)
    clim = training_climatology(h5, man, folds_doc)
    res = {"per_seed": {}, "climatology": clim}
    for s in SEEDS:
        val = {a: load(a, s, "val") for a in ARMS}
        ids = cat(val["E4-B-fit"], "sample_ids")
        for a in ARMS:
            assert np.array_equal(cat(val[a], "sample_ids"), ids), f"STOP: {a} evaluation samples differ (seed {s})"
            q = cat(val[a], "pred_q")
            assert np.isfinite(q).all() and (np.diff(q, axis=1) >= -1e-9).all(), f"STOP: invalid quantiles {a}"
        v0 = {key: cat(val["E4-B-fit"], key) for key in ("sample_ids", "target", "clear_day", "fold", "day", "origin_time_us")}
        st = strata(v0, ramp["is_ramp"])
        sep_oct = (v0["fold"] == 1) & np.isin(v0["origin_time_us"].astype("datetime64[us]").astype("datetime64[M]").astype(str),
                                              ["2017-09", "2017-10"])
        st["fold 1 Sep-Oct 2017"] = sep_oct
        y = v0["target"]
        dayk = np.char.add(v0["fold"].astype(str), np.char.add("_", v0["day"].astype(str)))
        crps = {a: crps_quantile_interp(cat(val[a], "pred_q"), y, val[a][0]["quantile_levels"]) for a in ARMS}
        raw = {a: {g: float(crps[a][mk].mean()) for g, mk in st.items()} for a in ARMS}
        per_fold_ramp = {a: [float(crps[a][st["ramp"] & (v0["fold"] == k)].mean()) for k in FOLDS] for a in ARMS}
        contrasts = {name: {g: bootstrap_diff(crps[x][st[g]], crps[b][st[g]], dayk[st[g]], v0["clear_day"][st[g]], kind="mean")
                            for g in GROUPS} for name, (x, b) in CONTRASTS.items()}
        contrasts["E6-S - E6-QR (gate)"] = contrasts["decomposition vs residual quantiles (E6-S - E6-QR)"]
        # calibrated intervals (rolling conformal) and raw intervals
        cov = {}
        for a in ARMS:
            ci = conformal(load(a, s, "cal"), val[a])
            rq = cat(val[a], "pred_q")
            cov[a] = {}
            for kind, bnds in (("raw", {n: base_interval(rq, val[a][0]["quantile_levels"], n) for n in ALPHA}), ("rolling conformal", ci)):
                cov[a][kind] = {g: {n: {"coverage": float(np.mean((y[mk] >= bnds[n][0][mk]) & (y[mk] <= bnds[n][1][mk]))),
                                        "width_kw": float(np.mean(bnds[n][1][mk] - bnds[n][0][mk])),
                                        "interval_score": float(interval_score(bnds[n][0][mk], bnds[n][1][mk], y[mk], ALPHA[n]).mean())}
                                    for n in ALPHA} for g, mk in st.items()}
        # event occurrence comparator: P(|dK15| >= tau_eval), quantile-implied, every arm in its own K space
        tau_eval = np.concatenate([np.full(len(p["target"]), ramp["tau_ramp"][k]) for k, p in zip(FOLDS, val["E4-B-fit"])])
        occ = st["ramp"].astype(float)
        event_occ = {}
        for a in ARMS:
            parts = []
            for k, p in zip(FOLDS, val[a]):
                qk, kn, _ = k_space(p)
                parts.append(implied_event_probability(qk, p["quantile_levels"], kn, ramp["tau_ramp"][k]))
            pr = np.concatenate(parts)
            bins = np.clip((pr * 10).astype(int), 0, 9)
            event_occ[a] = {"brier": float(np.mean((pr - occ) ** 2)),
                            "reliability": {f"{b / 10:.1f}-{(b + 1) / 10:.1f}": [float(pr[bins == b].mean()), float(occ[bins == b].mean()), int((bins == b).sum())]
                                            for b in range(10) if (bins == b).any()}}
        event_occ["climatology_brier"] = float(np.mean((occ.mean() - occ) ** 2))
        # arrival-class quality (E6-U, E6-S) vs fit-day climatology
        arrival = {}
        for a in ("E6-U", "E6-S"):
            tp, tc, ok = cat(val[a], "time_probs"), cat(val[a], "time_class"), cat(val[a], "event_valid").astype(bool)
            fold_ids = cat(val[a], "fold")
            cl = np.array([clim[int(f)] for f in fold_ids])
            oh = np.eye(4)[tc]
            ev = ok & (tc < 3)
            arrival[a] = {"brier": float(((tp - oh) ** 2).sum(1)[ok].mean()),
                          "brier_climatology": float(((cl - oh) ** 2).sum(1)[ok].mean()),
                          "logloss": float(-np.log(np.clip(tp[ok, tc[ok]], 1e-12, 1)).mean()),
                          "logloss_climatology": float(-np.log(np.clip(cl[ok, tc[ok]], 1e-12, 1)).mean()),
                          "bin_accuracy_true_events": float((tp[ev, :3].argmax(1) == tc[ev]).mean()),
                          "mean_p_event_on_true_events": float(1 - tp[ev, 3].mean()),
                          "mean_p_event_on_non_events": float(1 - tp[ok & (tc == 3), 3].mean())}
        # severity diagnostics on true ramp samples (each arm's own K space)
        sev = {}
        r_mask = st["ramp"]
        for a in ARMS:
            qk = np.concatenate([k_space(p)[0] for p in val[a]])
            kn = np.concatenate([k_space(p)[1] for p in val[a]])
            kt = np.concatenate([k_space(p)[2] for p in val[a]])
            dq, dk = qk - kn[:, None], kt - kn
            lv = val[a][0]["quantile_levels"]
            lo80, hi80 = base_interval(dq, lv, "80")
            sev[a] = {"median_dK_mae": float(np.abs(dq[r_mask, 4] - dk[r_mask]).mean()),
                      "sign_accuracy": float((np.sign(dq[r_mask, 4]) == np.sign(dk[r_mask])).mean()),
                      "dK_80pct_coverage": float(((dk >= lo80) & (dk <= hi80))[r_mask].mean())}
        gate = {"ramp_S_minus_QR": contrasts["E6-S - E6-QR (gate)"]["ramp"],
                "ramp_S_minus_U": contrasts["event-supervision gain (E6-S - E6-U)"]["ramp"],
                "overall_S_minus_QR": raw["E6-S"]["overall"] - raw["E6-QR"]["overall"]}
        gate["pass"] = bool(gate["ramp_S_minus_QR"][2] < 0 and gate["ramp_S_minus_U"][2] < 0 and gate["overall_S_minus_QR"] <= DELTA)
        res["per_seed"][s] = {"raw_crps": raw, "per_fold_ramp_crps": per_fold_ramp, "contrasts": contrasts,
                              "intervals": cov, "event_occurrence": event_occ, "arrival": arrival, "severity": sev,
                              "gate3": gate, "n": int(len(y))}
    seed_mean_fold = {a: np.mean([res["per_seed"][s]["per_fold_ramp_crps"][a] for s in SEEDS], axis=0).tolist() for a in ARMS}
    folds_won = int(sum(seed_mean_fold["E6-S"][k] < seed_mean_fold["E6-QR"][k] for k in FOLDS))
    n_pass = sum(res["per_seed"][s]["gate3"]["pass"] for s in SEEDS)
    res["gate3"] = {"seeds_passing": int(n_pass), "folds_won_E6S_vs_QR_seed_mean": folds_won,
                    "retain_decomposition": bool(n_pass >= 2 and folds_won >= 3), "delta_kw": DELTA,
                    "seed_mean_ramp_crps_per_fold": seed_mean_fold}
    Path("reports/gate3_results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["gate3"], indent=1))


if __name__ == "__main__":
    main()
