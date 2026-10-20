"""E8 Track B (reports/e8_prereg.md section 4): motion-augmented gate G5 vs G3/G4, frozen experts, CPU only.

Common population only (calibration and validation). Gates refitted with the E7b procedure on calibration days.
Also reported: the frozen E7b G3/G4 (fitted on the full population) evaluated on the same common samples.
Discrimination (gate weight / single features as ramp scores: AUPRC, Brier) is reported separately from CRPS.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.cal2 import ALPHA, interval_score
from src.calibrate import base_interval
from src.compare import bootstrap_diff
from src.e8_features import FEAT, verify_frozen
from src.gate2_report import ramp_labels, strata
from src.gate3_report import conformal
from src.gate7b import apply_gate, average_precision, raw_features, select_and_fit
from src.metrics import crps_quantile_interp

SEEDS, FOLDS = (42, 123, 456), range(5)
DIRECT, RESID = Path("experiments/E4_calib/v1/csi"), Path("experiments/E6/v1/E6-S/csi")
E7B = Path("experiments/E7b_gate/v1")
OUT = Path("experiments/E8_trackB/v1")
GATES = {"G3c": ("g", "d"), "G4c": ("g", "s_D", "s_R", "d"), "G5": ("g", "d", "inflow", "ring0", "shiftmag")}
METHODS = ("E4-B-fit", "E6-S", "G3 (E7b frozen)", "G4 (E7b frozen)") + tuple(GATES)


def motion_at_origin(sample_ids, sid_to_dev, origin_row, M):
    rows = origin_row[[sid_to_dev[s] for s in sample_ids]]
    m = M[rows]
    return {"inflow": m[:, 5].astype(float), "ring0": m[:, 0].astype(float), "shiftmag": np.hypot(m[:, 3], m[:, 4]).astype(float)}


def main():
    verify_frozen()
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    frozen = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    ramp = ramp_labels("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz", folds_doc)
    win = np.load(FEAT / "e8_windows_v1.npz")
    M = np.load(FEAT / "e8_record_features_v1.npz")["M"]
    sid_to_dev = {s: i for i, s in enumerate(win["sample_ids"])}
    common_ids = set(win["sample_ids"][win["eligible_common"]])
    origin_row = win["H15"][:, 0]
    res = {"per_seed": {}, "params": {}, "cost": {"fit_s": []}}
    for s in SEEDS:
        vals, cals, meta, w_all, mot_all = {m: [] for m in METHODS}, {m: [] for m in METHODS}, [], {g: [] for g in GATES}, []
        res["params"][s] = {}
        for k in FOLDS:
            tau = frozen[str(k)]["training_labels_fit_days"]
            sp = {}
            for split in ("cal", "val"):
                dd = dict(np.load(DIRECT / f"seed{s}" / f"fold_{k}" / f"preds_{split}.npz"))
                rr = dict(np.load(RESID / f"seed{s}" / f"fold_{k}" / f"preds_{split}.npz"))
                assert np.array_equal(dd["sample_ids"], rr["sample_ids"]), "STOP: expert samples differ"
                keep = np.array([i in common_ids for i in dd["sample_ids"]])
                dd = {key: (v[keep] if np.ndim(v) and len(v) == len(keep) else v) for key, v in dd.items()}
                rr = {key: (v[keep] if np.ndim(v) and len(v) == len(keep) else v) for key, v in rr.items()}
                feats = raw_features(dd, rr, tau)
                feats.update(motion_at_origin(dd["sample_ids"], sid_to_dev, origin_row, M))
                sp[split] = (dd, rr, feats, keep)
            (dc, rc, fc, _), (dv, rv, fv, keep_v) = sp["cal"], sp["val"]
            lv, yc = dc["quantile_levels"], dc["target"]
            order = np.argsort(dc["origin_time_us"], kind="stable")
            e7b = np.load(E7B / f"seed{s}" / f"fold_{k}" / "preds_val.npz")
            assert np.array_equal(e7b["sample_ids"][keep_v], dv["sample_ids"]), "STOP: E7b samples differ"
            blends = {"E4-B-fit": (dv["pred_q"], dc["pred_q"]), "E6-S": (rv["pred_q"], rc["pred_q"])}
            # frozen E7b gates: validation predictions as stored (subset); calibration counterpart rebuilt from frozen params
            e7p = json.loads((E7B / f"seed{s}" / f"fold_{k}" / "params.json").read_text())
            for name, key in (("G3 (E7b frozen)", "G3"), ("G4 (E7b frozen)", "G4")):
                wc = apply_gate(e7p[key], fc)
                blends[name] = (e7b[f"q_{key}"][keep_v], wc[:, None] * dc["pred_q"] + (1 - wc[:, None]) * rc["pred_q"])
            res["params"][s][k] = {}
            for name, feats in GATES.items():
                t0 = time.perf_counter()
                p = select_and_fit(fc, feats, dc["pred_q"], rc["pred_q"], yc, lv, order)  # calibration days only
                res["cost"]["fit_s"].append(time.perf_counter() - t0)
                wv, wc = apply_gate(p, fv), apply_gate(p, fc)
                assert np.isfinite(wv).all() and np.isfinite(wc).all(), "STOP: non-finite gate weights"
                qv = wv[:, None] * dv["pred_q"] + (1 - wv[:, None]) * rv["pred_q"]
                qc = wc[:, None] * dc["pred_q"] + (1 - wc[:, None]) * rc["pred_q"]
                assert np.isfinite(qv).all() and (np.diff(qv, axis=1) >= -1e-9).all(), "STOP: invalid blended quantiles"
                blends[name] = (qv, qc)
                w_all[name].append(wv)
                res["params"][s][k][name] = p
            (OUT / f"seed{s}" / f"fold_{k}").mkdir(parents=True, exist_ok=True)
            (OUT / f"seed{s}" / f"fold_{k}" / "params.json").write_text(json.dumps(res["params"][s][k], indent=1))
            np.savez_compressed(OUT / f"seed{s}" / f"fold_{k}" / "preds_val.npz", sample_ids=dv["sample_ids"],
                                **{f"w_{g}": w_all[g][-1] for g in GATES}, **{f"q_{m}": blends[m][0] for m in METHODS})
            for m in METHODS:
                vals[m].append({**{x: dv[x] for x in ("origin_time_us", "target_time_us", "target", "quantile_levels")}, "pred_q": blends[m][0]})
                cals[m].append({**{x: dc[x] for x in ("origin_time_us", "target_time_us", "target", "quantile_levels")}, "pred_q": blends[m][1]})
            mot_all.append({key: fv[key] for key in ("g", "d", "inflow", "ring0", "shiftmag")})
            meta.append(dv)
        cat = lambda key: np.concatenate([v[key] for v in meta])
        v0 = {key: cat(key) for key in ("sample_ids", "target", "clear_day", "fold", "day", "origin_time_us")}
        assert set(v0["sample_ids"]) <= common_ids
        st = strata(v0, ramp["is_ramp"])
        y, lv = v0["target"], meta[0]["quantile_levels"]
        dayk = np.char.add(v0["fold"].astype(str), np.char.add("_", v0["day"].astype(str)))
        q = {m: np.concatenate([v["pred_q"] for v in vals[m]]) for m in METHODS}
        crps = {m: crps_quantile_interp(q[m], y, lv) for m in METHODS}
        raw = {m: {g: float(crps[m][mk].mean()) for g, mk in st.items()} for m in METHODS}
        pairs = {"G5 - G3c": ("G5", "G3c"), "G5 - G4c": ("G5", "G4c"), "G4c - G3c": ("G4c", "G3c"),
                 "G5 - G3 (E7b frozen)": ("G5", "G3 (E7b frozen)"), "G5 - G4 (E7b frozen)": ("G5", "G4 (E7b frozen)")}
        diffs = {n: {g: bootstrap_diff(crps[a][st[g]], crps[b][st[g]], dayk[st[g]], v0["clear_day"][st[g]], kind="mean")
                     for g in ("overall", "sunny", "cloudy", "ramp", "non-ramp")} for n, (a, b) in pairs.items()}
        lab = st["ramp"]
        disc = {"ramp_base_rate": float(lab.mean())}
        for gname in GATES:
            w = np.concatenate(w_all[gname])
            disc[gname] = {"auprc": average_precision(w, lab), "brier": float(np.mean((w - lab) ** 2)),
                           "w_ramp_mean": float(w[lab].mean()), "w_nonramp_mean": float(w[~lab].mean()),
                           "w_ramp_p90": float(np.percentile(w[lab], 90)), "w_nonramp_p90": float(np.percentile(w[~lab], 90))}
        for f in ("g", "d", "inflow", "ring0", "shiftmag"):
            x = np.concatenate([mm[f] for mm in mot_all])
            disc[f"feature_{f}_auprc"] = max(average_precision(x, lab), average_precision(-x, lab))
        intervals = {}
        for m in METHODS:
            ci = conformal(cals[m], vals[m])
            intervals[m] = {kind: {g: {n: {"coverage": float(np.mean((y[mk] >= b_[n][0][mk]) & (y[mk] <= b_[n][1][mk]))),
                                           "width_kw": float(np.mean(b_[n][1][mk] - b_[n][0][mk])),
                                           "interval_score": float(interval_score(b_[n][0][mk], b_[n][1][mk], y[mk], ALPHA[n]).mean())}
                                       for n in ALPHA} for g, mk in st.items() if g in ("overall", "ramp", "non-ramp")}
                            for kind, b_ in (("raw", {n: base_interval(q[m], lv, n) for n in ALPHA}), ("rolling conformal", ci))}
        rule = {"ramp_ci_below_0": bool(diffs["G5 - G3c"]["ramp"][2] < 0),
                "overall_within_0.005": bool(raw["G5"]["overall"] - raw["G3c"]["overall"] <= 0.005)}
        res["per_seed"][s] = {"raw_crps": raw, "contrasts": diffs, "discrimination": disc, "intervals": intervals,
                              "rule_G5_vs_G3c": rule, "n_val": int(len(y))}
    n_pass = sum(all(res["per_seed"][s]["rule_G5_vs_G3c"].values()) for s in SEEDS)
    res["decision"] = {"G5_adds_routing_information": bool(n_pass >= 2), "seeds_passing": int(n_pass)}
    res["cost"]["fit_s"] = float(np.mean(res["cost"]["fit_s"]))
    Path("reports/e8_trackB_results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["decision"], indent=1))


if __name__ == "__main__":
    main()
