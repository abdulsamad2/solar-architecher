"""Run E7b (reports/e7b_gate_prereg.md) -> reports/e7b_results.json, experiments/E7b_gate/v1/."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.blend import gated_w
from src.cal2 import ALPHA, interval_score
from src.calibrate import base_interval
from src.compare import bootstrap_diff
from src.gate2_report import ramp_labels, strata
from src.gate3_report import conformal
from src.gate7b import GATES, BlendCRPS, apply_gate, average_precision, grid_quantiles, raw_features, select_and_fit
from src.metrics import crps_quantile_interp

SEEDS, FOLDS = (42, 123, 456), range(5)
DIRECT, RESID = Path("experiments/E4_calib/v1/csi"), Path("experiments/E6/v1/E6-S/csi")
E7 = Path("experiments/E7_blend/v1")
OUT = Path("experiments/E7b_gate/v1")
REFS = ("E4-B-fit", "E6-S", "E7-B1")
METHODS = REFS + tuple(GATES)


def main():
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    frozen = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    ramp = ramp_labels("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz", folds_doc)
    res = {"per_seed": {}, "params": {}, "cost": {"fit_s": [], "feature_us": [], "gate_us": []}}
    for s in SEEDS:
        vals, cals, meta, w_all, g_all, params = {m: [] for m in METHODS}, {m: [] for m in METHODS}, [], {g: [] for g in GATES}, [], {}
        for k in FOLDS:
            tau = frozen[str(k)]["training_labels_fit_days"]
            sp = {}
            for split in ("cal", "val"):
                dd = dict(np.load(DIRECT / f"seed{s}" / f"fold_{k}" / f"preds_{split}.npz"))
                rr = dict(np.load(RESID / f"seed{s}" / f"fold_{k}" / f"preds_{split}.npz"))
                assert np.array_equal(dd["sample_ids"], rr["sample_ids"]), f"STOP: expert samples differ ({split})"
                t0 = time.perf_counter()
                feats = raw_features(dd, rr, tau)
                if split == "val":
                    res["cost"]["feature_us"].append((time.perf_counter() - t0) / len(dd["target"]) * 1e6)
                sp[split] = (dd, rr, feats)
            (dc, rc, fc), (dv, rv, fv) = sp["cal"], sp["val"]
            lv, yc = dc["quantile_levels"], dc["target"]
            order = np.argsort(dc["origin_time_us"], kind="stable")
            e7p = json.loads((E7 / f"seed{s}" / f"fold_{k}" / "params.json").read_text())
            e7v = np.load(E7 / f"seed{s}" / f"fold_{k}" / "preds_val.npz")
            assert np.array_equal(e7v["sample_ids"], dv["sample_ids"]), "STOP: E7 samples differ"
            assert np.allclose(e7v["gate"], fv["g"], atol=1e-12), "STOP: gate feature g differs from E7"
            wc_b1 = gated_w(fc["g"], e7p["B1_a"], e7p["B1_b"])
            blends = {"E4-B-fit": (dv["pred_q"], dc["pred_q"]), "E6-S": (rv["pred_q"], rc["pred_q"]),
                      "E7-B1": (e7v["q_B1 gated"] if "q_B1 gated" in e7v.files else e7v["q_B1"],
                                wc_b1[:, None] * dc["pred_q"] + (1 - wc_b1[:, None]) * rc["pred_q"])}
            params[k] = {}
            for name, feats in GATES.items():
                t0 = time.perf_counter()
                p = select_and_fit(fc, feats, dc["pred_q"], rc["pred_q"], yc, lv, order)   # calibration days only
                res["cost"]["fit_s"].append(time.perf_counter() - t0)
                wc = apply_gate(p, fc)
                chk = BlendCRPS(grid_quantiles(dc["pred_q"], lv), grid_quantiles(rc["pred_q"], lv), yc).per_sample(torch.as_tensor(wc)).numpy()
                ref = crps_quantile_interp(wc[:, None] * dc["pred_q"] + (1 - wc[:, None]) * rc["pred_q"], yc, lv)
                assert np.abs(chk - ref).max() < 1e-8, "STOP: fitting CRPS disagrees with reference scorer"
                t0 = time.perf_counter()
                wv = apply_gate(p, fv)
                res["cost"]["gate_us"].append((time.perf_counter() - t0) / len(wv) * 1e6)
                assert np.isfinite(wv).all() and np.isfinite(wc).all(), "STOP: non-finite gate weights"
                qv = wv[:, None] * dv["pred_q"] + (1 - wv[:, None]) * rv["pred_q"]
                qc = wc[:, None] * dc["pred_q"] + (1 - wc[:, None]) * rc["pred_q"]
                assert np.isfinite(qv).all() and (np.diff(qv, axis=1) >= -1e-9).all(), "STOP: invalid blended quantiles"
                blends[name] = (qv, qc)
                w_all[name].append(wv)
                params[k][name] = p
            (OUT / f"seed{s}" / f"fold_{k}").mkdir(parents=True, exist_ok=True)
            (OUT / f"seed{s}" / f"fold_{k}" / "params.json").write_text(json.dumps(params[k], indent=1))
            np.savez_compressed(OUT / f"seed{s}" / f"fold_{k}" / "preds_val.npz", sample_ids=dv["sample_ids"],
                                **{f"w_{g}": w_all[g][-1] for g in GATES}, **{f"q_{m}": blends[m][0] for m in METHODS})
            for m in METHODS:
                base = {x: dv[x] for x in ("origin_time_us", "target_time_us", "target", "quantile_levels")}
                basec = {x: dc[x] for x in ("origin_time_us", "target_time_us", "target", "quantile_levels")}
                vals[m].append({**base, "pred_q": blends[m][0]})
                cals[m].append({**basec, "pred_q": blends[m][1]})
            g_all.append(fv["g"])
            meta.append(dv)
        res["params"][s] = params
        cat = lambda key: np.concatenate([v[key] for v in meta])
        v0 = {key: cat(key) for key in ("sample_ids", "target", "clear_day", "fold", "day", "origin_time_us")}
        st = strata(v0, ramp["is_ramp"])
        st["fold 1 Sep-Oct 2017"] = (v0["fold"] == 1) & np.isin(
            v0["origin_time_us"].astype("datetime64[us]").astype("datetime64[M]").astype(str), ["2017-09", "2017-10"])
        y, lv = v0["target"], meta[0]["quantile_levels"]
        dayk = np.char.add(v0["fold"].astype(str), np.char.add("_", v0["day"].astype(str)))
        q = {m: np.concatenate([v["pred_q"] for v in vals[m]]) for m in METHODS}
        crps = {m: crps_quantile_interp(q[m], y, lv) for m in METHODS}
        raw = {m: {g: float(crps[m][mk].mean()) for g, mk in st.items()} for m in METHODS}
        pairs = {f"{g} - G0": (g, "G0") for g in ("G1", "G2", "G3", "G4")}
        pairs.update({f"{g} - E7-B1": (g, "E7-B1") for g in GATES})
        pairs.update({"G0 - E7-B1": ("G0", "E7-B1")})
        diffs = {n: {g: bootstrap_diff(crps[a][st[g]], crps[b][st[g]], dayk[st[g]], v0["clear_day"][st[g]], kind="mean")
                     for g in ("overall", "sunny", "cloudy", "ramp", "non-ramp")} for n, (a, b) in pairs.items()}
        lab = st["ramp"]
        gcat = np.concatenate(g_all)
        diag = {"ramp_base_rate": float(lab.mean()),
                "g_alone": {"auprc": average_precision(gcat, lab), "brier": float(np.mean((gcat - lab) ** 2))}}
        for gname in GATES:
            w = np.concatenate(w_all[gname])
            diag[gname] = {"auprc": average_precision(w, lab), "brier": float(np.mean((w - lab) ** 2)),
                           "w_ramp": [float(w[lab].mean())] + np.percentile(w[lab], [10, 50, 90]).round(4).tolist(),
                           "w_nonramp": [float(w[~lab].mean())] + np.percentile(w[~lab], [10, 50, 90]).round(4).tolist()}
        intervals = {}
        for m in METHODS:
            ci = conformal(cals[m], vals[m])
            intervals[m] = {kind: {g: {n: {"coverage": float(np.mean((y[mk] >= b_[n][0][mk]) & (y[mk] <= b_[n][1][mk]))),
                                           "width_kw": float(np.mean(b_[n][1][mk] - b_[n][0][mk])),
                                           "interval_score": float(interval_score(b_[n][0][mk], b_[n][1][mk], y[mk], ALPHA[n]).mean())}
                                       for n in ALPHA} for g, mk in st.items() if g in ("overall", "ramp", "non-ramp")}
                            for kind, b_ in (("raw", {n: base_interval(q[m], lv, n) for n in ALPHA}), ("rolling conformal", ci))}
        info = {gname: {"ramp_ci_below_0": bool(diffs[f"{gname} - G0"]["ramp"][2] < 0),
                        "overall_within_0.005": bool(raw[gname]["overall"] - raw["G0"]["overall"] <= 0.005)}
                for gname in ("G1", "G2", "G3", "G4")}
        res["per_seed"][s] = {"raw_crps": raw, "contrasts": diffs, "gate_diag": diag, "intervals": intervals, "info_test": info}
    # decision rules
    passing = [g for g in ("G1", "G2", "G3", "G4")
               if sum(all(res["per_seed"][s]["info_test"][g].values()) for s in SEEDS) >= 2]
    mean_raw = lambda m, g: float(np.mean([res["per_seed"][s]["raw_crps"][m][g] for s in SEEDS]))
    rec = min(passing, key=lambda g: (mean_raw(g, "ramp"), mean_raw(g, "overall"))) if passing else None
    closes = None
    if rec:
        closes = sum((res["per_seed"][s]["raw_crps"][rec]["ramp"] - res["per_seed"][s]["raw_crps"]["E4-B-fit"]["ramp"] <= 0.02)
                     and (res["per_seed"][s]["raw_crps"][rec]["overall"] - res["per_seed"][s]["raw_crps"]["E6-S"]["overall"] <= 0.02)
                     for s in SEEDS) >= 2
    res["decision"] = {"gates_adding_information": passing, "recommended_gate": rec, "closes_routing_gap": closes}
    res["cost"] = {k: float(np.mean(v)) for k, v in res["cost"].items()}
    Path("reports/e7b_results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["decision"], indent=1))


if __name__ == "__main__":
    main()
