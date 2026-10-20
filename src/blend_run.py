"""Run E7-BLEND (reports/e7_blend_prereg.md) -> reports/e7_blend_results.json, experiments/E7_blend/v1/."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.blend import apply, fit, gate_feature
from src.cal2 import ALPHA, interval_score
from src.calibrate import base_interval
from src.compare import bootstrap_diff
from src.gate2_report import ramp_labels, strata
from src.gate3_report import conformal
from src.metrics import crps_quantile_interp

SEEDS, FOLDS = (42, 123, 456), range(5)
RAMP = Path("experiments/E4_calib/v1/csi")
CALM = Path("experiments/E6/v1/E6-S/csi")
OUT = Path("experiments/E7_blend/v1")
METHODS = ("E4-B-fit (ramp expert)", "E6-S (calm expert)", "B0 constant", "B1 gated", "B2 switch")
DELTA = 0.02


def main():
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    frozen = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    ramp = ramp_labels("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz", folds_doc)
    res = {"per_seed": {}, "params": {}}
    for s in SEEDS:
        vals, cals, params = {m: [] for m in METHODS}, {m: [] for m in METHODS}, {}
        g_val_all, v_meta = [], []
        for k in FOLDS:
            tau = frozen[str(k)]["training_labels_fit_days"]
            d = {}
            for split in ("cal", "val"):
                r = dict(np.load(RAMP / f"seed{s}" / f"fold_{k}" / f"preds_{split}.npz"))
                c = dict(np.load(CALM / f"seed{s}" / f"fold_{k}" / f"preds_{split}.npz"))
                assert np.array_equal(r["sample_ids"], c["sample_ids"]), f"STOP: expert samples differ ({split})"
                assert np.allclose(r["target"], c["target"])
                d[split] = (r, c, gate_feature(c, tau))
            (rc, cc, gc), (rv, cv, gv) = d["cal"], d["val"]
            lv = rv["quantile_levels"]
            p = fit(rc["pred_q"], cc["pred_q"], gc, rc["target"], lv)      # calibration days only
            params[k] = p
            (OUT / f"seed{s}" / f"fold_{k}").mkdir(parents=True, exist_ok=True)
            (OUT / f"seed{s}" / f"fold_{k}" / "params.json").write_text(json.dumps(p, indent=1))
            t0 = time.perf_counter()
            bv = apply(p, rv["pred_q"], cv["pred_q"], gv)
            p["apply_us_per_sample"] = (time.perf_counter() - t0) / len(gv) * 1e6
            bc = apply(p, rc["pred_q"], cc["pred_q"], gc)
            for qs in list(bv.values()) + list(bc.values()):
                assert np.isfinite(qs).all() and (np.diff(qs, axis=1) >= -1e-9).all(), "STOP: invalid blended quantiles"
            np.savez_compressed(OUT / f"seed{s}" / f"fold_{k}" / "preds_val.npz", sample_ids=rv["sample_ids"],
                                gate=gv, **{f"q_{m}": q for m, q in bv.items()})
            for m, (qv, qc) in zip(METHODS, [(rv["pred_q"], rc["pred_q"]), (cv["pred_q"], cc["pred_q"]),
                                             (bv["B0"], bc["B0"]), (bv["B1"], bc["B1"]), (bv["B2"], bc["B2"])]):
                vals[m].append({**{x: rv[x] for x in ("origin_time_us", "target_time_us", "target", "quantile_levels")}, "pred_q": qv})
                cals[m].append({**{x: rc[x] for x in ("origin_time_us", "target_time_us", "target", "quantile_levels")}, "pred_q": qc})
            g_val_all.append(gv)
            v_meta.append(rv)
        res["params"][s] = params
        cat = lambda key: np.concatenate([v[key] for v in v_meta])
        v0 = {key: cat(key) for key in ("sample_ids", "target", "clear_day", "fold", "day", "origin_time_us")}
        st = strata(v0, ramp["is_ramp"])
        st["fold 1 Sep-Oct 2017"] = (v0["fold"] == 1) & np.isin(
            v0["origin_time_us"].astype("datetime64[us]").astype("datetime64[M]").astype(str), ["2017-09", "2017-10"])
        y, lv = v0["target"], v_meta[0]["quantile_levels"]
        dayk = np.char.add(v0["fold"].astype(str), np.char.add("_", v0["day"].astype(str)))
        q = {m: np.concatenate([v["pred_q"] for v in vals[m]]) for m in METHODS}
        crps = {m: crps_quantile_interp(q[m], y, lv) for m in METHODS}
        raw = {m: {g: float(crps[m][mk].mean()) for g, mk in st.items()} for m in METHODS}
        pairs = {"B1 - ramp expert": ("B1 gated", "E4-B-fit (ramp expert)"), "B1 - calm expert": ("B1 gated", "E6-S (calm expert)"),
                 "B1 - B0": ("B1 gated", "B0 constant"), "B2 - B1": ("B2 switch", "B1 gated"),
                 "B0 - calm expert": ("B0 constant", "E6-S (calm expert)")}
        diffs = {n: {g: bootstrap_diff(crps[a][st[g]], crps[b][st[g]], dayk[st[g]], v0["clear_day"][st[g]], kind="mean")
                     for g in ("overall", "sunny", "cloudy", "ramp", "non-ramp")} for n, (a, b) in pairs.items()}
        intervals = {}
        for m in METHODS:
            ci = conformal(cals[m], vals[m])
            intervals[m] = {kind: {g: {n: {"coverage": float(np.mean((y[mk] >= b_[n][0][mk]) & (y[mk] <= b_[n][1][mk]))),
                                           "width_kw": float(np.mean(b_[n][1][mk] - b_[n][0][mk])),
                                           "interval_score": float(interval_score(b_[n][0][mk], b_[n][1][mk], y[mk], ALPHA[n]).mean())}
                                       for n in ALPHA} for g, mk in st.items() if g in ("overall", "ramp", "non-ramp", "fold 1")}
                            for kind, b_ in (("raw", {n: base_interval(q[m], lv, n) for n in ALPHA}), ("rolling conformal", ci))}
        g_all = np.concatenate(g_val_all)
        w_b1 = np.concatenate([__import__("src.blend", fromlist=["gated_w"]).gated_w(gv, params[k]["B1_a"], params[k]["B1_b"])
                               for k, gv in zip(FOLDS, g_val_all)])
        gate_diag = {"brier_g_vs_ramp": float(np.mean((g_all - st["ramp"]) ** 2)),
                     "mean_w_B1_ramp": float(w_b1[st["ramp"]].mean()), "mean_w_B1_nonramp": float(w_b1[~st["ramp"]].mean()),
                     "share_switch_on_ramp": float(np.mean(np.concatenate([gv >= params[k]["B2_theta"] for k, gv in zip(FOLDS, g_val_all)])[st["ramp"]])),
                     "share_switch_on_nonramp": float(np.mean(np.concatenate([gv >= params[k]["B2_theta"] for k, gv in zip(FOLDS, g_val_all)])[~st["ramp"]]))}
        c1 = raw["B1 gated"]["ramp"] - raw["E4-B-fit (ramp expert)"]["ramp"] <= DELTA
        c2 = raw["B1 gated"]["overall"] - raw["E6-S (calm expert)"]["overall"] <= DELTA
        c3 = diffs["B1 - B0"]["ramp"][2] < 0
        res["per_seed"][s] = {"raw_crps": raw, "contrasts": diffs, "intervals": intervals, "gate": gate_diag,
                              "rule": {"c1_ramp_noninferior": bool(c1), "c2_overall_noninferior": bool(c2),
                                       "c3_gating_beats_constant": bool(c3)}}
    n = {c: sum(res["per_seed"][s]["rule"][c] for s in SEEDS) for c in ("c1_ramp_noninferior", "c2_overall_noninferior", "c3_gating_beats_constant")}
    retain = sum(all(res["per_seed"][s]["rule"].values()) for s in SEEDS) >= 2
    ens_only = (not retain) and sum(res["per_seed"][s]["rule"]["c1_ramp_noninferior"] and res["per_seed"][s]["rule"]["c2_overall_noninferior"] for s in SEEDS) >= 2
    res["decision"] = {"seeds_meeting_each_condition": n, "retain_B1": bool(retain), "ensembling_not_gating": bool(ens_only)}
    Path("reports/e7_blend_results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["decision"], indent=1))


if __name__ == "__main__":
    main()
