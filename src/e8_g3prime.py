"""E8 Track B follow-up (pre-registered, section 4): G3' = G3 recipe with the direct expert replaced by the best
passing Track-A arm, plus rolling-conformal interval metrics for all Track-A arms. CPU only; common population.

Best passing arm (pre-registered selection: lowest seed-mean ramp CRPS among arms passing the Track-A rule).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.cal2 import ALPHA, interval_score
from src.calibrate import base_interval
from src.compare import bootstrap_diff
from src.e8_features import FEAT
from src.gate2_report import ramp_labels, strata
from src.gate3_report import conformal
from src.gate7b import apply_gate, raw_features, select_and_fit
from src.metrics import crps_quantile_interp

SEEDS, FOLDS = (42, 123, 456), range(5)
E8 = Path("experiments/E8/v1")
E4BFIT, RESID = Path("experiments/E4_calib/v1/csi"), Path("experiments/E6/v1/E6-S/csi")


def load_split(path, split, keep_ids=None):
    d = dict(np.load(path / f"preds_{split}.npz"))
    if keep_ids is not None:
        keep = np.array([i in keep_ids for i in d["sample_ids"]])
        d = {k: (v[keep] if np.ndim(v) and len(v) == len(keep) else v) for k, v in d.items()}
    return d


def main():
    ta = json.loads(Path("reports/e8_trackA_results.json").read_text())
    passing = [a for a, v in ta["decision"].items() if v["improves_ramp_anticipation"]]
    mean_ramp = lambda a: np.mean([ta["per_seed"][s]["common"]["raw_crps"][a]["ramp"] for s in ta["per_seed"]])
    best = min(passing, key=mean_ramp) if passing else None
    assert best is not None, "no passing Track-A arm: G3' not defined"
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    frozen = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    rl = ramp_labels("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz", folds_doc)
    win = np.load(FEAT / "e8_windows_v1.npz")
    common_ids = set(win["sample_ids"][win["eligible_common"]])
    res = {"best_passing_arm": best, "per_seed": {}}
    for s in SEEDS:
        vals = {m: [] for m in ("G3c", "G3'", *ta["arms"])}
        cals = {m: [] for m in vals}
        meta = []
        for k in FOLDS:
            tau = frozen[str(k)]["training_labels_fit_days"]
            sp = {}
            for split in ("cal", "val"):
                r = load_split(RESID / f"seed{s}" / f"fold_{k}", split, common_ids)
                e4 = load_split(E4BFIT / f"seed{s}" / f"fold_{k}", split, common_ids)
                arms = {a: load_split(E8 / a / "csi" / f"seed{s}" / f"fold_{k}", split) for a in ta["arms"]}
                for a, d in arms.items():
                    assert np.array_equal(d["sample_ids"], r["sample_ids"]), f"STOP: {a} {split} samples differ"
                assert np.array_equal(e4["sample_ids"], r["sample_ids"])
                sp[split] = (r, e4, arms)
            (rc, e4c, ac), (rv, e4v, av) = sp["cal"], sp["val"]
            lv, order = rc["quantile_levels"], np.argsort(rc["origin_time_us"], kind="stable")
            for name, dc, dv in (("G3c", e4c, e4v), ("G3'", ac[best], av[best])):
                fc, fv = raw_features(dc, rc, tau), raw_features(dv, rv, tau)
                p = select_and_fit(fc, ("g", "d"), dc["pred_q"], rc["pred_q"], rc["target"], lv, order)
                wv, wc = apply_gate(p, fv), apply_gate(p, fc)
                qv = wv[:, None] * dv["pred_q"] + (1 - wv[:, None]) * rv["pred_q"]
                qc = wc[:, None] * dc["pred_q"] + (1 - wc[:, None]) * rc["pred_q"]
                assert np.isfinite(qv).all() and (np.diff(qv, axis=1) >= -1e-9).all()
                vals[name].append({**{x: rv[x] for x in ("origin_time_us", "target_time_us", "target", "quantile_levels")}, "pred_q": qv})
                cals[name].append({**{x: rc[x] for x in ("origin_time_us", "target_time_us", "target", "quantile_levels")}, "pred_q": qc})
            for a in ta["arms"]:
                vals[a].append({**{x: av[a][x] for x in ("origin_time_us", "target_time_us", "target", "quantile_levels", "pred_q")}})
                cals[a].append({**{x: ac[a][x] for x in ("origin_time_us", "target_time_us", "target", "quantile_levels", "pred_q")}})
            meta.append(rv)
        cat = lambda key: np.concatenate([v[key] for v in meta])
        v0 = {key: cat(key) for key in ("sample_ids", "target", "clear_day", "fold", "day", "origin_time_us")}
        st = strata(v0, rl["is_ramp"])
        y, lv = v0["target"], meta[0]["quantile_levels"]
        dayk = np.char.add(v0["fold"].astype(str), np.char.add("_", v0["day"].astype(str)))
        q = {m: np.concatenate([v["pred_q"] for v in vals[m]]) for m in vals}
        crps = {m: crps_quantile_interp(q[m], y, lv) for m in vals}
        groups = ("overall", "sunny", "cloudy", "ramp", "non-ramp")
        raw = {m: {g: float(crps[m][st[g]].mean()) for g in groups} for m in vals}
        diffs = {n: {g: bootstrap_diff(crps[a][st[g]], crps[b][st[g]], dayk[st[g]], v0["clear_day"][st[g]], kind="mean") for g in groups}
                 for n, (a, b) in {"G3' - G3c": ("G3'", "G3c"), f"G3' - {best}": ("G3'", best)}.items()}
        intervals = {}
        for m in vals:
            ci = conformal(cals[m], vals[m])
            intervals[m] = {g: {n: {"coverage": float(np.mean((y[mk] >= ci[n][0][mk]) & (y[mk] <= ci[n][1][mk]))),
                                    "width_kw": float(np.mean(ci[n][1][mk] - ci[n][0][mk])),
                                    "interval_score": float(interval_score(ci[n][0][mk], ci[n][1][mk], y[mk], ALPHA[n]).mean())}
                                for n in ALPHA} for g, mk in st.items() if g in ("overall", "ramp", "non-ramp")}
        rule = {"ramp_ci_below_0": bool(diffs["G3' - G3c"]["ramp"][2] < 0),
                "overall_within_0.005": bool(raw["G3'"]["overall"] - raw["G3c"]["overall"] <= 0.005)}
        res["per_seed"][s] = {"raw_crps": raw, "contrasts": diffs, "rolling_conformal": intervals, "rule_G3prime_vs_G3c": rule}
    n = sum(all(res["per_seed"][s]["rule_G3prime_vs_G3c"].values()) for s in SEEDS)
    res["decision"] = {"G3prime_adds_routing_value": bool(n >= 2), "seeds_passing": int(n)}
    Path("reports/e8_g3prime_results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"best": best, **res["decision"]}, indent=1))


if __name__ == "__main__":
    main()
