"""E8 Track A analysis (reports/e8_prereg.md sections 5-6). Run only after the Track A runs exist.

Primary: common population, raw ramp CRPS vs the matched baseline D0. Secondary: full-population fallback
(arm where window-eligible, E4-B-fit otherwise) with the sunrise-ramp subgroup (ramps outside the common population).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.compare import bootstrap_diff
from src.events import implied_event_probability
from src.gate2_report import ramp_labels, strata
from src.gate7b import average_precision
from src.metrics import crps_quantile_interp

SEEDS, FOLDS = (42, 123, 456), range(5)
ROOT = Path("experiments/E8/v1")
E4BFIT = Path("experiments/E4_calib/v1/csi")
ARMS = ("D0", "D1", "D2", "D3", "D4", "D5", "D6")
D_O, D_S = 0.02, 0.015


def assemble_fallback(full_ids, base_q, arm_ids, arm_q):
    """Full-population quantiles: the arm's prediction where it has one, the base forecaster otherwise."""
    pos = {s: i for i, s in enumerate(full_ids)}
    out = base_q.copy()
    used = np.zeros(len(full_ids), bool)
    for s, q in zip(arm_ids, arm_q):
        i = pos[s]
        assert not used[i], "duplicate arm prediction"
        out[i], used[i] = q, True
    return out, used


def load_arm(arm, seed, extra=False):
    parts = []
    for k in FOLDS:
        p = ROOT / arm / "csi" / f"seed{seed}" / f"fold_{k}" / ("preds_val_extra.npz" if extra else "preds_val.npz")
        if p.exists():
            parts.append(dict(np.load(p)))
    return parts


def main():
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    rl = ramp_labels("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz", folds_doc)
    common_ids = set(np.load("data/features/e8_windows_v1.npz")["sample_ids"][np.load("data/features/e8_windows_v1.npz")["eligible_common"]])
    arms = [a for a in ARMS if (ROOT / a / "csi" / "seed42").exists()]
    res = {"arms": arms, "per_seed": {}}
    for s in SEEDS:
        base = [dict(np.load(E4BFIT / f"seed{s}" / f"fold_{k}" / "preds_val.npz")) for k in FOLDS]
        cat = lambda parts, key: np.concatenate([p[key] for p in parts])
        full_ids, full_y, full_q = cat(base, "sample_ids"), cat(base, "target"), cat(base, "pred_q")
        lv = base[0]["quantile_levels"]
        full_meta = {key: cat(base, key) for key in ("clear_day", "fold", "day", "origin_time_us")}
        full_meta["sample_ids"], full_meta["target"] = full_ids, full_y
        st_full = strata(full_meta, rl["is_ramp"])
        in_common = np.array([i in common_ids for i in full_ids])
        st_full["sunrise ramp"] = st_full["ramp"] & ~in_common
        per = {"common": {}, "fallback": {}}
        crps_c, crps_f = {}, {}
        ref_ids = None
        for a in arms:
            parts = load_arm(a, s)
            ids = cat(parts, "sample_ids")
            ref_ids = ids if ref_ids is None else ref_ids
            assert np.array_equal(ids, ref_ids), f"STOP: {a} common-population samples differ"
            assert set(ids) <= common_ids
            crps_c[a] = crps_quantile_interp(cat(parts, "pred_q"), cat(parts, "target"), lv)
            ex = load_arm(a, s, extra=True)
            arm_ids = np.concatenate([ids] + ([cat(ex, "sample_ids")] if ex else []))
            arm_q = np.concatenate([cat(parts, "pred_q")] + ([cat(ex, "pred_q")] if ex else []))
            fq, used = assemble_fallback(full_ids, full_q, arm_ids, arm_q)
            crps_f[a] = crps_quantile_interp(fq, full_y, lv)
            per["fallback"].setdefault("share_arm_used", {})[a] = float(used.mean())
        cm = {key: cat(load_arm(arms[0], s), key) for key in ("sample_ids", "target", "clear_day", "fold", "day", "origin_time_us")}
        st_c = strata(cm, rl["is_ramp"])
        dayk_c = np.char.add(cm["fold"].astype(str), np.char.add("_", cm["day"].astype(str)))
        dayk_f = np.char.add(full_meta["fold"].astype(str), np.char.add("_", full_meta["day"].astype(str)))
        groups = ("overall", "sunny", "cloudy", "ramp", "non-ramp")
        per["common"]["raw_crps"] = {a: {g: float(crps_c[a][st_c[g]].mean()) for g in groups} for a in arms}
        per["fallback"]["raw_crps"] = {a: {g: float(crps_f[a][st_full[g]].mean()) for g in groups + ("sunrise ramp",)} for a in arms}
        contrasts = {}
        for a in arms[1:]:
            contrasts[f"{a} - D0"] = {g: bootstrap_diff(crps_c[a][st_c[g]], crps_c["D0"][st_c[g]], dayk_c[st_c[g]], cm["clear_day"][st_c[g]], kind="mean") for g in groups}
        if "D4" in arms and "D2" in arms:
            contrasts["D4 - D2"] = {g: bootstrap_diff(crps_c["D4"][st_c[g]], crps_c["D2"][st_c[g]], dayk_c[st_c[g]], cm["clear_day"][st_c[g]], kind="mean") for g in groups}
        per["common"]["contrasts"] = contrasts
        per["rule"] = {a: {"ramp_ci_below_0": bool(contrasts[f"{a} - D0"]["ramp"][2] < 0),
                           "overall_noninferior": bool(per["common"]["raw_crps"][a]["overall"] - per["common"]["raw_crps"]["D0"]["overall"] <= D_O),
                           "sunny_noninferior": bool(per["common"]["raw_crps"][a]["sunny"] - per["common"]["raw_crps"]["D0"]["sunny"] <= D_S)}
                       for a in arms[1:]}
        # event occurrence quality on the common population (quantile-implied probability, each arm's K space)
        ev = {}
        for a in arms:
            prs = []
            for k, p in zip(FOLDS, load_arm(a, s)):
                qk = p["pred_q"] / np.maximum(p["pclear_target_kw"], 1.0)[:, None]
                kn = p["pv_now_kw"] / np.maximum(p["pclear_now_kw"], 1.0)
                prs.append(implied_event_probability(qk, lv, kn, rl["tau_ramp"][k]))
            pr = np.concatenate(prs)
            ev[a] = {"brier": float(np.mean((pr - st_c["ramp"]) ** 2)), "auprc": average_precision(pr, st_c["ramp"])}
        per["event"] = ev
        res["per_seed"][s] = per
    res["decision"] = {a: {"seeds_passing": int(sum(all(res["per_seed"][s]["rule"][a].values()) for s in SEEDS))} for a in arms[1:]}
    for a in res["decision"]:
        res["decision"][a]["improves_ramp_anticipation"] = res["decision"][a]["seeds_passing"] >= 2
    Path("reports/e8_trackA_results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["decision"], indent=1))


if __name__ == "__main__":
    main()
