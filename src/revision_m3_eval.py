"""Revision R-M3: the single evaluation of method v3 (three-way gated probability mixture M-R'RC with the fallback M-DC)
on the official test set, the forward period and the external days (protocol
revision/protocols/R-M3_method_v3_development.md, Amendment 3). Scores, contrasts and reading rules are those of R-R9,
R-R7 and R-R10; method v2 is recomputed and must reproduce its stored score. Run: python -m src.revision_m3_eval test|forward|external
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from src import revision_posthoc_dump as PD
from src.revision_common import LEVELS, save
from src.revision_r7_forward_v2 import apply_mix, fit_mix

NROOT = Path("D:/skippd_hires/R_M3/Rdk/csi")          # R' (Stage A), five folds x three seeds
NFWD = "D:/skippd_hires/R_M3/fwd/Rdk"                # R' trained on the forward fit days, five seeds
OUT = Path("D:/skippd_hires/R_M3/eval")
PD.EXPERTS["Rdk"] = (NROOT, "H60")
KEYS = ("crps", "tw_ramp", "tw_calm")
V3, V2, D3 = "method v3", "method v2", ("N", "R", "C")
f3 = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"


def stored(name):
    return json.loads(Path("revision/results", name).read_text())


# ======================================================================= official test (as R-R9)
def test():
    from src.e8_features import FEAT
    from src.ramp_scoring import ramp_weighted_crps
    from src.revision_common import day_bootstrap, frozen_test_predictions, test_context
    from src.revision_m1_method_v2 import ROOT
    from src.revision_posthoc_test import TAU, mdict, members, sub
    from src.revision_r2_day_level_inference import sign_flip, t_interval
    from src.revision_z_common import ramp_region

    if not (PD.OUT / "test_members_Rdk.npz").exists():
        PD.main(["Rdk"])
    c = test_context()
    h60, y, day, sunny = c["h60"], c["y"], c["day"], c["sunny"]
    _, lo, hi = ramp_region(c)
    d, r, a, nn = members("E4-B-fit"), members("E6-S"), members("A2"), members("Rdk")
    common = np.load(FEAT / "e8_windows_v1.npz")
    common = common["sample_ids"][common["eligible_common"]]
    cal = lambda root, s, k: dict(np.load(root / f"seed{s}" / f"fold_{k}" / "preds_cal.npz"))
    acc2, acc3 = [], []
    for i, (s, k) in enumerate(r["members"]):
        assert tuple(nn["members"][i]) == (s, k)
        tau = TAU[str(k)]["training_labels_fit_days"]
        cd, cc, cr, cn = cal(ROOT["D"], s, k), cal(ROOT["C"], s, k), cal(ROOT["R"], s, k), cal(NROOT, s, k)
        p_dc = fit_mix({"D": cd, "C": cc}, tau, ("D", "C"))
        cc2, cr2, cn2 = (sub(x, np.isin(x["sample_ids"], common)) for x in (cc, cr, cn))
        assert np.array_equal(cc2["sample_ids"], cr2["sample_ids"]) and np.array_equal(cc2["sample_ids"], cn2["sample_ids"])
        p_rc = fit_mix({"R": cr2, "C": cc2}, tau, ("R", "C"))
        p_3 = fit_mix({"N": cn2, "R": cr2, "C": cc2}, tau, D3)
        md, mr, ma, mn = mdict(d, i), mdict(r, i), mdict(a, i), mdict(nn, i)
        q2 = apply_mix(p_dc, {"D": md, "C": mr}, tau, ("D", "C"))
        q3 = q2.copy()
        q2[h60] = apply_mix(p_rc, {"R": sub(ma, h60), "C": sub(mr, h60)}, tau, ("R", "C"))
        q3[h60] = apply_mix(p_3, {"N": sub(mn, h60), "R": sub(ma, h60), "C": sub(mr, h60)}, tau, D3)
        acc2.append(q2); acc3.append(q3)
        print(f"member {i + 1}/15 (seed {s}, fold {k}) done", flush=True)
    fz = frozen_test_predictions(c["sample_ids"])
    qs = {V3: np.mean(acc3, 0), V2: np.mean(acc2, 0), "G3''-F": fz["G3''-F"], "G4": fz["G4"], "E6-S": fz["E6-S"], "E4-B-fit": fz["E4-B-fit"],
          "PV-only P15": np.nanmean(members("P15")["pred_q"], 0), "R' (new 60-min expert)": np.where(h60[:, None], np.nanmean(nn["pred_q"], 0), np.nan)}
    r9 = stored("r9_method_v2_official_test.json")
    sc = {m: dict(zip(KEYS, ramp_weighted_crps(v, LEVELS, y, lo, hi))) for m, v in qs.items() if not m.startswith("R'")}
    assert abs(sc[V2]["crps"].mean() - r9["scores"]["method v2"]["crps"]) < 1e-9, "STOP: method v2 not reproduced"
    ramp = c["ramp"]
    res = {"protocol": "revision/protocols/R-M3_method_v3_development.md (Amendment 3)", "n": int(len(y)),
           "scores": {m: {kk: float(v[kk].mean()) for kk in KEYS} | {"sunny": float(v["crps"][sunny].mean()), "cloudy": float(v["crps"][~sunny].mean()),
                                                                       "ramp_outcomes": float(v["crps"][ramp].mean()), "nonramp_outcomes": float(v["crps"][~ramp].mean())}
                      for m, v in sc.items()}, "v3_minus": {}}
    days, inv = np.unique(day, return_inverse=True)
    sday = np.zeros(len(days), bool)
    sday[inv] = sunny
    for m in sc:
        if m == V3:
            continue
        dd = {kk: sc[V3][kk] - sc[m][kk] for kk in KEYS}
        S, n = np.bincount(inv, weights=dd["crps"]), np.bincount(inv).astype(float)
        rr = {kk: day_bootstrap(dd[kk], day, sunny) for kk in KEYS} | {"sign_flip_p": sign_flip(S, n)[0], "t_interval_day_means": t_interval(S / n, sday)[0],
                                                                       "days_v3_better": int((S < 0).sum()), "skill": float(1 - sc[V3]["crps"].mean() / sc[m]["crps"].mean())}
        rr["better"] = bool(rr["crps"][2] < 0 and rr["sign_flip_p"] < 0.05)
        res["v3_minus"][m] = rr
        print(f"v3 - {m:12s} CRPS {res['scores'][m]['crps']:.4f}: {f3(rr['crps'])} skill {rr['skill']:+.1%} | ramp region {f3(rr['tw_ramp'])} | calm {f3(rr['tw_calm'])} | "
              f"sign-flip p {rr['sign_flip_p']:.5f} ({rr['days_v3_better']}/20 days) => {'BETTER' if rr['better'] else 'not shown'}")
    refs = stored("r3_probabilistic_references.json")["references"]
    v3 = res["scores"][V3]["crps"]
    res["skill_vs_naive_references"] = {k: float(1 - v3 / refs[k]["crps"]) for k in ("SP-dressed adaptive", "PeEn-120", "PeEn-60", "PeEn-30", "SP-dressed static", "CH-PeEn")}
    res["skill_vs_official_smart_persistence_point"] = float(1 - v3 / 1.707)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "test_quantiles.npz", sample_ids=c["sample_ids"], y=y, **{m.replace(" ", "_"): qs[m] for m in (V3, V2)})
    print(f"method v3 CRPS {v3:.4f} (v2 {res['scores'][V2]['crps']:.4f}); skill vs naive references { {k: round(v, 3) for k, v in res['skill_vs_naive_references'].items()} }")
    print("saved", save("m3_method_v3_official_test.json", res))


# ======================================================================= forward period (as R-R7)
def forward():
    from src.baselines import official_persistence
    from src.cal2 import ALPHA, conformity, replay
    from src.calibrate import base_interval
    from src.final_eval import Split, day_bootstrap
    from src.ramp_scoring import ramp_weighted_crps, selection_null
    from src.revision_e3 import FOLDS, MODELS, SEEDS, W, ensemble_on, fit_anchor, fit_gates, predict, sel_preds, sub
    from src.revision_r7_forward_v2 import P, block_tests, naive_references

    dev = Split("dev", torch.device("cuda"))
    anchor = fit_anchor(dev)
    day_all = dev.ds.day.astype(str)
    idx = np.flatnonzero(np.isin(day_all, FOLDS["eval_days"]))
    gates_by_seed = {s: fit_gates(s) for s in SEEDS}
    q, h60 = ensemble_on(dev, idx, anchor, gates_by_seed)
    acc2, acc3, sel3 = [], [], []
    for s in SEEDS:
        tau = gates_by_seed[s][1]
        r, d, a, nw = sel_preds("E6-S", s), sel_preds("E4-B-fit", s), sel_preds("A2", s), sel_preds(NFWD, s)
        assert np.array_equal(a["sample_ids"], nw["sample_ids"]), "STOP: selection samples of R' differ from those of the 60-min expert"
        hs = np.isin(r["sample_ids"], a["sample_ids"])
        p_dc = fit_mix({"D": d, "C": r}, tau, ("D", "C"))
        p_rc = fit_mix({"R": a, "C": sub(r, hs)}, tau, ("R", "C"))
        p_3 = fit_mix({"N": nw, "R": a, "C": sub(r, hs)}, tau, D3)
        qs = apply_mix(p_dc, {"D": d, "C": r}, tau, ("D", "C"))
        qs[hs] = apply_mix(p_3, {"N": nw, "R": a, "C": sub(r, hs)}, tau, D3)
        sel3.append(qs)
        e4, e6 = (predict(dev, n, s, idx, anchor) for n in ("E4-B-fit", "E6-S"))
        a2, n2 = predict(dev, "A2", s, idx[h60], anchor), predict(dev, NFWD, s, idx[h60], anchor)
        q2 = apply_mix(p_dc, {"D": e4, "C": e6}, tau, ("D", "C"))
        q3 = q2.copy()
        q2[h60] = apply_mix(p_rc, {"R": a2, "C": sub(e6, h60)}, tau, ("R", "C"))
        q3[h60] = apply_mix(p_3, {"N": n2, "R": a2, "C": sub(e6, h60)}, tau, D3)
        acc2.append(q2); acc3.append(q3)
        print(f"seed {s} done", flush=True)
    q[V2], q[V3] = np.mean(acc2, 0), np.mean(acc3, 0)
    tau = gates_by_seed[SEEDS[0]][1]
    y = dev.pv[dev.ds.target_rows[idx]]
    d_ = day_all[idx]
    sunny = np.array([FOLDS["day_labels"][x] for x in d_])
    pc_t = np.maximum(anchor(dev.rec_t), 1.0)[dev.ds.target_rows[idx]]
    kn = (dev.pv / np.maximum(anchor(dev.rec_t), 1.0))[dev.ds.hist_rows[idx, -1]]
    lo, hi = (kn - tau) * pc_t, (kn + tau) * pc_t
    ramp = np.abs(y / pc_t - kn) >= tau
    q |= naive_references(dev, anchor, idx, FOLDS["folds"][0]["train_days"])
    sc = {m: dict(zip(KEYS, ramp_weighted_crps(v, LEVELS, y, lo, hi))) for m, v in q.items() if m != "A2"}
    pers = official_persistence(dev.ds.origin_time[idx].astype(object), dev.pv[dev.ds.hist_rows[idx, -1]])
    sc["official smart persistence (point)"] = dict(zip(KEYS, ramp_weighted_crps(np.repeat(pers[:, None], 9, 1), LEVELS, y, lo, hi)))
    r7 = stored("r7_forward_method_v2.json")
    assert abs(sc[V2]["crps"].mean() - r7["scores"]["method v2"]["crps"]) < 1e-6, "STOP: method v2 (R-R7) not reproduced"
    res = {"protocol": "revision/protocols/R-M3_method_v3_development.md (Amendment 3)", "n": int(len(idx)), "days": int(len(np.unique(d_))), "ramp_share": float(ramp.mean()),
           "scores": {m: {kk: float(v[kk].mean()) for kk in KEYS} | {"sunny": float(v["crps"][sunny].mean()), "cloudy": float(v["crps"][~sunny].mean()),
                                                                       "ramp_outcomes": float(v["crps"][ramp].mean())} for m, v in sc.items()}, "v3_minus": {}}
    for m in sc:
        if m == V3:
            continue
        dd = {kk: sc[V3][kk] - sc[m][kk] for kk in KEYS}
        r = {kk: day_bootstrap(dd[kk], d_, sunny) for kk in KEYS} | {"blocks": block_tests(dd["crps"], d_, FOLDS["eval_days"]),
                                                                      "skill": float(1 - sc[V3]["crps"].mean() / sc[m]["crps"].mean())}
        r["better"] = bool(r["crps"][2] < 0 and r["blocks"]["sign_flip_p"] < 0.05)
        res["v3_minus"][m] = r
        print(f"v3 - {m:36s} CRPS {res['scores'][m]['crps']:.4f}: {f3(r['crps'])} skill {r['skill']:+.1%} | ramp region {f3(r['tw_ramp'])} | calm {f3(r['tw_calm'])} | "
              f"7-day blocks p {r['blocks']['sign_flip_p']:.4f} ({r['blocks']['blocks_first_better']}/{r['blocks']['blocks']}) => {'BETTER' if r['better'] else 'not shown'}", flush=True)
    # continuous conformal intervals of method v3 and the selection reference for ramp coverage (as R-R7)
    rs = sel_preds("E6-S", SEEDS[0])
    qsel, ys, os_ = np.mean(sel3, 0), rs["target"], rs["origin_time_us"]
    order_s = np.argsort(os_, kind="stable")
    org = dev.ds.origin_time.astype("datetime64[us]").astype(np.int64)[idx]
    tgt = dev.ds.target_time.astype("datetime64[us]").astype(np.int64)[idx]
    ci = {}
    for nm, a in ALPHA.items():
        scs = conformity(*base_interval(qsel, LEVELS, nm), ys)[order_s]
        l_, h_ = base_interval(q[V3], LEVELS, nm)
        rep = replay(org, tgt, l_, h_, y, a, "rolling", W["G3''-F"], scs[-W["G3''-F"]:])
        ci[nm] = (rep["lo"], rep["hi"])
    nodes = np.sort(np.stack([ci["90"][0], ci["80"][0], ci["50"][0], q[V3][:, 4], ci["50"][1], ci["80"][1], ci["90"][1]], 1), 1)
    res["coverage"] = {}
    for nm in ALPHA:
        cov = (y >= ci[nm][0]) & (y <= ci[nm][1])
        null = selection_null(nodes, P, ci[nm][0], ci[nm][1], lo, hi, y=y, cap=1.5 * pc_t, seed=20261115)
        nominal = 1 - ALPHA[nm]
        res["coverage"][nm] = {"overall": float(cov.mean()), "ramp_observed": null["observed_ramp_coverage"], "ramp_selection_reference": null["null_ramp_coverage"],
                               "reference_range": null["null_ramp_coverage_range"], "gap": null["gap"],
                               "selection_share": float((nominal - null["null_ramp_coverage"]) / (nominal - null["observed_ramp_coverage"])) if nominal > null["observed_ramp_coverage"] else None,
                               "width_kw": float(np.mean(ci[nm][1] - ci[nm][0]))}
        c = res["coverage"][nm]
        print(f"{nm}% intervals: overall {c['overall']:.3f}, ramps observed {c['ramp_observed']:.3f}, selection reference {c['ramp_selection_reference']:.3f}, width {c['width_kw']:.2f} kW")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "forward_quantiles.npz", idx=idx, y=y, method_v3=q[V3], method_v2=q[V2], lo90=ci["90"][0], hi90=ci["90"][1])
    print("saved", save("m3_method_v3_forward.json", res))


# ======================================================================= external days (as R-R10)
def external():
    from src.cal2 import replay
    from src.clear_sky import ClearSkyPV
    from src.e8_features import FEAT
    from src.final_eval import SEEDS, blend, day_bootstrap, gate_params
    from src.metrics import crps_quantile_interp
    from src.revision_c21 import crps_ens, sample
    from src.revision_m1_method_v2 import ROOT
    from src.revision_posthoc_test import sub
    from src.revision_r10_external_full import PUBLISHED, US_MIN, build_split

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    split, t, y, info = build_split(torch.device("cuda"))
    h60, n = split.h60_ok, len(y)
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    tau_all = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    common = np.load(FEAT / "e8_windows_v1.npz")
    common = common["sample_ids"][common["eligible_common"]]
    cal = lambda root, s, k: dict(np.load(root / f"seed{s}" / f"fold_{k}" / "preds_cal.npz"))
    pv_now = split.pv[split.ds.hist_rows[:, -1]]
    names = ("method v3, routed", "method v2, routed", "frozen rule, routed", "G4", "calm expert", "direct expert")
    acc = {m: [] for m in names}
    for s in SEEDS:
        for k in range(5):
            fold, tau = {**folds_doc["folds"][k]}, tau_all[str(k)]["training_labels_fit_days"]
            rd, rc, ra, rn = (PD.run_member(split, e, s, fold, tau) for e in ("E4-B-fit", "E6-S", "A2", "Rdk"))
            md, mr, ma, mn = ({"pred_q": r["pred_q"], "pclear_target_kw": r["pclear_target_kw"], "pclear_now_kw": r["pclear_now_kw"], "pv_now_kw": pv_now,
                               "quantile_levels": LEVELS} for r in (rd, rc, ra, rn))
            routed = blend(md, mr, gate_params("G3", s, k), tau)
            routed[h60] = blend(sub(ma, h60), sub(mr, h60), gate_params("G3''", s, k), tau)
            acc["frozen rule, routed"].append(routed)
            acc["G4"].append(blend(md, mr, gate_params("G4", s, k), tau))
            acc["calm expert"].append(mr["pred_q"]); acc["direct expert"].append(md["pred_q"])
            cd, cc, cr, cn = cal(ROOT["D"], s, k), cal(ROOT["C"], s, k), cal(ROOT["R"], s, k), cal(NROOT, s, k)
            p_dc = fit_mix({"D": cd, "C": cc}, tau, ("D", "C"))
            cc2, cr2, cn2 = (sub(x, np.isin(x["sample_ids"], common)) for x in (cc, cr, cn))
            p_rc = fit_mix({"R": cr2, "C": cc2}, tau, ("R", "C"))
            p_3 = fit_mix({"N": cn2, "R": cr2, "C": cc2}, tau, D3)
            v2 = apply_mix(p_dc, {"D": md, "C": mr}, tau, ("D", "C"))
            v3 = v2.copy()
            v2[h60] = apply_mix(p_rc, {"R": sub(ma, h60), "C": sub(mr, h60)}, tau, ("R", "C"))
            v3[h60] = apply_mix(p_3, {"N": sub(mn, h60), "R": sub(ma, h60), "C": sub(mr, h60)}, tau, D3)
            acc["method v2, routed"].append(v2); acc["method v3, routed"].append(v3)
            print(f"member seed {s} fold {k} done", flush=True)
    q = {m: np.mean(v, 0) for m, v in acc.items()}
    r10 = stored("r10_external_days_full_method.json")
    crps = {m: crps_quantile_interp(v, y, LEVELS) for m, v in q.items()}
    for m in ("method v2, routed", "frozen rule, routed", "G4"):
        assert abs(crps[m].mean() - r10["scores"][m]["crps_exact"]) < 1e-6, f"STOP: {m} (R-R10) not reproduced"
    day = t.astype("datetime64[D]").astype(str)
    zero = np.zeros(n, bool)
    rdj = json.loads(Path("reports/test_ramp_definition_frozen.json").read_text())
    anchor = ClearSkyPV(rdj["anchor_coef"])
    ramp = np.abs(y / np.maximum(anchor(t + np.timedelta64(15, "m")), 1.0) - pv_now / np.maximum(anchor(t), 1.0)) >= rdj["tau_test"]
    rng = np.random.default_rng(20261116)
    res = {"protocol": "revision/protocols/R-M3_method_v3_development.md (Amendment 3)", **info, "n_ramp": int(ramp.sum()), "published": PUBLISHED, "scores": {}, "paired": {}}
    for m, v in q.items():
        r = {"crps_exact": float(crps[m].mean()), "ramp": float(crps[m][ramp].mean()), "non_ramp": float(crps[m][~ramp].mean()), "with_60min_history": float(crps[m][h60].mean())}
        for mm in (10, 30, 100):
            r[f"standard_m{mm}"] = float(np.mean([crps_ens(sample(v, mm, rng), y, False).mean() for _ in range(20)]))
        res["scores"][m] = r
        print(f"{m:22s} exact {r['crps_exact']:.3f} | m=10 {r['standard_m10']:.3f}  m=30 {r['standard_m30']:.3f}  m=100 {r['standard_m100']:.3f} | ramp {r['ramp']:.3f} non-ramp {r['non_ramp']:.3f}", flush=True)
    for b in names[1:]:
        res["paired"][f"method v3, routed - {b}"] = day_bootstrap(crps["method v3, routed"] - crps[b], day, zero)
        print(f"v3 - {b}: {f3(res['paired'][f'method v3, routed - {b}'])} (five days; descriptive)")
    cald = np.load("experiments/final/dev_oof_calibration_v1.npz")
    W = json.loads(Path("experiments/final/dev_oof_calibration_v1.json").read_text())["W"]["G3''-F"]
    init = cald["G3''-F|90"][np.argsort(cald["origin_us"].astype(np.int64), kind="stable")][-W:]
    org = t.astype(np.int64)
    tgt = org + 15 * US_MIN
    lo, hi = q["method v3, routed"][:, 0], q["method v3, routed"][:, -1]
    out = {"raw": (lo, hi)}
    for meth, par in (("continuous rolling", ("rolling", W)), ("ACI", ("aci", 0.005))):
        rep = replay(org, tgt, lo, hi, y, 0.1, par[0], par[1], init)
        out[meth] = (rep["lo"], rep["hi"])
    res["coverage90"] = {kk: {"coverage": float(np.mean((y >= l) & (y <= h))), "ramp_coverage": float(np.mean((y[ramp] >= l[ramp]) & (y[ramp] <= h[ramp]))),
                              "width_kw": float(np.mean(h - l))} for kk, (l, h) in out.items()}
    print("90% interval, method v3:", " | ".join(f"{kk} cov {v['coverage']:.3f} (ramps {v['ramp_coverage']:.3f}) width {v['width_kw']:.2f}" for kk, v in res["coverage90"].items()))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "external_quantiles.npz", y=y, method_v3=q["method v3, routed"], method_v2=q["method v2, routed"])
    print("saved", save("m3_method_v3_external.json", res))


if __name__ == "__main__":
    {"test": test, "forward": forward, "external": external}[sys.argv[1]]()
