"""Revision R-R15: evaluation of the 15-min form (gated probability mixture of the direct and calm experts on every
forecast) as the proposed method (protocol revision/protocols/R-R15_method_15min_evaluation.md). Post hoc; no training;
nothing here selects or changes a model, a gate or the calibration window.

Run: python -m src.revision_r15_method_15min dev       (gate ablation on development out-of-fold forecasts; CPU)
     python -m src.revision_r15_method_15min test      (official test set; CPU)
     python -m src.revision_r15_method_15min forward   (forward period; GPU inference of the comparators)
     python -m src.revision_r15_method_15min external  (external days; GPU inference)
Each invocation writes its key into revision/results/r15_method_15min.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.e8_features import FEAT
from src.ramp_scoring import ramp_weighted_crps
from src.revision_common import LEVELS, RESULTS, save
from src.revision_m1_method_v2 import ROOT, TAU
from src.revision_r13_explanatory import (EVAL, GATES, KEYS, apply_gate, day_tests, direction_coverage, fit_gate, gap_bootstrap, nodes7, null_counts,
                                          summary)

OUT = "r15_method_15min.json"
PROTOCOL = "revision/protocols/R-R15_method_15min_evaluation.md"
NAME = "SHADOW-MoE (15 min)"
V2 = "60-min design (method v2)"
QB = "quantile blend, same experts"
SEEDS, W = (42, 123, 456), 4620
GROUPS = ("overall", "sunny", "cloudy", "ramp", "non-ramp")
f3 = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"
stored = lambda n: json.loads((RESULTS / n).read_text())


def put(**parts):
    p = RESULTS / OUT
    res = json.loads(p.read_text()) if p.exists() else {"protocol": PROTOCOL, "method": NAME}
    res.update(parts)
    print("saved", save(OUT, res), "keys:", ", ".join(parts))


def ld(e, s, k, part):
    return dict(np.load(ROOT[e] / f"seed{s}" / f"fold_{k}" / f"preds_{part}.npz"))


def dev_oof(with_gates=False):
    """Development out-of-fold quantiles of the 15-min form per seed, in the order of the frozen development calibration file."""
    ref = np.load("experiments/final/dev_oof_calibration_v1.npz")
    sid, org = ref["sample_ids"], ref["origin_us"].astype(np.int64)
    order = np.argsort(sid, kind="stable")
    names = list(GATES) if with_gates else ["gate on g and d"]
    acc = {s: {g: np.full((len(sid), len(LEVELS)), np.nan) for g in names} for s in SEEDS}
    y, day, sunny, lo, hi = np.full(len(sid), np.nan), np.empty(len(sid), object), np.zeros(len(sid), bool), np.full(len(sid), np.nan), np.full(len(sid), np.nan)
    from src.revision_r1_gate_proper_score import band
    for s in SEEDS:
        for k in range(5):
            tau = TAU[str(k)]["training_labels_fit_days"]
            cd, cc, vd, vc = ld("D", s, k, "cal"), ld("C", s, k, "cal"), ld("D", s, k, "val"), ld("C", s, k, "val")
            assert np.array_equal(cd["sample_ids"], cc["sample_ids"]) and np.array_equal(vd["sample_ids"], vc["sample_ids"])
            pos = order[np.searchsorted(sid[order], vc["sample_ids"])]
            assert np.array_equal(sid[pos], vc["sample_ids"]), "STOP: validation samples not in the development calibration file"
            for g in names:
                acc[s][g][pos] = apply_gate(fit_gate({"D": cd, "C": cc}, tau, ("D", "C"), GATES[g]), {"D": vd, "C": vc}, tau, ("D", "C"))
            y[pos], day[pos], sunny[pos] = vc["target"], vc["day"], vc["clear_day"].astype(bool)
            lo[pos], hi[pos] = band(vc, tau)
            print(f"development seed {s} fold {k} done", flush=True)
    assert all(np.isfinite(acc[s][g]).all() for s in SEEDS for g in names), "STOP: not every development sample has three seeds"
    return acc, y, org, day.astype(str), sunny, lo, hi


# ======================================================================= development: gate ablation
def dev():
    from src.revision_common import day_bootstrap

    acc, y, _, day, sunny, lo, hi = dev_oof(with_gates=True)
    m1 = stored("m1_method_v2_development.json")["per_seed"]
    D = {"per_seed": {}, "gate_minus": {g: {} for g in GATES if g != "gate on g and d"}}
    for s in SEEDS:
        sc = {g: dict(zip(KEYS, ramp_weighted_crps(acc[s][g], LEVELS, y, lo, hi))) for g in GATES}
        assert abs(sc["gate on g and d"]["crps"].mean() - m1[str(s)]["M-DC"]["crps"]) < 1e-9, "STOP: M-DC (R-M1) not reproduced"
        assert abs(sc["constant weight"]["crps"].mean() - m1[str(s)]["M-DC const"]["crps"]) < 1e-9, "STOP: M-DC const (R-M1) not reproduced"
        D["per_seed"][str(s)] = {g: summary(sc[g], sunny) for g in GATES}
        for g in D["gate_minus"]:
            D["gate_minus"][g][str(s)] = {kk: day_bootstrap(sc["gate on g and d"][kk] - sc[g][kk], day, sunny) for kk in KEYS}
        print(f"seed {s}: " + "  ".join(f"{g} {sc[g]['crps'].mean():.4f}" for g in GATES), flush=True)
    D["seed_mean"] = {g: {kk: float(np.mean([D["per_seed"][str(s)][g][kk] for s in SEEDS])) for kk in KEYS} for g in GATES}
    D["seeds_with_interval_below_zero"] = {g: {kk: int(sum(v[kk][2] < 0 for v in e.values())) for kk in KEYS} for g, e in D["gate_minus"].items()}
    D["ensemble_of_three_seeds_crps"] = float(ramp_weighted_crps(np.mean([acc[s]["gate on g and d"] for s in SEEDS], 0), LEVELS, y, lo, hi)[0].mean())
    for g, e in D["gate_minus"].items():
        print(f"gate(g,d) - {g:18s} " + " | ".join(f"seed {s}: {f3(v['crps'])}" for s, v in e.items()))
    put(development=D)


# ======================================================================= official test set
def test():
    from src import revision_r4_ramp_warnings as r4
    from src import revision_r5_reserve as r5
    from src.baselines import official_persistence
    from src.cal2 import ALPHA, conformity, replay
    from src.calibrate import base_interval
    from src.final_eval import blend, gate_params, rolling_conformal_daily
    from src.metrics import crps_quantile_interp
    from src.revision_c07 import interval_table, past_only_init
    from src.revision_common import day_bootstrap, frozen_test_predictions, test_context
    from src.revision_posthoc_test import mdict, members, test_k
    from src.revision_r3_probabilistic_references import main as r3_refs
    from src.revision_r5_reserve import dressed_q05, equalise
    from src.revision_r11_test_reliability import point
    from src.revision_x2_linear_pool import _cdf
    from src.revision_z_common import ramp_region

    c = test_context()
    y, day, sunny, ramp, masks = c["y"], c["day"], c["sunny"], c["ramp"], c["masks"]
    pc, lo, hi = ramp_region(c)
    T = lambda d, m=None: day_tests(d if m is None else d[m], day if m is None else day[m], sunny if m is None else sunny[m], day_bootstrap)
    score = lambda q: dict(zip(KEYS, ramp_weighted_crps(q, LEVELS, y, lo, hi)))
    z = np.load(EVAL / "test_quantiles_r13.npz")
    assert np.array_equal(z["sample_ids"], c["sample_ids"])
    q15 = z["mdc_all"]
    r13 = stored("r13_explanatory_analyses_method_v2.json")
    assert abs(score(q15)["crps"].mean() - r13["A_test"]["scores"]["M-DC all"]["crps"]) < 1e-9, "STOP: 15-min form (R-R13) not reproduced"

    # ---- members: gate variants and the quantile blend of the same experts with the frozen 15-min gate
    d, r = members("E4-B-fit"), members("E6-S")
    acc = {g: [] for g in list(GATES) + [QB]}
    for i, (s, k) in enumerate(r["members"]):
        tau = TAU[str(k)]["training_labels_fit_days"]
        cd, cc = ld("D", s, k, "cal"), ld("C", s, k, "cal")
        md, mr = mdict(d, i), mdict(r, i)
        for g, cols in GATES.items():
            acc[g].append(apply_gate(fit_gate({"D": cd, "C": cc}, tau, ("D", "C"), cols), {"D": md, "C": mr}, tau, ("D", "C")))
        acc[QB].append(blend(md, mr, gate_params("G3", s, k), tau))
        print(f"member {i + 1}/15 (seed {s}, fold {k}) done", flush=True)
    ens = {g: np.mean(v, 0) for g, v in acc.items()}
    assert np.abs(ens["gate on g and d"] - q15).max() < 1e-9, "STOP: 15-min form not reproduced from its members"

    # ---- 1: scores and contrasts
    fz = frozen_test_predictions(c["sample_ids"])
    z9 = np.load(EVAL / "test_quantiles.npz")
    assert np.array_equal(z9["sample_ids"], c["sample_ids"])
    qref, _ = r3_refs(return_q=True)
    pers = official_persistence(c["ds"].origin_time.astype(object), c["pv"][c["ds"].hist_rows[:, -1]])
    qs = {NAME: q15, V2: z9["method_v2"], "G3''-F": fz["G3''-F"], "G4": fz["G4"], "E6-S": fz["E6-S"], "E4-B-fit": fz["E4-B-fit"],
          "PV-only P15": np.nanmean(members("P15")["pred_q"], 0), QB: ens[QB]} | dict(qref) | {"smart persistence (point)": np.repeat(pers[:, None], 9, 1)}
    sc = {m: score(q) for m, q in qs.items()}
    r9, r3 = stored("r9_method_v2_official_test.json")["scores"], stored("r3_probabilistic_references.json")["references"]
    for m in ("G3''-F", "G4", "E6-S", "E4-B-fit", "PV-only P15"):
        assert abs(sc[m]["crps"].mean() - r9[m]["crps"]) < 1e-9, f"STOP: {m} (R-R9) not reproduced"
    for m in qref:
        assert abs(sc[m]["crps"].mean() - r3[m]["crps"]) < 1e-9, f"STOP: reference {m} (R-R3) not reproduced"
    A = {"n": int(len(y)), "days": int(len(np.unique(day))), "ramp_share": float(ramp.mean()), "scores": {m: summary(v, sunny, ramp) for m, v in sc.items()}, "minus": {}}
    for m in qs:
        if m == NAME:
            continue
        A["minus"][m] = {kk: T(sc[NAME][kk] - sc[m][kk]) for kk in KEYS} | {"relative": float(sc[NAME]["crps"].mean() / sc[m]["crps"].mean() - 1)}
        v = A["minus"][m]
        print(f"15-min form - {m:36s} {A['scores'][m]['crps']:.4f}: {f3(v['crps']['bootstrap'])} p {v['crps']['sign_flip_p']:.5f} rel {v['relative']:+.2%} | ramp region "
              f"{f3(v['tw_ramp']['bootstrap'])} | calm {f3(v['tw_calm']['bootstrap'])} => {'SHOWN' if v['crps']['shown_first_better'] else 'not shown'}", flush=True)

    # ---- 2: gate
    sg = {g: score(ens[g]) for g in GATES}
    Gt = {"scores": {g: summary(sg[g], sunny, ramp) for g in GATES},
          "gate_minus": {g: {kk: T(sg["gate on g and d"][kk] - sg[g][kk]) for kk in KEYS} for g in GATES if g != "gate on g and d"}}
    for g, e in Gt["gate_minus"].items():
        print(f"gate(g,d) - {g:18s} {f3(e['crps']['bootstrap'])} p {e['crps']['sign_flip_p']:.5f}", flush=True)

    # ---- 3: intervals
    accd, yd, org_d, *_ = dev_oof()
    qd = np.mean([accd[s]["gate on g and d"] for s in SEEDS], 0)
    gm = {g: masks[g] for g in GROUPS}
    days = np.unique(day)
    start = {dd: c["org"][day == dd].min() for dd in days}
    first = c["org"].min()
    I = {"W": W, "development_oof": {"n": int(len(yd)), "crps_kw": float(crps_quantile_interp(qd, yd, LEVELS).mean()), "raw_coverage": {}}, "levels": {}}
    ci = {}
    for nm, al in ALPHA.items():
        l_, h_ = base_interval(q15, LEVELS, nm)
        dsc = conformity(*base_interval(qd, LEVELS, nm), yd)
        I["development_oof"]["raw_coverage"][nm] = float(np.mean(dsc <= 0))
        init = past_only_init(org_d, dsc, first, W)[0]
        assert len(init) == W
        rep = replay(c["org"], c["tgt"], l_, h_, y, al, "rolling", W, init)
        ci[nm] = (rep["lo"], rep["hi"])
        daily = rolling_conformal_daily(c["org"], c["tgt"], day, l_, h_, y, al, W, {dd: past_only_init(org_d, dsc, start[dd], W)[0] for dd in days})
        I["levels"][nm] = {"raw": interval_table(l_, h_, y, al, gm), "continuous_window": interval_table(*ci[nm], y, al, gm), "daily_refill": interval_table(*daily, y, al, gm)}
        v = I["levels"][nm]
        print(f"{nm}%: raw {v['raw']['overall']['coverage']:.3f} | continuous {v['continuous_window']['overall']['coverage']:.3f} (ramps {v['continuous_window']['ramp']['coverage']:.3f}, "
              f"cloudy {v['continuous_window']['cloudy']['coverage']:.3f}) width {v['continuous_window']['overall']['width_kw']:.2f} score {v['continuous_window']['overall']['interval_score']:.3f} "
              f"(raw {v['raw']['overall']['interval_score']:.3f}) | daily refill {v['daily_refill']['overall']['coverage']:.3f}", flush=True)
    I["quantile_reliability"] = {g: [float(np.mean(y[masks[g]] <= q15[masks[g], j])) for j in range(len(LEVELS))] for g in ("overall", "cloudy", "sunny")}
    frozen = json.loads(Path("reports/test_results.json").read_text())["scores"]["persistence"]
    I["point"] = point(q15[:, 4], y, masks, frozen)
    print(f"median: RMSE {I['point']['overall']['rmse_kw']:.3f} MAE {I['point']['overall']['mae_kw']:.3f} kW", flush=True)

    # ---- 4: ramp coverage against the selection reference (continuous window)
    nd = nodes7(ci, q15[:, 4])
    E = {"gap": {}, "direction": {}}
    for nm in ALPHA:
        ev, ec = null_counts(nd, ci[nm][0], ci[nm][1], lo, hi, 1.5 * pc, 20261118)
        E["gap"][nm] = gap_bootstrap(y, ci[nm][0], ci[nm][1], lo, hi, ev, ec, day, sunny)
        E["direction"][nm] = direction_coverage(y, ci[nm][0], ci[nm][1], lo, hi)
        g = E["gap"][nm]
        print(f"{nm}%: ramps observed {g['ramp_observed']:.3f}, reference {g['ramp_selection_reference']:.3f}, gap {g['gap']:+.3f} [{g['gap_interval'][0]:+.3f}, {g['gap_interval'][1]:+.3f}] | "
              f"falling {E['direction'][nm]['falling ramps']['coverage']:.3f} rising {E['direction'][nm]['rising ramps']['coverage']:.3f}", flush=True)
    np.savez_compressed(EVAL / "test_intervals_r15.npz", sample_ids=c["sample_ids"], y=y, q=q15, **{f"lo{nm}": ci[nm][0] for nm in ALPHA}, **{f"hi{nm}": ci[nm][1] for nm in ALPHA})

    # ---- 5: warnings and reserve (R-R4 and R-R5 with the 15-min form as the forecaster), operator quantities, cloudy-day reserve
    print("\n=== warnings", flush=True)
    r4.main(forecaster=q15, out_name="r15_ramp_warnings_15min.json", protocol=PROTOCOL)
    print("\n=== reserve", flush=True)
    r5.main(forecaster=q15, name=NAME, out="r15_reserve_15min.json", protocol=PROTOCOL)
    kn, dk, tau = c["k_now"], c["dk"], c["tau"]
    qk = test_k(c, q15)
    lab = dk <= -tau
    p = _cdf((kn - tau)[:, None], qk)[:, 0]
    base = float(lab.mean())
    first_s = np.r_[True, (day[1:] != day[:-1]) | (np.diff(c["tgt"]) > 90_000_000)]
    cl = ~sunny
    ncl = len(np.unique(day[cl]))

    def episodes(w):
        st = w & (first_s | ~np.r_[False, w[:-1]])
        return st, (np.bincount(np.cumsum(st)[w])[1:] if w.any() else np.array([0]))

    Gw = {"base_rate": base, "n_events": int(lab.sum()), "cloudy_days": int(ncl), "sources": {}}
    for nm, w in [("probability >= base rate", p >= base)] + [(f"probability >= {a_}", p >= a_) for a_ in (0.2, 0.3, 0.5)] + [("median", qk[:, 4] - kn <= -tau)]:
        st, ln = episodes(w)
        fa = w & ~lab
        Gw["sources"][nm] = {"share_of_forecasts_under_warning": float(w.mean()), "share_under_warning_cloudy_days": float(w[cl].mean()), "precision": float((w & lab).sum() / max(w.sum(), 1)),
                             "hit_rate": float(w[lab].mean()), "false_alarm_rate": float(fa.sum() / (~lab).sum()), "warning_episodes_per_cloudy_day": float(st[cl].sum() / ncl),
                             "median_episode_length_forecasts": float(np.median(ln)), "runs_of_false_alarms_per_cloudy_day": float(episodes(fa)[0][cl].sum() / ncl)}
    sp05, spf = dressed_q05(c)
    spec = {m: (q[:, 4], q[:, 4] - q[:, 0]) for m, q in ((NAME, q15), (V2, qs[V2]), ("G3''-F", fz["G3''-F"]), ("G4", fz["G4"]), ("E6-S", fz["E6-S"]))}
    spec["smart persistence, dressed (120 min)"] = (spf, spf - sp05)
    spec["constant reserve, official smart persistence"] = (pers, np.zeros(len(y)))
    H, res_c = {}, {}
    for m, (f, b) in spec.items():
        dl = equalise(b[cl], f[cl], y[cl])
        res_c[m] = np.maximum(b[cl] + dl, 0)
        H[m] = {"delta_kw": float(dl), "reliability": float(np.mean(y[cl] >= f[cl] - res_c[m])), "mean_reserve_pct_capacity": float(100 * res_c[m].mean() / 30.1)}
    for m in spec:
        if m != NAME:
            H[m]["method_minus_this"] = day_tests(res_c[NAME] - res_c[m], day[cl], sunny[cl], day_bootstrap)
    put(test={"scores_and_contrasts": A, "gate": Gt, "intervals": I, "ramp_coverage": E, "median_lag": r13["D_median_lag"]["summary"]["M-DC all"],
              "warning_operator_quantities": Gw, "reserve_equal_reliability_within_cloudy_days": H})


# ======================================================================= forward period
def forward():
    from src.baselines import official_persistence
    from src.cal2 import ALPHA, conformity, replay
    from src.calibrate import base_interval
    from src.final_eval import Split, day_bootstrap
    from src.revision_e3 import FOLDS, SEEDS as FSEEDS, W as WF, ensemble_on, fit_anchor, fit_gates, predict, sel_preds
    from src.revision_r7_forward_v2 import apply_mix, block_tests, fit_mix, naive_references

    dev_ = Split("dev", torch.device("cuda"))
    anchor = fit_anchor(dev_)
    day_all = dev_.ds.day.astype(str)
    idx = np.flatnonzero(np.isin(day_all, FOLDS["eval_days"]))
    gates_by_seed = {s: fit_gates(s) for s in FSEEDS}
    q, _ = ensemble_on(dev_, idx, anchor, gates_by_seed)
    q.pop("A2", None)
    acc, sel_acc = [], []
    for s in FSEEDS:
        tau = gates_by_seed[s][1]
        r, d = sel_preds("E6-S", s), sel_preds("E4-B-fit", s)
        p_dc = fit_mix({"D": d, "C": r}, tau, ("D", "C"))
        sel_acc.append(apply_mix(p_dc, {"D": d, "C": r}, tau, ("D", "C")))
        e4, e6 = (predict(dev_, n, s, idx, anchor) for n in ("E4-B-fit", "E6-S"))
        acc.append(apply_mix(p_dc, {"D": e4, "C": e6}, tau, ("D", "C")))
        print(f"forward seed {s} done", flush=True)
    q[NAME] = np.mean(acc, 0)
    z = np.load(EVAL / "forward_quantiles_r13.npz")
    assert np.array_equal(z["idx"], idx) and np.abs(z["mdc_all"] - q[NAME]).max() < 1e-6, "STOP: 15-min form (R-R13 forward) not reproduced"
    q[V2] = z["method_v2"]
    tau = gates_by_seed[FSEEDS[0]][1]
    y = dev_.pv[dev_.ds.target_rows[idx]]
    d_ = day_all[idx]
    sunny = np.array([FOLDS["day_labels"][x] for x in d_])
    pc_t = np.maximum(anchor(dev_.rec_t), 1.0)[dev_.ds.target_rows[idx]]
    kn = (dev_.pv / np.maximum(anchor(dev_.rec_t), 1.0))[dev_.ds.hist_rows[idx, -1]]
    lo, hi = (kn - tau) * pc_t, (kn + tau) * pc_t
    ramp = np.abs(y / pc_t - kn) >= tau
    q |= naive_references(dev_, anchor, idx, FOLDS["folds"][0]["train_days"])
    pers = official_persistence(dev_.ds.origin_time[idx].astype(object), dev_.pv[dev_.ds.hist_rows[idx, -1]])
    q["smart persistence (point)"] = np.repeat(pers[:, None], 9, 1)
    sc = {m: dict(zip(KEYS, ramp_weighted_crps(v, LEVELS, y, lo, hi))) for m, v in q.items()}
    r13, r7 = stored("r13_explanatory_analyses_method_v2.json"), stored("r7_forward_method_v2.json")
    assert abs(sc[NAME]["crps"].mean() - r13["A_forward"]["scores"]["M-DC all"]["crps"]) < 1e-6, "STOP: 15-min form (R-R13 forward) not reproduced"
    for m in ("G3''-F", "E4-B-fit", "E6-S", "G4", "P15"):
        assert abs(sc[m]["crps"].mean() - r7["scores"][m]["crps"]) < 1e-6, f"STOP: {m} (R-R7) not reproduced"

    def tests(dd):
        r = {"bootstrap": day_bootstrap(dd, d_, sunny), "blocks": block_tests(dd, d_, FOLDS["eval_days"])}
        r["shown_first_better"] = bool(r["bootstrap"][2] < 0 and r["blocks"]["sign_flip_p"] < 0.05)
        r["shown_second_better"] = bool(r["bootstrap"][1] > 0 and r["blocks"]["sign_flip_p"] < 0.05)
        return r

    A = {"n": int(len(idx)), "days": int(len(np.unique(d_))), "ramp_share": float(ramp.mean()), "scores": {m: summary(v, sunny, ramp) for m, v in sc.items()}, "minus": {}}
    for m in sc:
        if m == NAME:
            continue
        A["minus"][m] = {kk: tests(sc[NAME][kk] - sc[m][kk]) for kk in KEYS} | {"relative": float(sc[NAME]["crps"].mean() / sc[m]["crps"].mean() - 1)}
        v = A["minus"][m]
        print(f"forward 15-min form - {m:30s} {A['scores'][m]['crps']:.4f}: {f3(v['crps']['bootstrap'])} blocks p {v['crps']['blocks']['sign_flip_p']:.4f} rel {v['relative']:+.2%} | ramp region "
              f"{f3(v['tw_ramp']['bootstrap'])} | calm {f3(v['tw_calm']['bootstrap'])} => {'SHOWN' if v['crps']['shown_first_better'] else 'not shown'}", flush=True)

    rs = sel_preds("E6-S", FSEEDS[0])
    qsel, ys, os_ = np.mean(sel_acc, 0), rs["target"], rs["origin_time_us"]
    order_s = np.argsort(os_, kind="stable")
    org = dev_.ds.origin_time.astype("datetime64[us]").astype(np.int64)[idx]
    tgt = dev_.ds.target_time.astype("datetime64[us]").astype(np.int64)[idx]
    ci, C = {}, {}
    for nm, al in ALPHA.items():
        scs = conformity(*base_interval(qsel, LEVELS, nm), ys)[order_s]
        l_, h_ = base_interval(q[NAME], LEVELS, nm)
        rep = replay(org, tgt, l_, h_, y, al, "rolling", WF["G3''-F"], scs[-WF["G3''-F"]:])
        ci[nm] = (rep["lo"], rep["hi"])
        cov, raw = (y >= ci[nm][0]) & (y <= ci[nm][1]), (y >= l_) & (y <= h_)
        C[nm] = {"overall": float(cov.mean()), "sunny": float(cov[sunny].mean()), "cloudy": float(cov[~sunny].mean()), "raw_overall": float(raw.mean()), "width_kw": float(np.mean(ci[nm][1] - ci[nm][0]))}
    nd = nodes7(ci, q[NAME][:, 4])
    E = {"gap": {}, "direction": {}}
    for nm in ALPHA:
        ev, ec = null_counts(nd, ci[nm][0], ci[nm][1], lo, hi, 1.5 * pc_t, 20261118)
        E["gap"][nm] = gap_bootstrap(y, ci[nm][0], ci[nm][1], lo, hi, ev, ec, d_, sunny)
        E["direction"][nm] = direction_coverage(y, ci[nm][0], ci[nm][1], lo, hi)
        g = E["gap"][nm]
        print(f"forward {nm}%: overall {C[nm]['overall']:.3f} (raw {C[nm]['raw_overall']:.3f}) width {C[nm]['width_kw']:.2f} | ramps observed {g['ramp_observed']:.3f}, reference "
              f"{g['ramp_selection_reference']:.3f}, gap {g['gap']:+.3f} [{g['gap_interval'][0]:+.3f}, {g['gap_interval'][1]:+.3f}]", flush=True)
    put(forward={"scores_and_contrasts": A, "coverage": C, "ramp_coverage": E})


# ======================================================================= external days
def external():
    from src.cal2 import ALPHA, conformity, replay
    from src.calibrate import base_interval
    from src.clear_sky import ClearSkyPV
    from src.final_eval import SEEDS as XSEEDS, blend, day_bootstrap, gate_params
    from src.metrics import crps_quantile_interp
    from src.revision_posthoc_dump import run_member
    from src.revision_posthoc_test import sub
    from src.revision_r7_forward_v2 import apply_mix, fit_mix
    from src.revision_r10_external_full import US_MIN, build_split

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    split, t, y, info = build_split(torch.device("cuda"))
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    tau_all = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    pv_now = split.pv[split.ds.hist_rows[:, -1]]
    names = (NAME, QB, "G4", "calm expert", "direct expert")
    acc = {m: [] for m in names}
    for s in XSEEDS:
        for k in range(5):
            fold, tau = {**folds_doc["folds"][k]}, tau_all[str(k)]["training_labels_fit_days"]
            rd, rc = (run_member(split, e, s, fold, tau) for e in ("E4-B-fit", "E6-S"))
            md, mr = ({"pred_q": r["pred_q"], "pclear_target_kw": r["pclear_target_kw"], "pclear_now_kw": r["pclear_now_kw"], "pv_now_kw": pv_now, "quantile_levels": LEVELS} for r in (rd, rc))
            acc[QB].append(blend(md, mr, gate_params("G3", s, k), tau))
            acc["G4"].append(blend(md, mr, gate_params("G4", s, k), tau))
            acc["calm expert"].append(mr["pred_q"]); acc["direct expert"].append(md["pred_q"])
            cd, cc = ld("D", s, k, "cal"), ld("C", s, k, "cal")
            acc[NAME].append(apply_mix(fit_mix({"D": cd, "C": cc}, tau, ("D", "C")), {"D": md, "C": mr}, tau, ("D", "C")))
            print(f"member seed {s} fold {k} done", flush=True)
    q = {m: np.mean(v, 0) for m, v in acc.items()}
    zx = np.load(EVAL / "external_quantiles.npz")
    assert np.array_equal(zx["y"], y)
    q[V2] = zx["method_v2"]
    crps = {m: crps_quantile_interp(v, y, LEVELS) for m, v in q.items()}
    r10 = stored("r10_external_days_full_method.json")
    for m, k in ((NAME, "method v2, fallback only"), (QB, "frozen rule, fallback only"), ("G4", "G4"), ("calm expert", "calm expert"), ("direct expert", "direct expert"), (V2, "method v2, routed")):
        assert abs(crps[m].mean() - r10["scores"][k]["crps_exact"]) < 1e-4, f"STOP: {m} (R-R10) not reproduced: {crps[m].mean()} against {r10['scores'][k]['crps_exact']}"
    day = t.astype("datetime64[D]").astype(str)
    zero = np.zeros(len(y), bool)
    rdj = json.loads(Path("reports/test_ramp_definition_frozen.json").read_text())
    anchor = ClearSkyPV(rdj["anchor_coef"])
    ramp = np.abs(y / np.maximum(anchor(t + np.timedelta64(15, "m")), 1.0) - pv_now / np.maximum(anchor(t), 1.0)) >= rdj["tau_test"]
    X = {"n": int(len(y)), "days": sorted(set(day)), "n_ramp": int(ramp.sum()), "published": r10["published"],
         "scores": {m: {"crps_exact": float(crps[m].mean()), "ramp": float(crps[m][ramp].mean()), "non_ramp": float(crps[m][~ramp].mean())} for m in q},
         "sample_based_r10": {kk: r10["scores"]["method v2, fallback only"][kk] for kk in ("standard_m10", "standard_m30", "standard_m100", "fair")}, "minus": {}, "coverage": {}}
    for m in q:
        if m != NAME:
            X["minus"][m] = day_bootstrap(crps[NAME] - crps[m], day, zero)
            print(f"external 15-min form - {m:30s} {X['scores'][m]['crps_exact']:.3f}: {f3(X['minus'][m])} (five days; descriptive)", flush=True)
    accd, yd, org_d, *_ = dev_oof()
    qd = np.mean([accd[s]["gate on g and d"] for s in SEEDS], 0)
    org = t.astype("datetime64[us]").astype(np.int64)
    tgt = org + 15 * US_MIN
    order = np.argsort(org_d, kind="stable")
    for nm, al in ALPHA.items():
        init = conformity(*base_interval(qd, LEVELS, nm), yd)[order][-W:]
        lo, hi = base_interval(q[NAME], LEVELS, nm)
        out = {"raw": (lo, hi)}
        for meth, par in (("continuous window", ("rolling", W)), ("ACI", ("aci", 0.005))):
            rep = replay(org, tgt, lo, hi, y, al, par[0], par[1], init)
            out[meth] = (rep["lo"], rep["hi"])
        X["coverage"][nm] = {kk: {"coverage": float(np.mean((y >= l) & (y <= h))), "ramp_coverage": float(np.mean((y[ramp] >= l[ramp]) & (y[ramp] <= h[ramp]))), "width_kw": float(np.mean(h - l))}
                             for kk, (l, h) in out.items()}
        print(f"external {nm}%: " + " | ".join(f"{kk} cov {v['coverage']:.3f} (ramps {v['ramp_coverage']:.3f}) width {v['width_kw']:.2f}" for kk, v in X["coverage"][nm].items()), flush=True)
    put(external=X)


if __name__ == "__main__":
    {"dev": dev, "test": test, "forward": forward, "external": external}[sys.argv[1]]()
