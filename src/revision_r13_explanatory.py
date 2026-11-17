"""Revision R-R13: explanatory and verification analyses of method v2 (protocol
revision/protocols/R-R13_explanatory_analyses_method_v2.md). No training; nothing here selects or changes the method.

Run: python -m src.revision_r13_explanatory dev      (parts B and C on development out-of-fold forecasts; CPU)
     python -m src.revision_r13_explanatory test     (parts A-H on the official test set and the external days; CPU)
     python -m src.revision_r13_explanatory forward  (parts A and E in the forward period; GPU inference)
Each invocation writes its keys into revision/results/r13_explanatory_analyses_method_v2.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.e8_features import FEAT
from src.gate7b import grid_quantiles
from src.ramp_scoring import draw_outcomes, ramp_weighted_crps
from src.revision_common import LEVELS, RESULTS, save
from src.revision_m1_method_v2 import MixN, ROOT, TAU, features, fit, mixture_quantiles, select_and_fit, weights
from src.revision_r2_day_level_inference import sign_flip, t_interval

OUT = "r13_explanatory_analyses_method_v2.json"
PROTOCOL = "revision/protocols/R-R13_explanatory_analyses_method_v2.md"
EVAL = Path("D:/skippd_hires/R_M3/eval")
KEYS = ("crps", "tw_ramp", "tw_calm")
SEEDS = (42, 123, 456)
PARTNERS = {"D0 (2D, 15 min)": Path("experiments/E8/v1/D0/csi"), "D2 (2D, 60 min)": Path("experiments/E8/v1/D2/csi"),
            "A2-H15 (3D, 15 min)": Path("experiments/R_E4/A2_H15/csi"), "A2 (3D, 60 min)": ROOT["R"]}
DUMP = {"D0 (2D, 15 min)": "D0", "D2 (2D, 60 min)": "D2", "A2-H15 (3D, 15 min)": "A2-H15", "A2 (3D, 60 min)": "A2"}
EFFECTS = (("history within 3D (A2 - A2-H15)", "A2 (3D, 60 min)", "A2-H15 (3D, 15 min)"), ("history within 2D (D2 - D0)", "D2 (2D, 60 min)", "D0 (2D, 15 min)"),
           ("encoder at 60 min (A2 - D2)", "A2 (3D, 60 min)", "D2 (2D, 60 min)"), ("encoder at 15 min (A2-H15 - D0)", "A2-H15 (3D, 15 min)", "D0 (2D, 15 min)"))
GATES = {"constant weight": None, "gate on g only": (0,), "gate on d only": (1,), "gate on g and d": (0, 1)}
P7 = np.array([.05, .10, .25, .50, .75, .90, .95])
f3 = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"


def put(**parts):
    p = RESULTS / OUT
    res = json.loads(p.read_text()) if p.exists() else {"protocol": PROTOCOL}
    res.update(parts)
    print("saved", save(OUT, res), "keys:", ", ".join(parts))


# ----------------------------------------------------------------------------------------------- gates and mixtures
def fit_gate(pd_, tau, comps, cols):
    """Gate of the mixture of `comps` (last: calm expert) on the feature columns `cols` (0: logit g, 1: log d); None: constant weight."""
    f = MixN([grid_quantiles(pd_[e]["pred_q"], LEVELS) for e in comps], pd_["C"]["target"])
    if cols is None:
        b0, _ = fit(np.zeros((len(pd_["C"]["target"]), 0)), f, 0.0, len(comps), gated=False)
        return {"b": b0, "cols": None}
    order = np.argsort(pd_["C"]["origin_time_us"], kind="stable")
    return select_and_fit(features(pd_, tau, comps)[:, list(cols)], f, order, len(comps)) | {"cols": list(cols)}


def apply_gate(p, pd_, tau, comps):
    n = len(pd_["C"]["pred_q"])
    if p["cols"] is None:
        w = weights(torch.zeros(n, 0, dtype=torch.float64), p["b"].expand(n, -1), None).numpy()
    else:
        z = features(pd_, tau, comps)[:, p["cols"]]
        w = weights(torch.as_tensor((z - p["mu"]) / p["sd"]), p["b"].expand(n, -1), p["B"]).numpy()
    return mixture_quantiles([pd_[e]["pred_q"] for e in comps], w)


def day_tests(d, day, sunny, boot):
    """Day bootstrap, exact day-level sign-flip p and t interval of a paired per-forecast difference."""
    days, inv = np.unique(day, return_inverse=True)
    sday = np.zeros(len(days), bool)
    sday[inv] = sunny
    S, n = np.bincount(inv, weights=d), np.bincount(inv).astype(float)
    r = {"bootstrap": boot(d, day, sunny), "sign_flip_p": sign_flip(S, n)[0], "t_interval_day_means": t_interval(S / n, sday)[0],
         "days_first_better": int((S < 0).sum()), "days": int(len(days))}
    r["shown_first_better"] = bool(r["bootstrap"][2] < 0 and r["sign_flip_p"] < 0.05)
    r["shown_second_better"] = bool(r["bootstrap"][1] > 0 and r["sign_flip_p"] < 0.05)
    return r


def summary(v, sunny, ramp=None):
    r = {k: float(v[k].mean()) for k in KEYS} | {"sunny": float(v["crps"][sunny].mean()), "cloudy": float(v["crps"][~sunny].mean())}
    if ramp is not None:
        r["ramp_outcomes"] = float(v["crps"][ramp].mean())
    return r


# ----------------------------------------------------------------------------------------------- coverage helpers
def null_counts(nodes, lower, upper, band_lo, band_hi, cap, seed, m=2000, chunk=100):
    """Per-forecast counts over m outcomes drawn from the forecast itself: ramp outcomes, and ramp outcomes inside [lower, upper]
    (the selection reference of R-Z1, kept per forecast so that it can be resampled with the observed coverage)."""
    rng = np.random.default_rng(seed)
    ev, ec = np.zeros(len(lower)), np.zeros(len(lower))
    for _ in range(m // chunk):
        ys = draw_outcomes(nodes, P7, rng.random((len(lower), chunk)), cap, "exp")
        e = (ys <= band_lo[:, None]) | (ys >= band_hi[:, None])
        ev += e.sum(1)
        ec += (e & (ys >= lower[:, None]) & (ys <= upper[:, None])).sum(1)
    return ev, ec


def gap_bootstrap(y, lower, upper, band_lo, band_hi, ev, ec, day, sunny, n_boot=10_000, seed=20261022):
    """Observed ramp coverage minus selection reference, with a stratified day bootstrap of both together."""
    ramp = (y <= band_lo) | (y >= band_hi)
    cov = ramp & (y >= lower) & (y <= upper)
    days, inv = np.unique(day, return_inverse=True)
    sday = np.zeros(len(days), bool)
    sday[inv] = sunny
    T = np.stack([np.bincount(inv, weights=x.astype(float), minlength=len(days)) for x in (ramp, cov, ev, ec)], 1)
    gap = lambda t: t[1] / t[0] - t[3] / t[2]
    rng = np.random.default_rng(seed)
    groups = [g for g in (np.flatnonzero(sday), np.flatnonzero(~sday)) if len(g)]
    boots = np.array([gap(T[np.concatenate([rng.choice(g, len(g)) for g in groups])].sum(0)) for _ in range(n_boot)])
    t = T.sum(0)
    return {"ramp_observed": float(t[1] / t[0]), "ramp_selection_reference": float(t[3] / t[2]), "gap": float(gap(t)),
            "gap_interval": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))], "n_ramp": int(t[0]),
            "deficit_beyond_selection": bool(np.quantile(boots, 0.975) < 0)}


def direction_coverage(y, lower, upper, band_lo, band_hi):
    out = {}
    for nm, m in (("falling ramps", y <= band_lo), ("rising ramps", y >= band_hi), ("non-ramps", (y > band_lo) & (y < band_hi))):
        miss = np.maximum(lower[m] - y[m], y[m] - upper[m])
        out[nm] = {"n": int(m.sum()), "coverage": float(np.mean(miss <= 0)), "below": float(np.mean(y[m] < lower[m])), "above": float(np.mean(y[m] > upper[m])),
                   "mean_miss_distance_kw": float(miss[miss > 0].mean()) if (miss > 0).any() else 0.0}
    return out


def nodes7(ci, med):
    return np.sort(np.stack([ci["90"][0], ci["80"][0], ci["50"][0], med, ci["50"][1], ci["80"][1], ci["90"][1]], 1), 1)


# ======================================================================= development: parts B and C
def dev():
    from src.e8_features import verify_frozen
    from src.revision_common import day_bootstrap
    from src.revision_r1_gate_proper_score import band
    from src.revision_x2_linear_pool import load

    verify_frozen()
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = win["sample_ids"][win["eligible_common"]]
    m1 = json.loads((RESULTS / "m1_method_v2_development.json").read_text())["per_seed"]
    B = {"per_seed": {}, "effects": {e[0]: {} for e in EFFECTS}}
    C = {"per_seed": {}, "gate_minus": {g: {} for g in GATES if g != "gate on g and d"}, "share_of_gate_gain_recovered": {}}
    for s in SEEDS:
        qs = {m: [] for m in list(PARTNERS) + list(GATES)}
        ys, los, his, days, sun = [], [], [], [], []
        for k in range(5):
            tau = TAU[str(k)]["training_labels_fit_days"]
            cc, vc = load(ROOT["C"], s, k, "cal", common), load(ROOT["C"], s, k, "val", common)
            for name, root in PARTNERS.items():
                cx, vx = load(root, s, k, "cal", common), load(root, s, k, "val", common)
                assert np.array_equal(cx["sample_ids"], cc["sample_ids"]) and np.array_equal(vx["sample_ids"], vc["sample_ids"]), f"STOP: samples differ ({name})"
                p = fit_gate({"X": cx, "C": cc}, tau, ("X", "C"), (0, 1))
                qs[name].append(apply_gate(p, {"X": vx, "C": vc}, tau, ("X", "C")))
                if name.startswith("A2 ("):
                    for g, cols in GATES.items():
                        qs[g].append(qs[name][-1] if cols == (0, 1) else apply_gate(fit_gate({"X": cx, "C": cc}, tau, ("X", "C"), cols), {"X": vx, "C": vc}, tau, ("X", "C")))
            lo, hi = band(vc, tau)
            ys.append(vc["target"]); los.append(lo); his.append(hi); days.append(vc["day"]); sun.append(vc["clear_day"])
            print(f"development seed {s} fold {k} done", flush=True)
        cat = np.concatenate
        y, lo, hi, day, sunny = cat(ys), cat(los), cat(his), cat(days), cat(sun).astype(bool)
        sc = {m: dict(zip(KEYS, ramp_weighted_crps(cat(v), LEVELS, y, lo, hi))) for m, v in qs.items()}
        assert abs(sc["A2 (3D, 60 min)"]["crps"].mean() - m1[str(s)]["M-RC"]["crps"]) < 1e-9, "STOP: M-RC (R-M1) not reproduced"
        assert abs(sc["constant weight"]["crps"].mean() - m1[str(s)]["M-RC const"]["crps"]) < 1e-9, "STOP: M-RC const (R-M1) not reproduced"
        B["per_seed"][str(s)] = {m: summary(sc[m], sunny) for m in PARTNERS}
        C["per_seed"][str(s)] = {m: summary(sc[m], sunny) for m in GATES}
        for name, a, b in EFFECTS:
            B["effects"][name][str(s)] = {kk: day_bootstrap(sc[a][kk] - sc[b][kk], day, sunny) for kk in KEYS}
        for g in C["gate_minus"]:
            C["gate_minus"][g][str(s)] = {kk: day_bootstrap(sc["gate on g and d"][kk] - sc[g][kk], day, sunny) for kk in KEYS}
        cm = lambda m: sc[m]["crps"].mean()
        gain = cm("constant weight") - cm("gate on g and d")
        C["share_of_gate_gain_recovered"][str(s)] = {g: float((cm("constant weight") - cm(g)) / gain) for g in ("gate on g only", "gate on d only")}
        print(f"seed {s}: " + "  ".join(f"{m} {cm(m):.4f}" for m in sc), flush=True)
    mean = lambda d, m, kk: float(np.mean([d["per_seed"][str(s)][m][kk] for s in SEEDS]))
    B["seed_mean"] = {m: {kk: mean(B, m, kk) for kk in KEYS} for m in PARTNERS}
    C["seed_mean"] = {m: {kk: mean(C, m, kk) for kk in KEYS} for m in GATES}
    neg = lambda e, kk: int(sum(v[kk][2] < 0 for v in e.values()))
    B["seeds_with_interval_below_zero"] = {name: {kk: neg(e, kk) for kk in KEYS} for name, e in B["effects"].items()}
    C["seeds_with_interval_below_zero"] = {g: {kk: neg(e, kk) for kk in KEYS} for g, e in C["gate_minus"].items()}
    h3, h2 = B["seeds_with_interval_below_zero"]["history within 3D (A2 - A2-H15)"]["crps"], B["seeds_with_interval_below_zero"]["history within 2D (D2 - D0)"]["crps"]
    e60, e15 = B["seeds_with_interval_below_zero"]["encoder at 60 min (A2 - D2)"]["crps"], B["seeds_with_interval_below_zero"]["encoder at 15 min (A2-H15 - D0)"]["crps"]
    B["reading"] = {"gain credited to the 60-min history (>= 2/3 seeds within both encoders)": bool(h3 >= 2 and h2 >= 2),
                    "gain credited to the 3D encoder (>= 2/3 seeds at both history lengths)": bool(e60 >= 2 and e15 >= 2)}
    C["reading"] = {"gate beats constant weight (3/3 seeds)": C["seeds_with_interval_below_zero"]["constant weight"]["crps"] == 3,
                    "g-only gate not shown worse (seeds with interval below zero)": C["seeds_with_interval_below_zero"]["gate on g only"]["crps"],
                    "d-only gate not shown worse (seeds with interval below zero)": C["seeds_with_interval_below_zero"]["gate on d only"]["crps"]}
    for name, e in B["effects"].items():
        print(f"B {name:34s} " + " | ".join(f"seed {s}: {f3(v['crps'])} ramp {f3(v['tw_ramp'])} calm {f3(v['tw_calm'])}" for s, v in e.items()))
    for g, e in C["gate_minus"].items():
        print(f"C gate(g,d) - {g:18s} " + " | ".join(f"seed {s}: {f3(v['crps'])}" for s, v in e.items()))
    print("B seed mean:", {m: round(v["crps"], 4) for m, v in B["seed_mean"].items()}, B["reading"])
    print("C seed mean:", {m: round(v["crps"], 4) for m, v in C["seed_mean"].items()}, C["share_of_gate_gain_recovered"], C["reading"])
    put(B_partner_development=B, C_gate_development=C)


# ======================================================================= official test set and external days
def test():
    from src.baselines import official_persistence
    from src.cal2 import ALPHA, conformity, interval_score, replay
    from src.calibrate import base_interval
    from src.lag_analysis import per_day
    from src.metrics import crps_quantile_interp
    from src.revision_c07 import interval_table, past_only_init
    from src.revision_c21 import crps_ens, sample
    from src.revision_common import day_bootstrap, frozen_test_predictions, test_context
    from src.revision_posthoc_test import mdict, members, sub, test_k
    from src.revision_r3_probabilistic_references import main as r3_refs
    from src.revision_r5_reserve import dressed_q05, equalise
    from src.revision_r11_test_reliability import W, dev_oof_v2
    from src.revision_x2_linear_pool import _cdf
    from src.revision_z_common import calibrated_intervals, ramp_region

    stored = lambda n: json.loads((RESULTS / n).read_text())
    c = test_context()
    h60, y, day, sunny, ramp = c["h60"], c["y"], c["day"], c["sunny"], c["ramp"]
    pc, lo, hi = ramp_region(c)
    T = lambda d, m=None: day_tests(d if m is None else d[m], day if m is None else day[m], sunny if m is None else sunny[m], day_bootstrap)
    score = lambda q, m=None: dict(zip(KEYS, ramp_weighted_crps(q if m is None else q[m], LEVELS, y if m is None else y[m], lo if m is None else lo[m], hi if m is None else hi[m])))
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = win["sample_ids"][win["eligible_common"]]
    cal = lambda root, s, k: dict(np.load(root / f"seed{s}" / f"fold_{k}" / "preds_cal.npz"))
    d, r, a = members("E4-B-fit"), members("E6-S"), members("A2")
    dumps = {n: members(DUMP[n]) for n in PARTNERS}
    acc = {m: [] for m in ["v2", "M-DC all"] + list(PARTNERS) + list(GATES)}
    for i, (s, k) in enumerate(r["members"]):
        tau = TAU[str(k)]["training_labels_fit_days"]
        cd, cc = cal(ROOT["D"], s, k), cal(ROOT["C"], s, k)
        assert np.array_equal(cd["sample_ids"], cc["sample_ids"])
        cc2 = sub(cc, np.isin(cc["sample_ids"], common))
        md, mr = mdict(d, i), mdict(r, i)
        mr60 = sub(mr, h60)
        q_dc = apply_gate(fit_gate({"D": cd, "C": cc}, tau, ("D", "C"), (0, 1)), {"D": md, "C": mr}, tau, ("D", "C"))
        acc["M-DC all"].append(q_dc)
        for name, root in PARTNERS.items():
            assert tuple(dumps[name]["members"][i]) == (s, k), f"STOP: member order differs ({name})"
            cx = cal(root, s, k)
            cx = sub(cx, np.isin(cx["sample_ids"], common))
            assert np.array_equal(cx["sample_ids"], cc2["sample_ids"]), f"STOP: calibration samples differ ({name})"
            mx = sub(mdict(dumps[name], i), h60)
            acc[name].append(apply_gate(fit_gate({"X": cx, "C": cc2}, tau, ("X", "C"), (0, 1)), {"X": mx, "C": mr60}, tau, ("X", "C")))
            if name.startswith("A2 ("):
                for g, cols in GATES.items():
                    acc[g].append(acc[name][-1] if cols == (0, 1) else apply_gate(fit_gate({"X": cx, "C": cc2}, tau, ("X", "C"), cols), {"X": mx, "C": mr60}, tau, ("X", "C")))
        q2 = q_dc.copy()
        q2[h60] = acc["A2 (3D, 60 min)"][-1]
        acc["v2"].append(q2)
        print(f"member {i + 1}/15 (seed {s}, fold {k}) done", flush=True)
    ens = {m: np.mean(v, 0) for m, v in acc.items()}
    fz = frozen_test_predictions(c["sample_ids"])
    z = np.load(EVAL / "test_quantiles.npz")
    assert np.array_equal(z["sample_ids"], c["sample_ids"]) and np.abs(ens["v2"] - z["method_v2"]).max() < 1e-9, "STOP: method v2 quantiles (R-R9) not reproduced"
    v2 = z["method_v2"]
    r9 = stored("r9_method_v2_official_test.json")
    assert abs(score(v2)["crps"].mean() - r9["scores"]["method v2"]["crps"]) < 1e-9, "STOP: method v2 (R-R9) not reproduced"

    # ---- A: the method's own 15-min form
    qa = {"method v2": v2, "M-DC all": ens["M-DC all"], "G4": fz["G4"], "G3''-F": fz["G3''-F"]}
    sa, sa60 = {m: score(q) for m, q in qa.items()}, {m: score(q, h60) for m, q in qa.items()}
    A = {"n": int(len(y)), "n_with_60min_history": int(h60.sum()), "scores": {m: summary(v, sunny, ramp) for m, v in sa.items()},
         "scores_with_60min_history": {m: summary(v, sunny[h60], ramp[h60]) for m, v in sa60.items()}, "contrasts": {}}
    for a_, b_ in (("method v2", "M-DC all"), ("M-DC all", "G4"), ("M-DC all", "G3''-F"), ("method v2", "G4")):
        A["contrasts"][f"{a_} - {b_}"] = {kk: T(sa[a_][kk] - sa[b_][kk]) for kk in KEYS} | {"relative": float(sa[a_]["crps"].mean() / sa[b_]["crps"].mean() - 1),
                                                                                         "with_60min_history": {kk: day_tests(sa60[a_][kk] - sa60[b_][kk], day[h60], sunny[h60], day_bootstrap) for kk in KEYS}}
        cc_ = A["contrasts"][f"{a_} - {b_}"]
        print(f"A {a_} - {b_}: CRPS {f3(cc_['crps']['bootstrap'])} p {cc_['crps']['sign_flip_p']:.4f} ({cc_['crps']['days_first_better']}/20 days) rel {cc_['relative']:+.2%} | "
              f"ramp region {f3(cc_['tw_ramp']['bootstrap'])} | calm {f3(cc_['tw_calm']['bootstrap'])} | 60-min forecasts {f3(cc_['with_60min_history']['crps']['bootstrap'])} "
              f"p {cc_['with_60min_history']['crps']['sign_flip_p']:.4f}", flush=True)
    k_ = A["contrasts"]["method v2 - M-DC all"]["crps"]
    A["reading"] = {"the 60-min expert improves the method on the official test set": k_["shown_first_better"], "M-DC all better than method v2": k_["shown_second_better"]}
    np.savez_compressed(EVAL / "test_quantiles_r13.npz", sample_ids=c["sample_ids"], y=y, mdc_all=ens["M-DC all"])

    # ---- B and C on the forecasts with a 60-min history
    y6, d6, s6 = y[h60], day[h60], sunny[h60]
    s60 = {m: dict(zip(KEYS, ramp_weighted_crps(ens[m], LEVELS, y6, lo[h60], hi[h60]))) for m in list(PARTNERS) + list(GATES)}
    Bt = {"n": int(h60.sum()), "scores": {m: summary(s60[m], s6, ramp[h60]) for m in PARTNERS},
          "effects": {name: {kk: day_tests(s60[a_][kk] - s60[b_][kk], d6, s6, day_bootstrap) for kk in KEYS} for name, a_, b_ in EFFECTS}}
    Ct = {"n": int(h60.sum()), "scores": {m: summary(s60[m], s6, ramp[h60]) for m in GATES},
          "gate_minus": {g: {kk: day_tests(s60["gate on g and d"][kk] - s60[g][kk], d6, s6, day_bootstrap) for kk in KEYS} for g in GATES if g != "gate on g and d"}}
    cm = lambda m: s60[m]["crps"].mean()
    Ct["share_of_gate_gain_recovered"] = {g: float((cm("constant weight") - cm(g)) / (cm("constant weight") - cm("gate on g and d"))) for g in ("gate on g only", "gate on d only")}
    for name, e in Bt["effects"].items():
        print(f"B test {name:34s} {f3(e['crps']['bootstrap'])} p {e['crps']['sign_flip_p']:.4f} | ramp region {f3(e['tw_ramp']['bootstrap'])} | calm {f3(e['tw_calm']['bootstrap'])}")
    for g, e in Ct["gate_minus"].items():
        print(f"C test gate(g,d) - {g:18s} {f3(e['crps']['bootstrap'])} p {e['crps']['sign_flip_p']:.4f}")
    print("B test scores:", {m: round(cm(m), 4) for m in PARTNERS}, "| C test scores:", {m: round(cm(m), 4) for m in GATES}, Ct["share_of_gate_gain_recovered"], flush=True)

    # ---- D: lag of the median on the ten cloudy days
    pers = official_persistence(c["ds"].origin_time.astype(object), c["pv"][c["ds"].hist_rows[:, -1]])
    series = {"method v2": v2[:, 4], "M-DC all": ens["M-DC all"][:, 4], "G3''-F": fz["G3''-F"][:, 4], "E4-B-fit": fz["E4-B-fit"][:, 4], "E6-S": fz["E6-S"][:, 4],
              "smart persistence": pers}
    t = c["ds"].target_time.astype("datetime64[m]")
    cl = ~sunny
    pdy = per_day(day[cl], t[cl], {kk: v[cl] for kk, v in series.items()}, y[cl])
    D = {"days": int(len(pdy)), "per_day_best_lag_min": {m: {dd: pdy[dd][m][0] for dd in pdy} for m in series},
         "summary": {m: {"median_best_lag_min": float(np.median([pdy[dd][m][0] for dd in pdy])), "mean_corr_lag0": float(np.mean([pdy[dd][m][1] for dd in pdy])),
                         "mean_corr_lag15": float(np.mean([pdy[dd][m][2] for dd in pdy])), "days_with_best_lag_at_least_14_min": int(sum(pdy[dd][m][0] >= 14 for dd in pdy))}
                     for m in series}}
    old = json.loads(Path("reports/lag_analysis.json").read_text())["SKIPPD test (10 cloudy days)"]["summary"]
    for m in ("G3''-F", "E4-B-fit", "E6-S", "smart persistence"):
        assert abs(D["summary"][m]["mean_corr_lag15"] - old[m]["mean_corr_lag15"]) < 1e-9 and D["summary"][m]["median_best_lag_min"] == old[m]["median_best_lag_min"], f"STOP: lag of {m} not reproduced"
    D["reading"] = {"the median of the method lags by 15 min (median best lag >= 14 min)": bool(D["summary"]["method v2"]["median_best_lag_min"] >= 14)}
    print("D lag:", {m: (v["median_best_lag_min"], round(v["mean_corr_lag0"], 3), round(v["mean_corr_lag15"], 3), v["days_with_best_lag_at_least_14_min"]) for m, v in D["summary"].items()}, flush=True)

    # ---- E: interval coverage
    iv = np.load(EVAL / "test_intervals_v2.npz")
    assert np.array_equal(iv["sample_ids"], c["sample_ids"]) and np.abs(iv["q"] - v2).max() < 1e-12
    ci = {nm: (iv[f"lo{nm}"], iv[f"hi{nm}"]) for nm in ALPHA}
    r11 = stored("r11_test_reliability_point_error.json")
    nd = nodes7(ci, v2[:, 4])
    E = {"gap": {}, "direction": {}, "continuous_window_across_test_days": {}, "daily_initialisation_r11": {}}
    for nm in ALPHA:
        ev, ec = null_counts(nd, ci[nm][0], ci[nm][1], lo, hi, 1.5 * pc, 20261117)
        E["gap"][nm] = gap_bootstrap(y, ci[nm][0], ci[nm][1], lo, hi, ev, ec, day, sunny)
        ref = r11["ramp_coverage_selection_reference"][nm]
        assert abs(E["gap"][nm]["ramp_selection_reference"] - ref["ramp_selection_reference"]) < 1e-12 and abs(E["gap"][nm]["ramp_observed"] - ref["ramp_observed"]) < 1e-12, "STOP: R-R11 selection reference not reproduced"
        E["direction"][nm] = direction_coverage(y, ci[nm][0], ci[nm][1], lo, hi)
        g = E["gap"][nm]
        print(f"E {nm}%: ramps observed {g['ramp_observed']:.3f}, reference {g['ramp_selection_reference']:.3f}, gap {g['gap']:+.3f} [{g['gap_interval'][0]:+.3f}, {g['gap_interval'][1]:+.3f}] | "
              f"falling {E['direction'][nm]['falling ramps']['coverage']:.3f} rising {E['direction'][nm]['rising ramps']['coverage']:.3f}", flush=True)
    qd, yd, org_d = dev_oof_v2()
    gm = {g: c["masks"][g] for g in ("overall", "sunny", "cloudy", "ramp", "non-ramp")}
    first = c["org"].min()
    for nm, al in ALPHA.items():
        l_, h_ = base_interval(v2, LEVELS, nm)
        init = past_only_init(org_d, conformity(*base_interval(qd, LEVELS, nm), yd), first, W)[0]
        assert len(init) == W
        rep = replay(c["org"], c["tgt"], l_, h_, y, al, "rolling", W, init)
        E["continuous_window_across_test_days"][nm] = interval_table(rep["lo"], rep["hi"], y, al, gm)
        E["daily_initialisation_r11"][nm] = interval_table(*ci[nm], y, al, gm)
        assert abs(E["daily_initialisation_r11"][nm]["overall"]["coverage"] - r11["intervals"]["method v2"][nm]["calibrated"]["overall"]["coverage"]) < 1e-12
        a_, b_ = E["continuous_window_across_test_days"][nm], E["daily_initialisation_r11"][nm]
        print(f"E {nm}% continuous across test days: coverage {a_['overall']['coverage']:.3f} (ramps {a_['ramp']['coverage']:.3f}, cloudy {a_['cloudy']['coverage']:.3f}) width {a_['overall']['width_kw']:.2f} | "
              f"daily initialisation: {b_['overall']['coverage']:.3f} (ramps {b_['ramp']['coverage']:.3f}) width {b_['overall']['width_kw']:.2f}", flush=True)

    # ---- F: secondary comparisons
    qref, _ = r3_refs(return_q=True)
    r3 = stored("r3_probabilistic_references.json")["references"]
    cv2 = crps_quantile_interp(v2, y, LEVELS)
    F = {"v2_minus_reference": {}, "sign_flip_for_secondary": {}, "external_fair_estimator": {}}
    for kk, q in qref.items():
        cr = crps_quantile_interp(q, y, LEVELS)
        assert abs(cr.mean() - r3[kk]["crps"]) < 1e-9, f"STOP: reference {kk} (R-R3) not reproduced"
        F["v2_minus_reference"][kk] = T(cv2 - cr) | {"reference_crps": float(cr.mean()), "skill": float(1 - cv2.mean() / cr.mean())}
    F["v2_minus_reference"]["smart persistence (point)"] = T(cv2 - np.abs(pers - y)) | {"reference_crps": float(np.abs(pers - y).mean()), "skill": float(1 - cv2.mean() / np.abs(pers - y).mean())}
    for kk, v in F["v2_minus_reference"].items():
        print(f"F v2 - {kk:28s} {f3(v['bootstrap'])} p {v['sign_flip_p']:.5f} skill {v['skill']:.1%}")
    for m in ("G3''-F", "G4"):
        cim = calibrated_intervals(c, fz[m], m)
        F["sign_flip_for_secondary"][f"abs error of the median, v2 - {m}"] = T(np.abs(v2[:, 4] - y) - np.abs(fz[m][:, 4] - y))
        F["sign_flip_for_secondary"][f"90% interval score, v2 - {m}"] = T(interval_score(*ci["90"], y, 0.1) - interval_score(*cim["90"], y, 0.1))
    for kk, v in F["sign_flip_for_secondary"].items():
        print(f"F {kk}: {f3(v['bootstrap'])} p {v['sign_flip_p']:.4f}")
    ex = np.load(EVAL / "external_quantiles.npz")
    rng = np.random.default_rng(20261117)
    F["external_fair_estimator"] = {"exact": float(crps_quantile_interp(ex["method_v2"], ex["y"], LEVELS).mean())}
    for mm in (10, 30, 100):
        dr = [sample(ex["method_v2"], mm, rng) for _ in range(20)]
        F["external_fair_estimator"][f"m={mm}"] = {"standard": float(np.mean([crps_ens(x, ex["y"], False).mean() for x in dr])), "fair": float(np.mean([crps_ens(x, ex["y"], True).mean() for x in dr]))}
    print("F external:", F["external_fair_estimator"], flush=True)

    # ---- G: operator quantities of the falling-ramp warnings
    kn, dk, tau = c["k_now"], c["dk"], c["tau"]
    qk = test_k(c, v2)
    lab = dk <= -tau
    p = _cdf((kn - tau)[:, None], qk)[:, 0]
    base = float(lab.mean())
    first_s = np.r_[True, (day[1:] != day[:-1]) | (np.diff(c["tgt"]) > 90_000_000)]
    ncl = len(np.unique(day[~sunny]))

    def episodes(w):
        start = w & (first_s | ~np.r_[False, w[:-1]])
        ids = np.cumsum(start)[w]
        return start, (np.bincount(ids)[1:] if w.any() else np.array([0]))

    G = {"base_rate": base, "n_events": int(lab.sum()), "cloudy_days": int(ncl), "sources": {}}
    for nm, w in [("probability >= base rate", p >= base)] + [(f"probability >= {a_}", p >= a_) for a_ in (0.2, 0.3, 0.5)] + [("median", qk[:, 4] - kn <= -tau)]:
        st, ln = episodes(w)
        fa = w & ~lab
        fst, _ = episodes(fa)
        G["sources"][nm] = {"share_of_forecasts_under_warning": float(w.mean()), "share_under_warning_cloudy_days": float(w[~sunny].mean()),
                            "share_under_warning_sunny_days": float(w[sunny].mean()), "precision": float((w & lab).sum() / max(w.sum(), 1)),
                            "hit_rate": float(w[lab].mean()), "false_alarm_rate": float(fa.sum() / (~lab).sum()),
                            "warning_episodes_per_cloudy_day": float(st[~sunny].sum() / ncl), "median_episode_length_forecasts": float(np.median(ln)),
                            "mean_episode_length_forecasts": float(np.mean(ln)), "false_alarm_episodes_per_cloudy_day": float(fst[~sunny].sum() / ncl)}
        v = G["sources"][nm]
        print(f"G {nm:26s} under warning {v['share_of_forecasts_under_warning']:.3f} (cloudy {v['share_under_warning_cloudy_days']:.3f}) precision {v['precision']:.3f} hit {v['hit_rate']:.3f} "
              f"episodes/cloudy day {v['warning_episodes_per_cloudy_day']:.1f} (median length {v['median_episode_length_forecasts']:.0f}) false-alarm episodes {v['false_alarm_episodes_per_cloudy_day']:.1f}")
    r12 = stored("r12_ramp_warnings_method_v2.json")["timing_falling_ramps"]
    g0 = G["sources"]["probability >= base rate"]
    assert abs(g0["hit_rate"] - r12["share_warned"]) < 1e-12 and abs(g0["false_alarm_episodes_per_cloudy_day"] - r12["false_alarm_episodes_per_cloudy_day"]) < 1e-12, "STOP: R-R12 warnings not reproduced"

    # ---- H: reserve
    sp05, spf = dressed_q05(c)
    spec = {m: (q[:, 4], q[:, 4] - q[:, 0]) for m, q in (("method v2", v2), ("G3''-F", fz["G3''-F"]), ("G4", fz["G4"]), ("E6-S", fz["E6-S"]))}
    spec["smart persistence, dressed (120 min)"] = (spf, spf - sp05)
    spec["constant reserve, official smart persistence"] = (pers, np.zeros(len(y)))
    r12r = stored("r12_reserve_method_v2.json")["methods"]
    udays = np.unique(day)
    odd = np.isin(day, udays[0::2])
    H = {"out_of_sample_constant": {}, "equal_reliability_within_cloudy_days": {}}
    res_c = {}
    for m, (f, b) in spec.items():
        assert abs(equalise(b, f, y) - r12r[m]["delta_kw"]) < 1e-9, f"STOP: R-R12 reserve constant of {m} not reproduced"
        oos = {}
        for nm, fit_m, app in (("fitted on odd days, applied to even days", odd, ~odd), ("fitted on even days, applied to odd days", ~odd, odd)):
            dl = equalise(b[fit_m], f[fit_m], y[fit_m])
            rr = np.maximum(b[app] + dl, 0)
            oos[nm] = {"delta_kw": float(dl), "reliability": float(np.mean(y[app] >= f[app] - rr)), "mean_reserve_kw": float(rr.mean()), "mean_reserve_pct_capacity": float(100 * rr.mean() / 30.1)}
        oos["mean_of_both"] = {kk: float(np.mean([v[kk] for v in oos.values()])) for kk in ("reliability", "mean_reserve_kw", "mean_reserve_pct_capacity")}
        H["out_of_sample_constant"][m] = oos
        dl = equalise(b[cl], f[cl], y[cl])
        res_c[m] = np.maximum(b[cl] + dl, 0)
        falling = (dk <= -tau)[cl]
        H["equal_reliability_within_cloudy_days"][m] = {"delta_kw": float(dl), "reliability": float(np.mean(y[cl] >= f[cl] - res_c[m])), "mean_reserve_kw": float(res_c[m].mean()),
                                                        "mean_reserve_pct_capacity": float(100 * res_c[m].mean() / 30.1),
                                                        "reliability_on_falling_ramps": float(np.mean((y[cl] >= f[cl] - res_c[m])[falling]))}
    for m in spec:
        if m != "method v2":
            H["equal_reliability_within_cloudy_days"][m]["v2_minus_this"] = day_tests(res_c["method v2"] - res_c[m], day[cl], sunny[cl], day_bootstrap)
        o, e = H["out_of_sample_constant"][m]["mean_of_both"], H["equal_reliability_within_cloudy_days"][m]
        dd = e.get("v2_minus_this")
        print(f"H {m:46s} out of sample: reliability {o['reliability']:.3f} reserve {o['mean_reserve_pct_capacity']:.1f}% | cloudy days at 95%: {e['mean_reserve_pct_capacity']:.1f}% "
              f"(falling ramps {e['reliability_on_falling_ramps']:.2f})" + (f" v2 - this {f3(dd['bootstrap'])} p {dd['sign_flip_p']:.4f}" if dd else ""))
    put(A_test=A, B_partner_test=Bt, C_gate_test=Ct, D_median_lag=D, E_coverage_test=E, F_secondary=F, G_warning_operator_quantities=G, H_reserve=H)


# ======================================================================= forward period (as R-R7)
def forward():
    from src.cal2 import ALPHA, conformity, replay
    from src.calibrate import base_interval
    from src.final_eval import Split, day_bootstrap
    from src.revision_e3 import FOLDS, SEEDS as FSEEDS, W, ensemble_on, fit_anchor, fit_gates, predict, sel_preds, sub
    from src.revision_r7_forward_v2 import apply_mix, block_tests, fit_mix

    dev_ = Split("dev", torch.device("cuda"))
    anchor = fit_anchor(dev_)
    day_all = dev_.ds.day.astype(str)
    idx = np.flatnonzero(np.isin(day_all, FOLDS["eval_days"]))
    gates_by_seed = {s: fit_gates(s) for s in FSEEDS}
    q, h60 = ensemble_on(dev_, idx, anchor, gates_by_seed)
    acc, acc_dc, sel_acc = [], [], []
    for s in FSEEDS:
        tau = gates_by_seed[s][1]
        r, d, a = sel_preds("E6-S", s), sel_preds("E4-B-fit", s), sel_preds("A2", s)
        hs = np.isin(r["sample_ids"], a["sample_ids"])
        p_dc = fit_mix({"D": d, "C": r}, tau, ("D", "C"))
        p_rc = fit_mix({"R": a, "C": sub(r, hs)}, tau, ("R", "C"))
        qs = apply_mix(p_dc, {"D": d, "C": r}, tau, ("D", "C"))
        qs[hs] = apply_mix(p_rc, {"R": a, "C": sub(r, hs)}, tau, ("R", "C"))
        sel_acc.append(qs)
        e4, e6 = (predict(dev_, n, s, idx, anchor) for n in ("E4-B-fit", "E6-S"))
        a2 = predict(dev_, "A2", s, idx[h60], anchor)
        qm = apply_mix(p_dc, {"D": e4, "C": e6}, tau, ("D", "C"))
        acc_dc.append(qm.copy())
        qm[h60] = apply_mix(p_rc, {"R": a2, "C": sub(e6, h60)}, tau, ("R", "C"))
        acc.append(qm)
        print(f"forward seed {s} done", flush=True)
    q["method v2"], q["M-DC all"] = np.mean(acc, 0), np.mean(acc_dc, 0)
    tau = gates_by_seed[FSEEDS[0]][1]
    y = dev_.pv[dev_.ds.target_rows[idx]]
    d_ = day_all[idx]
    sunny = np.array([FOLDS["day_labels"][x] for x in d_])
    pc_t = np.maximum(anchor(dev_.rec_t), 1.0)[dev_.ds.target_rows[idx]]
    kn = (dev_.pv / np.maximum(anchor(dev_.rec_t), 1.0))[dev_.ds.hist_rows[idx, -1]]
    lo, hi = (kn - tau) * pc_t, (kn + tau) * pc_t
    ramp = np.abs(y / pc_t - kn) >= tau
    names = ("method v2", "M-DC all", "G4", "G3''-F")
    sc = {m: dict(zip(KEYS, ramp_weighted_crps(q[m], LEVELS, y, lo, hi))) for m in names}
    r7 = json.loads((RESULTS / "r7_forward_method_v2.json").read_text())
    assert abs(sc["method v2"]["crps"].mean() - r7["scores"]["method v2"]["crps"]) < 1e-6, "STOP: method v2 (R-R7) not reproduced"

    def tests(dd, m=None):
        dd, dy, sn = (dd, d_, sunny) if m is None else (dd[m], d_[m], sunny[m])
        r = {"bootstrap": day_bootstrap(dd, dy, sn), "blocks": block_tests(dd, dy, FOLDS["eval_days"])}
        r["shown_first_better"] = bool(r["bootstrap"][2] < 0 and r["blocks"]["sign_flip_p"] < 0.05)
        r["shown_second_better"] = bool(r["bootstrap"][1] > 0 and r["blocks"]["sign_flip_p"] < 0.05)
        return r

    A = {"n": int(len(idx)), "days": int(len(np.unique(d_))), "n_with_60min_history": int(h60.sum()), "ramp_share": float(ramp.mean()),
         "scores": {m: summary(v, sunny, ramp) for m, v in sc.items()}, "contrasts": {}}
    for a_, b_ in (("method v2", "M-DC all"), ("M-DC all", "G4"), ("M-DC all", "G3''-F"), ("method v2", "G4")):
        A["contrasts"][f"{a_} - {b_}"] = {kk: tests(sc[a_][kk] - sc[b_][kk]) for kk in KEYS} | {
            "relative": float(sc[a_]["crps"].mean() / sc[b_]["crps"].mean() - 1), "with_60min_history": {kk: tests(sc[a_][kk] - sc[b_][kk], h60) for kk in KEYS}}
        cc_ = A["contrasts"][f"{a_} - {b_}"]
        print(f"A forward {a_} - {b_}: CRPS {f3(cc_['crps']['bootstrap'])} blocks p {cc_['crps']['blocks']['sign_flip_p']:.4f} rel {cc_['relative']:+.2%} | ramp region {f3(cc_['tw_ramp']['bootstrap'])} | "
              f"calm {f3(cc_['tw_calm']['bootstrap'])} | 60-min forecasts {f3(cc_['with_60min_history']['crps']['bootstrap'])}", flush=True)
    k_ = A["contrasts"]["method v2 - M-DC all"]["crps"]
    A["reading"] = {"the 60-min expert improves the method in the forward period": k_["shown_first_better"], "M-DC all better than method v2": k_["shown_second_better"]}

    # continuous conformal intervals of method v2 (as R-R7), gap interval and direction
    rs = sel_preds("E6-S", FSEEDS[0])
    qsel, ys, os_ = np.mean(sel_acc, 0), rs["target"], rs["origin_time_us"]
    order_s = np.argsort(os_, kind="stable")
    org = dev_.ds.origin_time.astype("datetime64[us]").astype(np.int64)[idx]
    tgt = dev_.ds.target_time.astype("datetime64[us]").astype(np.int64)[idx]
    ci = {}
    for nm, al in ALPHA.items():
        scs = conformity(*base_interval(qsel, LEVELS, nm), ys)[order_s]
        l_, h_ = base_interval(q["method v2"], LEVELS, nm)
        rep = replay(org, tgt, l_, h_, y, al, "rolling", W["G3''-F"], scs[-W["G3''-F"]:])
        ci[nm] = (rep["lo"], rep["hi"])
    nd = nodes7(ci, q["method v2"][:, 4])
    E = {"gap": {}, "direction": {}}
    for nm in ALPHA:
        st = r7["coverage"][nm]
        assert abs(np.mean((y >= ci[nm][0]) & (y <= ci[nm][1])) - st["overall"]) < 1e-9, "STOP: R-R7 coverage not reproduced"
        ev, ec = null_counts(nd, ci[nm][0], ci[nm][1], lo, hi, 1.5 * pc_t, 20261115)
        E["gap"][nm] = gap_bootstrap(y, ci[nm][0], ci[nm][1], lo, hi, ev, ec, d_, sunny)
        assert abs(E["gap"][nm]["ramp_selection_reference"] - st["ramp_selection_reference"]) < 1e-9 and abs(E["gap"][nm]["ramp_observed"] - st["ramp_observed"]) < 1e-9, "STOP: R-R7 selection reference not reproduced"
        E["direction"][nm] = direction_coverage(y, ci[nm][0], ci[nm][1], lo, hi)
        g = E["gap"][nm]
        print(f"E forward {nm}%: ramps observed {g['ramp_observed']:.3f}, reference {g['ramp_selection_reference']:.3f}, gap {g['gap']:+.3f} [{g['gap_interval'][0]:+.3f}, {g['gap_interval'][1]:+.3f}] | "
              f"falling {E['direction'][nm]['falling ramps']['coverage']:.3f} rising {E['direction'][nm]['rising ramps']['coverage']:.3f}", flush=True)
    np.savez_compressed(EVAL / "forward_quantiles_r13.npz", idx=idx, y=y, method_v2=q["method v2"], mdc_all=q["M-DC all"], **{f"lo{nm}": ci[nm][0] for nm in ALPHA}, **{f"hi{nm}": ci[nm][1] for nm in ALPHA})
    put(A_forward=A, E_coverage_forward=E)


if __name__ == "__main__":
    {"dev": dev, "test": test, "forward": forward}[sys.argv[1]]()
