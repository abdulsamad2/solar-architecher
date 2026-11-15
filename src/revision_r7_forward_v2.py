"""Revision R-R7: method v2 (gated probability mixture) and review-driven checks on the forward period
(protocol revision/protocols/R-R7_forward_method_v2.md). Forward experts of R-E3; inference only, no training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from src.baselines import official_persistence
from src.cal2 import ALPHA, conformity, replay
from src.calibrate import base_interval
from src.final_eval import Split, day_bootstrap
from src.gate7b import grid_quantiles
from src.ramp_scoring import ramp_weighted_crps, selection_null
from src.revision_common import LEVELS, save
from src.revision_e3 import FOLDS, MODELS, SEEDS, W, blend, ensemble_on, fit_anchor, fit_gates, predict, sel_preds, sub
from src.revision_m1_method_v2 import MixN, features, mixture_quantiles, select_and_fit, weights
from src.revision_r2_day_level_inference import sign_flip

MIN = 60_000_000
P = np.array([.05, .10, .25, .50, .75, .90, .95])
KEYS = ("crps", "tw_ramp", "tw_calm")


def fit_mix(pd_, tau, comps):
    """Gate of a mixture design fitted on selection-period predictions pd_ = {'D'|'R': dict, 'C': dict}."""
    order = np.argsort(pd_["C"]["origin_time_us"], kind="stable")
    f = MixN([grid_quantiles(pd_[e]["pred_q"], LEVELS) for e in comps], pd_["C"]["target"])
    return select_and_fit(features(pd_, tau, comps), f, order, len(comps))


def apply_mix(p, pd_, tau, comps):
    z = features(pd_, tau, comps)
    w = weights(torch.as_tensor((z - p["mu"]) / p["sd"]), p["b"].expand(len(z), -1), p["B"]).numpy()
    return mixture_quantiles([pd_[e]["pred_q"] for e in comps], w)


def block_tests(d, day, eval_days, size=7):
    blocks = {x: i // size for i, x in enumerate(sorted(eval_days))}
    b = np.array([blocks[x] for x in day])
    S, n = np.bincount(b, weights=d), np.bincount(b).astype(float)
    m = S / n
    q = stats.t.ppf(0.975, len(m) - 1)
    se = m.std(ddof=1) / np.sqrt(len(m))
    return {"blocks": int(len(m)), "sign_flip_p": sign_flip(S, n)[0], "t_interval_block_means": [float(m.mean()), float(m.mean() - q * se), float(m.mean() + q * se)],
            "blocks_first_better": int((S < 0).sum())}


def naive_references(dev, anchor, idx, fit_days):
    """PeEn-120, smart persistence dressed with the last 120 min of errors, CH-PeEn from the fit period (as R-R3)."""
    pc = np.maximum(anchor(dev.rec_t), 1.0)
    k = dev.pv / pc
    t = dev.rec_t.astype("datetime64[us]").astype(np.int64)
    d = dev.rec_t.astype("datetime64[D]")
    ds = dev.ds
    now, tgt = ds.hist_rows[:, -1], ds.target_rows
    fit_rec = np.isin(d.astype(str), fit_days)
    fit_s = np.isin(ds.day.astype(str), fit_days)
    q_static = np.quantile((k[tgt] - k[now])[fit_s], LEVELS)
    hour = dev.rec_t.astype("datetime64[h]").astype(np.int64) % 24
    ch = {h: np.quantile(k[fit_rec & (hour == h)], LEVELS) for h in np.unique(hour[fit_rec])}
    order = np.argsort(t, kind="stable")
    ts, ks, dsd = t[order], k[order], d[order]
    s_t0, s_tt, s_dk, s_day = t[now], t[tgt], k[tgt] - k[now], d[now]
    # restrict the pool of matured changes to the evaluation days (same-day window only)
    ev = np.zeros(len(now), bool)
    ev[idx] = True
    out = {n: np.empty((len(idx), 9)) for n in ("PeEn-120", "SP-dressed adaptive", "CH-PeEn")}
    by_day = {}
    for j in np.flatnonzero(ev):
        by_day.setdefault(s_day[j], []).append(j)
    by_day = {dd: np.array(v) for dd, v in by_day.items()}
    for r, i in enumerate(idx):
        t0 = s_t0[i]
        hi = np.searchsorted(ts, t0, side="right")
        lo = np.searchsorted(ts, t0 - 120 * MIN, side="right")
        mem = ks[lo:hi][dsd[lo:hi] == s_day[i]]
        out["PeEn-120"][r] = np.quantile(mem, LEVELS) if len(mem) >= 5 else k[now[i]]
        pool = by_day[s_day[i]]
        past = pool[(s_tt[pool] <= t0) & (s_t0[pool] > t0 - 120 * MIN)]
        out["SP-dressed adaptive"][r] = k[now[i]] + (np.quantile(s_dk[past], LEVELS) if len(past) >= 10 else q_static)
        h = hour[tgt[i]]
        out["CH-PeEn"][r] = ch[h] if h in ch else np.quantile(k[fit_rec], LEVELS)
    return {n: np.maximum(v, 0.0) * pc[tgt[idx]][:, None] for n, v in out.items()}


def main():
    dev = Split("dev", torch.device("cuda"))
    anchor = fit_anchor(dev)
    day_all = dev.ds.day.astype(str)
    idx = np.flatnonzero(np.isin(day_all, FOLDS["eval_days"]))
    gates_by_seed = {s: fit_gates(s) for s in SEEDS}
    q, h60 = ensemble_on(dev, idx, anchor, gates_by_seed)  # frozen rule, G4, experts (as R-E3)
    acc, sel_acc = [], []
    for s in SEEDS:
        tau = gates_by_seed[s][1]
        r, d, a = sel_preds("E6-S", s), sel_preds("E4-B-fit", s), sel_preds("A2", s)
        hs = np.isin(r["sample_ids"], a["sample_ids"])
        p_dc = fit_mix({"D": d, "C": r}, tau, ("D", "C"))
        p_rc = fit_mix({"R": a, "C": sub(r, hs)}, tau, ("R", "C"))
        qs = apply_mix(p_dc, {"D": d, "C": r}, tau, ("D", "C"))  # selection-period forecasts (gates in-sample; for initial scores)
        qs[hs] = apply_mix(p_rc, {"R": a, "C": sub(r, hs)}, tau, ("R", "C"))
        sel_acc.append(qs)
        e4, e6 = (predict(dev, n, s, idx, anchor) for n in ("E4-B-fit", "E6-S"))
        a2 = predict(dev, "A2", s, idx[h60], anchor)
        qm = apply_mix(p_dc, {"D": e4, "C": e6}, tau, ("D", "C"))
        qm[h60] = apply_mix(p_rc, {"R": a2, "C": sub(e6, h60)}, tau, ("R", "C"))
        acc.append(qm)
    q["method v2"] = np.mean(acc, 0)
    tau = gates_by_seed[SEEDS[0]][1]
    y = dev.pv[dev.ds.target_rows[idx]]
    d_ = day_all[idx]
    sunny = np.array([FOLDS["day_labels"][x] for x in d_])
    pc_t = np.maximum(anchor(dev.rec_t), 1.0)[dev.ds.target_rows[idx]]
    k = dev.pv / np.maximum(anchor(dev.rec_t), 1.0)
    kn = k[dev.ds.hist_rows[idx, -1]]
    lo, hi = (kn - tau) * pc_t, (kn + tau) * pc_t
    ramp = np.abs(y / pc_t - kn) >= tau
    q |= naive_references(dev, anchor, idx, FOLDS["folds"][0]["train_days"])
    sc = {m: dict(zip(KEYS, ramp_weighted_crps(v, LEVELS, y, lo, hi))) for m, v in q.items() if m != "A2"}
    pers = official_persistence(dev.ds.origin_time[idx].astype(object), dev.pv[dev.ds.hist_rows[idx, -1]])
    sc["official smart persistence (point)"] = dict(zip(KEYS, ramp_weighted_crps(np.repeat(pers[:, None], 9, 1), LEVELS, y, lo, hi)))
    stored = json.loads(Path("revision/results/e3_forward.json").read_text())["crps"]
    for m in MODELS:
        assert abs(sc[m]["crps"].mean() - stored[m]["overall"]) < 1e-6, f"STOP: R-E3 CRPS of {m} not reproduced"
    res = {"protocol": "revision/protocols/R-R7_forward_method_v2.md", "n": int(len(idx)), "days": int(len(np.unique(d_))), "ramp_share": float(ramp.mean()),
           "scores": {m: {kk: float(v[kk].mean()) for kk in KEYS} | {"sunny": float(v["crps"][sunny].mean()), "cloudy": float(v["crps"][~sunny].mean()),
                                                                       "ramp_outcomes": float(v["crps"][ramp].mean())} for m, v in sc.items()},
           "v2_minus": {}}
    f = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"
    for m in sc:
        if m == "method v2":
            continue
        dd = {kk: sc["method v2"][kk] - sc[m][kk] for kk in KEYS}
        r = {kk: day_bootstrap(dd[kk], d_, sunny) for kk in KEYS} | {"blocks": block_tests(dd["crps"], d_, FOLDS["eval_days"]),
                                                                      "skill": float(1 - sc["method v2"]["crps"].mean() / sc[m]["crps"].mean())}
        r["better"] = bool(r["crps"][2] < 0 and r["blocks"]["sign_flip_p"] < 0.05)
        res["v2_minus"][m] = r
        print(f"v2 - {m:36s} CRPS {res['scores'][m]['crps']:.4f}: {f(r['crps'])} skill {r['skill']:+.1%} | ramp region {f(r['tw_ramp'])} | calm {f(r['tw_calm'])} | "
              f"7-day blocks p {r['blocks']['sign_flip_p']:.4f} t {f(r['blocks']['t_interval_block_means'])} ({r['blocks']['blocks_first_better']}/{r['blocks']['blocks']}) => {'BETTER' if r['better'] else 'not shown'}", flush=True)
    print(f"method v2 CRPS {res['scores']['method v2']['crps']:.4f} (sunny {res['scores']['method v2']['sunny']:.4f}, cloudy {res['scores']['method v2']['cloudy']:.4f})")

    # --- continuous conformal intervals of method v2 and the selection reference for ramp coverage
    rs = sel_preds("E6-S", SEEDS[0])
    qsel, ys, os_ = np.mean(sel_acc, 0), rs["target"], rs["origin_time_us"]
    order_s = np.argsort(os_, kind="stable")
    org = dev.ds.origin_time.astype("datetime64[us]").astype(np.int64)[idx]
    tgt = dev.ds.target_time.astype("datetime64[us]").astype(np.int64)[idx]
    ci = {}
    for nm, a in ALPHA.items():
        scs = conformity(*base_interval(qsel, LEVELS, nm), ys)[order_s]
        l_, h_ = base_interval(q["method v2"], LEVELS, nm)
        rep = replay(org, tgt, l_, h_, y, a, "rolling", W["G3''-F"], scs[-W["G3''-F"]:])
        ci[nm] = (rep["lo"], rep["hi"])
    nodes = np.sort(np.stack([ci["90"][0], ci["80"][0], ci["50"][0], q["method v2"][:, 4], ci["50"][1], ci["80"][1], ci["90"][1]], 1), 1)
    res["coverage"] = {}
    for nm in ALPHA:
        cov = (y >= ci[nm][0]) & (y <= ci[nm][1])
        null = selection_null(nodes, P, ci[nm][0], ci[nm][1], lo, hi, y=y, cap=1.5 * pc_t, seed=20261115)
        nominal = 1 - ALPHA[nm]
        res["coverage"][nm] = {"overall": float(cov.mean()), "ramp_observed": null["observed_ramp_coverage"], "ramp_selection_reference": null["null_ramp_coverage"],
                               "reference_range": null["null_ramp_coverage_range"], "gap": null["gap"], "null_ramp_rate": null["null_ramp_rate"],
                               "selection_share": float((nominal - null["null_ramp_coverage"]) / (nominal - null["observed_ramp_coverage"])) if nominal > null["observed_ramp_coverage"] else None,
                               "width_kw": float(np.mean(ci[nm][1] - ci[nm][0]))}
        c = res["coverage"][nm]
        print(f"{nm}% intervals (continuous conformal): overall {c['overall']:.3f}, ramps observed {c['ramp_observed']:.3f}, selection reference {c['ramp_selection_reference']:.3f} "
              f"{np.round(c['reference_range'], 3).tolist()}, gap {c['gap']:+.3f}, selection share {c['selection_share']}, width {c['width_kw']:.2f} kW")
    print("saved", save("r7_forward_method_v2.json", res))


if __name__ == "__main__":
    main()
