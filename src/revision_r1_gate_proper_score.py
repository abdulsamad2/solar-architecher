"""Revision R-R1: which gate feature carries the benefit, under the overall and the proper ramp-weighted CRPS
(protocol revision/protocols/R-R1_gate_features_proper_score.md). Same fits and samples as R-E2; CPU only, no training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

from src.e8_features import FEAT, verify_frozen
from src.final_eval import blend, gate_params
from src.gate7b import BlendCRPS, apply_gate, grid_quantiles, raw_features, select_and_fit
from src.ramp_scoring import ramp_weighted_crps
from src.revision_common import LEVELS, day_bootstrap, save, test_context
from src.revision_e2 import A2, E4B, E6S, G7B, METHODS, constant_weight, load
from src.revision_posthoc_test import TAU, mdict, members, sub
from src.revision_z_common import ramp_region

PARTS = ("crps", "tw_ramp", "tw_calm")
ALTS = ("B50", "BC", "Gg", "Gd", "G4", "E6-S", "A2")


def band(d, tau):
    """Non-ramp band (kW) of each forecast from the fold's clear-sky anchor and threshold (known at issue time)."""
    pc = np.maximum(d["pclear_target_kw"], 1.0)
    kn = d["pv_now_kw"] / np.maximum(d["pclear_now_kw"], 1.0)
    return (kn - tau) * pc, (kn + tau) * pc


def summarise(q, y, lo, hi, day, sunny, stored):
    sc = {m: dict(zip(PARTS, ramp_weighted_crps(v, LEVELS, y, lo, hi))) for m, v in q.items()}
    for m in sc:
        assert np.abs(sc[m]["crps"] - sc[m]["tw_ramp"] - sc[m]["tw_calm"]).max() < 1e-9, "STOP: parts do not add up"
        assert abs(sc[m]["crps"].mean() - stored[m]) < 1e-9, f"STOP: stored R-E2 CRPS of {m} not reproduced"
    out = {"scores": {m: {p: float(v[p].mean()) for p in PARTS} for m, v in sc.items()},
           "G3pp_minus": {m: {p: day_bootstrap(sc["G3''"][p] - sc[m][p], day, sunny) for p in PARTS} for m in ALTS}}
    gain = out["scores"]["BC"]["crps"] - out["scores"]["G3''"]["crps"]
    out["share_of_gate_gain_recovered"] = {m: float((out["scores"]["BC"]["crps"] - out["scores"][m]["crps"]) / gain) for m in ("Gg", "Gd")}
    return out


def development():
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = set(win["sample_ids"][win["eligible_common"]].tolist())
    stored = json.loads(Path("revision/results/e2_gate_ablation_dev.json").read_text())["per_seed"]
    res = {}
    for s in (42, 123, 456):
        qs = {m: [] for m in METHODS}
        ys, los, his, days, sun, fg, fd = [], [], [], [], [], [], []
        for k in range(5):
            tau = TAU[str(k)]["training_labels_fit_days"]
            rc, rv = load(E6S, s, k, "cal", common), load(E6S, s, k, "val", common)
            dc, dv = load(A2, s, k, "cal", common), load(A2, s, k, "val", common)
            fc, fv = raw_features(dc, rc, tau), raw_features(dv, rv, tau)
            order = np.argsort(rc["origin_time_us"], kind="stable")
            fits = {n: select_and_fit(fc, f, dc["pred_q"], rc["pred_q"], rc["target"], LEVELS, order)
                    for n, f in (("Gg", ("g",)), ("Gd", ("d",)), ("G3''", ("g", "d")))}
            w = {"B50": np.full(len(rv["target"]), 0.5), "BC": np.full(len(rv["target"]), constant_weight(dc, rc))}
            w |= {n: apply_gate(p, fv) for n, p in fits.items()}
            for n, ww in w.items():
                qs[n].append(ww[:, None] * dv["pred_q"] + (1 - ww[:, None]) * rv["pred_q"])
            qs["A2"].append(dv["pred_q"]); qs["E6-S"].append(rv["pred_q"])
            ev = load(E4B, s, k, "val", common)
            p4 = json.loads((G7B / f"seed{s}" / f"fold_{k}" / "params.json").read_text())["G4"]
            w4 = apply_gate(p4, raw_features(ev, rv, tau))
            qs["G4"].append(w4[:, None] * ev["pred_q"] + (1 - w4[:, None]) * rv["pred_q"])
            lo, hi = band(rv, tau)
            ys.append(rv["target"]); los.append(lo); his.append(hi); days.append(rv["day"]); sun.append(rv["clear_day"])
            fg.append(fv["g"]); fd.append(fv["d"])
        cat = np.concatenate
        r = summarise({m: cat(v) for m, v in qs.items()}, cat(ys), cat(los), cat(his), cat(days), cat(sun),
                      {m: stored[str(s)]["crps"][m]["overall"] for m in METHODS})
        r["spearman_g_d"] = float(spearmanr(cat(fg), cat(fd)).statistic)
        res[str(s)] = r
        show(f"development seed {s}", r)
    return res


def test():
    c = test_context()
    h60 = c["h60"]
    _, lo, hi = ramp_region(c)
    r, a, d = members("E6-S"), members("A2"), members("E4-B-fit")
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = win["sample_ids"][win["eligible_common"]]
    exp = lambda root, s, k: dict(np.load(Path(root) / f"seed{s}" / f"fold_{k}" / "preds_cal.npz"))
    acc = {m: [] for m in METHODS}
    fg, fd = [], []
    for i, (s, k) in enumerate(r["members"]):
        tau = TAU[str(k)]["training_labels_fit_days"]
        rc, dc = exp(E6S, s, k), exp(A2, s, k)
        rc, dc = sub(rc, np.isin(rc["sample_ids"], common)), sub(dc, np.isin(dc["sample_ids"], common))
        fc = raw_features(dc, rc, tau)
        order = np.argsort(rc["origin_time_us"], kind="stable")
        mr, ma = sub(mdict(r, i), h60), sub(mdict(a, i), h60)
        fv = raw_features(ma, mr, tau)
        fb = BlendCRPS(grid_quantiles(dc["pred_q"], LEVELS), grid_quantiles(rc["pred_q"], LEVELS), rc["target"])
        n = len(rc["target"])
        wc = minimize_scalar(lambda x: float(fb.per_sample(torch.full((n,), float(x), dtype=torch.float64)).mean()),
                             bounds=(0, 1), method="bounded", options={"xatol": 1e-6}).x
        w = {"B50": np.full(h60.sum(), 0.5), "BC": np.full(h60.sum(), wc)}
        for nm, names in (("Gg", ("g",)), ("Gd", ("d",))):
            w[nm] = apply_gate(select_and_fit(fc, names, dc["pred_q"], rc["pred_q"], rc["target"], LEVELS, order), fv)
        w["G3''"] = apply_gate(gate_params("G3''", s, k), fv)  # frozen G3'' parameters
        for nm, ww in w.items():
            acc[nm].append(ww[:, None] * ma["pred_q"] + (1 - ww[:, None]) * mr["pred_q"])
        acc["A2"].append(ma["pred_q"]); acc["E6-S"].append(mr["pred_q"])
        acc["G4"].append(blend(sub(mdict(d, i), h60), mr, gate_params("G4", s, k), tau))
        fg.append(fv["g"]); fd.append(fv["d"])
    stored = json.loads(Path("revision/results/posthoc_test_e2.json").read_text())["crps"]
    res = summarise({m: np.mean(v, 0) for m, v in acc.items()}, c["y"][h60], lo[h60], hi[h60], c["day"][h60], c["sunny"][h60],
                    {m: stored[m]["overall"] for m in METHODS})
    res["n"] = int(h60.sum())
    res["spearman_g_d"] = float(spearmanr(np.mean(fg, 0), np.mean(fd, 0)).statistic)
    show("official test (60-min-eligible)", res)
    return res


def show(title, r):
    f = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"
    print(f"{title}: Spearman(g, d) {r['spearman_g_d']:.2f}; share of the gate's gain over the constant weight recovered: "
          f"g-only {r['share_of_gate_gain_recovered']['Gg']:.2f}, d-only {r['share_of_gate_gain_recovered']['Gd']:.2f}")
    for m in ("BC", "Gg", "Gd", "G4"):
        d = r["G3pp_minus"][m]
        print(f"   G3'' - {m:3s}: overall {f(d['crps'])}  ramp region {f(d['tw_ramp'])}  calm region {f(d['tw_calm'])}", flush=True)


def main():
    verify_frozen()
    res = {"protocol": "revision/protocols/R-R1_gate_features_proper_score.md", "development": development(), "test": test()}
    neg = lambda v: v[2] < 0
    T, D = res["test"]["G3pp_minus"], [res["development"][s]["G3pp_minus"] for s in ("42", "123", "456")]
    res["reading"] = {
        "state-dependent gating helps (vs constant weight, overall)": bool(neg(T["BC"]["crps"]) and all(neg(d["BC"]["crps"]) for d in D)),
        "removing d is worse: overall (test, dev seeds)": [bool(neg(T["Gg"]["crps"]))] + [bool(neg(d["Gg"]["crps"])) for d in D],
        "removing d is worse: ramp region (test, dev seeds)": [bool(neg(T["Gg"]["tw_ramp"]))] + [bool(neg(d["Gg"]["tw_ramp"])) for d in D],
        "removing g is worse: overall (test, dev seeds)": [bool(neg(T["Gd"]["crps"]))] + [bool(neg(d["Gd"]["crps"])) for d in D],
        "removing g is worse: ramp region (test, dev seeds)": [bool(neg(T["Gd"]["tw_ramp"]))] + [bool(neg(d["Gd"]["tw_ramp"])) for d in D]}
    a = res["reading"]
    res["reading"]["disagreement carries the benefit (protocol rule)"] = bool(
        all(a["removing d is worse: overall (test, dev seeds)"]) and all(a["removing d is worse: ramp region (test, dev seeds)"])
        and not all(a["removing g is worse: overall (test, dev seeds)"]) and not all(a["removing g is worse: ramp region (test, dev seeds)"]))
    for k, v in res["reading"].items():
        print(f"{k}: {v}")
    print("saved", save("r1_gate_proper_score.json", res))


if __name__ == "__main__":
    main()
