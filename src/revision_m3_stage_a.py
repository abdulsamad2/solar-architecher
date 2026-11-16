"""Revision R-M3, Stage A: does a 60-min expert with the change target and the event-mixture head (R') improve the
method? Development out-of-fold forecasts only (protocol revision/protocols/R-M3_method_v3_development.md). CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.e8_features import FEAT, verify_frozen
from src.gate7b import grid_quantiles
from src.metrics import crps_quantile_interp
from src.ramp_scoring import ramp_weighted_crps
from src.revision_common import LEVELS, day_bootstrap, save
from src.revision_m1_method_v2 import PARTS, ROOT as ROOT_V2, TAU, MixN, features, fit, mixture_quantiles, select_and_fit, weights
from src.revision_r1_gate_proper_score import band
from src.revision_x2_linear_pool import load

ROOT = {"R": ROOT_V2["R"], "C": ROOT_V2["C"], "N": Path("D:/skippd_hires/R_M3/Rdk/csi")}  # N = the new expert R'
DESIGNS = {"M-RC": ("R", "C"), "M-R'C": ("N", "C"), "M-R'RC": ("N", "R", "C")}  # the calm expert is the reference component
SEEDS = (42, 123, 456)


def main():
    verify_frozen()
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = win["sample_ids"][win["eligible_common"]]
    stored = json.loads(Path("revision/results/m1_method_v2_development.json").read_text())["per_seed"]
    methods = ["R'", "R", "C"] + [f"{d}{s}" for d in DESIGNS for s in ("", " const")]
    res = {"protocol": "revision/protocols/R-M3_method_v3_development.md", "per_seed": {}, "mean_weights": {}}
    sc_all = {}
    for s in SEEDS:
        qs = {m: [] for m in methods}
        ys, los, his, days, sun = [], [], [], [], []
        wmean = {d: [] for d in DESIGNS}
        for k in range(5):
            tau = TAU[str(k)]["training_labels_fit_days"]
            cal = {e: load(r, s, k, "cal", common) for e, r in ROOT.items()}
            val = {e: load(r, s, k, "val", common) for e, r in ROOT.items()}
            for e in ("R", "N"):
                assert np.array_equal(cal[e]["sample_ids"], cal["C"]["sample_ids"]) and np.array_equal(val[e]["sample_ids"], val["C"]["sample_ids"])
            order = np.argsort(cal["C"]["origin_time_us"], kind="stable")
            for name, e in (("R'", "N"), ("R", "R"), ("C", "C")):
                qs[name].append(val[e]["pred_q"])
            for d, comps in DESIGNS.items():
                f = MixN([grid_quantiles(cal[e]["pred_q"], LEVELS) for e in comps], cal["C"]["target"])
                zc, zv = features(cal, tau, comps), features(val, tau, comps)
                p = select_and_fit(zc, f, order, len(comps))
                wv = weights(torch.as_tensor((zv - p["mu"]) / p["sd"]), p["b"].expand(len(zv), -1), p["B"]).numpy()
                qs[d].append(mixture_quantiles([val[e]["pred_q"] for e in comps], wv))
                b0, _ = fit(zc, f, 0.0, len(comps), gated=False)
                wc = weights(torch.zeros(len(zv), 0, dtype=torch.float64), b0.expand(len(zv), -1), None).numpy()
                qs[f"{d} const"].append(mixture_quantiles([val[e]["pred_q"] for e in comps], wc))
                wmean[d].append(wv.mean(0))
            lo, hi = band(val["C"], tau)
            ys.append(val["C"]["target"]); los.append(lo); his.append(hi); days.append(val["C"]["day"]); sun.append(val["C"]["clear_day"])
        cat = np.concatenate
        y, lo, hi, day, sunny = cat(ys), cat(los), cat(his), cat(days), cat(sun).astype(bool)
        sc = {}
        for m in methods:
            q = cat(qs[m])
            sc[m] = dict(zip(PARTS, ramp_weighted_crps(q, LEVELS, y, lo, hi)))
            sc[m]["cov90"], sc[m]["w90"] = (y >= q[:, 0]) & (y <= q[:, -1]), q[:, -1] - q[:, 0]
            assert np.abs(sc[m]["crps"] - crps_quantile_interp(q, y, LEVELS)).max() < 1e-9
        assert abs(sc["M-RC"]["crps"].mean() - stored[str(s)]["M-RC"]["crps"]) < 1e-9, "STOP: method v2 (R-M1) not reproduced"
        sc_all[s] = (sc, day, sunny)
        res["per_seed"][str(s)] = {m: {"crps": float(v["crps"].mean()), "tw_ramp": float(v["tw_ramp"].mean()), "tw_calm": float(v["tw_calm"].mean()),
                                       "sunny": float(v["crps"][sunny].mean()), "cloudy": float(v["crps"][~sunny].mean()),
                                       "raw90_coverage": float(v["cov90"].mean()), "raw90_width_kw": float(v["w90"].mean())} for m, v in sc.items()}
        res["mean_weights"][str(s)] = {d: np.mean(v, 0).tolist() for d, v in wmean.items()}
        print(f"seed {s}: " + "  ".join(f"{m} {res['per_seed'][str(s)][m]['crps']:.4f}" for m in methods), flush=True)

    def contrast(a, b):
        return {str(s): {p: day_bootstrap(sc[a][p] - sc[b][p], day, sunny) for p in PARTS} for s, (sc, day, sunny) in sc_all.items()}

    pairs = [("R'", "M-RC"), ("M-R'C", "M-RC"), ("M-R'RC", "M-RC"), ("M-R'C", "R'"), ("M-R'RC", "R'"), ("M-R'RC", "M-R'C"), ("R'", "R"), ("R'", "C")]
    pairs += [(d, f"{d} const") for d in DESIGNS]
    res["contrasts"] = {f"{a} - {b}": contrast(a, b) for a, b in pairs}
    mean = lambda m: float(np.mean([res["per_seed"][str(s)][m]["crps"] for s in SEEDS]))
    res["seed_mean_crps"] = {m: mean(m) for m in methods}

    def qualifies(a, b):  # rule 1: a replaces b
        c = res["contrasts"][f"{a} - {b}"]
        better = all(c[s]["crps"][2] < 0 for s in c)
        gain = (mean(b) - mean(a)) / mean(b)
        calm_ok = not any(c[s]["tw_calm"][1] > 0 for s in c)
        return bool(better and gain >= 0.01 and calm_ok), {"3/3 seeds CI<0": bool(better), "seed-mean gain >= 1%": bool(gain >= 0.01),
                                                         "calm region not worse": bool(calm_ok), "relative_gain": gain}

    c_alone = res["contrasts"]["R' - M-RC"]
    alone_ok = not any(c_alone[s]["crps"][1] > 0 for s in c_alone)  # simpler: replaces the incumbent if not worse
    q_nc, d_nc = qualifies("M-R'C", "M-RC")
    q_nrc, d_nrc = qualifies("M-R'RC", "M-RC")
    chosen = "M-RC"
    if alone_ok:
        chosen = "R'"
        q1, d1 = qualifies("M-R'C", "R'")
        q2, d2 = qualifies("M-R'RC", "R'")
        res["against R' alone"] = {"M-R'C": d1, "M-R'RC": d2}
        if q1:
            chosen = "M-R'C"
        if q2 and (not q1 or qualifies("M-R'RC", "M-R'C")[0]):
            chosen = "M-R'RC"
    else:
        if q_nc:
            chosen = "M-R'C"
        if q_nrc and (not q_nc or qualifies("M-R'RC", "M-R'C")[0]):
            chosen = "M-R'RC"
    method = chosen
    if chosen in DESIGNS:
        gc = res["contrasts"][f"{chosen} - {chosen} const"]
        if not all(gc[s]["crps"][2] < 0 for s in gc):
            method = f"{chosen} const"
    res["selection"] = {"R' alone not worse than M-RC in any seed": bool(alone_ok), "M-R'C replaces M-RC": d_nc, "M-R'RC replaces M-RC": d_nrc,
                        "M-R'RC replaces M-R'C": qualifies("M-R'RC", "M-R'C")[1], "design": chosen, "method": method}
    f = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"
    for name, c in res["contrasts"].items():
        print(f"{name:20s} " + " | ".join(f"seed {s}: {f(c[s]['crps'])} ramp {c[s]['tw_ramp'][0]:+.4f} calm {c[s]['tw_calm'][0]:+.4f}" for s in c), flush=True)
    print("seed-mean CRPS:", {m: round(v, 4) for m, v in res["seed_mean_crps"].items()})
    print("selection:", json.dumps(res["selection"], indent=1))
    print("saved", save("m3_stage_a.json", res))


if __name__ == "__main__":
    main()
