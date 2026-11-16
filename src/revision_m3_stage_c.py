"""Revision R-M3, Stage C: continuous rolling conformal calibration in classes of the forecast-time ramp probability g,
against the pooled window (protocol revision/protocols/R-M3_method_v3_development.md, Amendment 2). Development
validation forecasts of method v3 (M-R'RC, one member per seed and fold); CPU only, no training.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from src.cal2 import ALPHA, interval_score
from src.calibrate import base_interval
from src.e8_features import FEAT, verify_frozen
from src.gate7b import grid_quantiles, raw_features
from src.revision_common import LEVELS, save
from src.revision_m1_method_v2 import TAU, MixN, features, mixture_quantiles, select_and_fit, weights
from src.revision_m3_stage_a import ROOT, SEEDS
from src.revision_r1_gate_proper_score import band
from src.revision_x2_linear_pool import load

COMPS = ("N", "R", "C")
W, N_MIN_RAW, N_MIN_CLASS = 4620, 30, 200
EDGES = np.array([0.1, 0.3, 0.5])
CLASSES = ("g < 0.1", "0.1-0.3", "0.3-0.5", "g >= 0.5")


def rolling(org, tgt, lo, hi, y, alpha, cls=None):
    """Calibrated bounds; a score enters a window only when its target time is not later than the issue time."""
    n = len(y)
    s = np.maximum(lo - y, y - hi)
    by_tgt = np.argsort(tgt, kind="stable")
    by_org = np.argsort(org, kind="stable")
    pool, n_pool = np.empty(n), 0
    ncls = 0 if cls is None else int(cls.max()) + 1
    cp, cn = [np.empty(n) for _ in range(ncls)], [0] * ncls
    q, ptr = np.zeros(n), 0

    def kth(a):
        m = len(a)
        k = min(int(np.ceil((m + 1) * (1 - alpha))), m)
        return np.partition(a, k - 1)[k - 1]

    for i in by_org:
        while ptr < n and tgt[by_tgt[ptr]] <= org[i]:
            j = by_tgt[ptr]
            pool[n_pool] = s[j]; n_pool += 1
            if cls is not None:
                c = cls[j]
                cp[c][cn[c]] = s[j]; cn[c] += 1
            ptr += 1
        if cls is not None and cn[cls[i]] >= N_MIN_CLASS:
            c = cls[i]
            q[i] = kth(cp[c][max(0, cn[c] - W):cn[c]])
        elif n_pool >= N_MIN_RAW:
            q[i] = kth(pool[max(0, n_pool - W):n_pool])
    return lo - q, hi + q


def summary(lo, hi, y, alpha, masks):
    cov, wid, sc = (y >= lo) & (y <= hi), hi - lo, interval_score(lo, hi, y, alpha)
    return {g: {"coverage": float(cov[m].mean()), "width_kw": float(wid[m].mean()), "interval_score": float(sc[m].mean()), "n": int(m.sum())} for g, m in masks.items() if m.any()}


def main():
    verify_frozen()
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = win["sample_ids"][win["eligible_common"]]
    res = {"protocol": "revision/protocols/R-M3_method_v3_development.md (Amendment 2)", "W": W, "class_edges": EDGES.tolist(), "per_seed": {}}
    for s in SEEDS:
        acc = {k: [] for k in ("q", "y", "org", "tgt", "g", "ramp")}
        for k in range(5):
            tau = TAU[str(k)]["training_labels_fit_days"]
            cal = {e: load(r, s, k, "cal", common) for e, r in ROOT.items()}
            val = {e: load(r, s, k, "val", common) for e, r in ROOT.items()}
            order = np.argsort(cal["C"]["origin_time_us"], kind="stable")
            f = MixN([grid_quantiles(cal[e]["pred_q"], LEVELS) for e in COMPS], cal["C"]["target"])
            p = select_and_fit(features(cal, tau, COMPS), f, order, len(COMPS))
            zv = features(val, tau, COMPS)
            wv = weights(torch.as_tensor((zv - p["mu"]) / p["sd"]), p["b"].expand(len(zv), -1), p["B"]).numpy()
            lo, hi = band(val["C"], tau)
            y = val["C"]["target"]
            acc["q"].append(mixture_quantiles([val[e]["pred_q"] for e in COMPS], wv)); acc["y"].append(y)
            acc["org"].append(val["C"]["origin_time_us"]); acc["tgt"].append(val["C"]["target_time_us"])
            acc["g"].append(raw_features(val["R"], val["C"], tau)["g"]); acc["ramp"].append((y < lo) | (y > hi))
        d = {k: np.concatenate(v) for k, v in acc.items()}
        cls = np.digitize(d["g"], EDGES)
        masks = {"overall": np.ones(len(d["y"]), bool), "ramp": d["ramp"], "non-ramp": ~d["ramp"]} | {c: cls == i for i, c in enumerate(CLASSES)}
        out = {"n": int(len(d["y"])), "ramp_share": float(d["ramp"].mean()), "class_share": {c: float((cls == i).mean()) for i, c in enumerate(CLASSES)}}
        for nm, a in ALPHA.items():
            lo, hi = base_interval(d["q"], LEVELS, nm)
            out[nm] = {"raw": summary(lo, hi, d["y"], a, masks), "U": summary(*rolling(d["org"], d["tgt"], lo, hi, d["y"], a), d["y"], a, masks),
                       "G": summary(*rolling(d["org"], d["tgt"], lo, hi, d["y"], a, cls), d["y"], a, masks)}
            print(f"seed {s} {nm}%: " + " | ".join(f"{v} cov {out[nm][v]['overall']['coverage']:.3f} ramp {out[nm][v]['ramp']['coverage']:.3f} width {out[nm][v]['overall']['width_kw']:.3f} "
                                                   f"score {out[nm][v]['overall']['interval_score']:.3f}" for v in ("raw", "U", "G")), flush=True)
        res["per_seed"][str(s)] = out
    key = next(nm for nm, a in ALPHA.items() if abs(a - 0.10) < 1e-9)
    ps = res["per_seed"]
    a_ok = all(abs(ps[s][key]["G"]["ramp"]["coverage"] - 0.9) < abs(ps[s][key]["U"]["ramp"]["coverage"] - 0.9) for s in ps)
    wr = float(np.mean([ps[s][key]["G"]["overall"]["width_kw"] for s in ps]) / np.mean([ps[s][key]["U"]["overall"]["width_kw"] for s in ps]))
    c_ok = all(ps[s][key]["G"]["overall"]["interval_score"] <= ps[s][key]["U"]["overall"]["interval_score"] for s in ps)
    res["acceptance_90"] = {"(a) ramp coverage closer to 0.90 in 3/3 seeds": bool(a_ok), "(b) width ratio G/U (seed mean)": wr, "(b) width at most 5% larger": bool(wr <= 1.05),
                            "(c) interval score not worse in any seed": bool(c_ok), "accepted": bool(a_ok and wr <= 1.05 and c_ok)}
    print("acceptance:", json.dumps(res["acceptance_90"], indent=1))
    print("saved", save("m3_stage_c.json", res))


if __name__ == "__main__":
    main()
