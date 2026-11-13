"""Revision E1 (PV-only) and E4 (history x encoder): development out-of-fold comparisons.

Protocols: revision/protocols/R-E1_pv_only.md, R-E4_history_encoder.md. Per seed: the 5 folds' validation
predictions are concatenated; paired day-block bootstrap (stratified sunny/cloudy) on identical samples.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.e8_features import FEAT
from src.final_eval import day_bootstrap
from src.gate2_report import ramp_labels, strata
from src.gate7b import apply_gate, k_quantiles, raw_features
from src.metrics import crps_quantile_interp
from src.revision_c18 import ramp_metrics
from src.revision_common import LEVELS, save
from src.train import build_model

ROOT = {"E4-B-fit": "experiments/E4_calib/v1/csi", "E6-S": "experiments/E6/v1/E6-S/csi", "A2": "experiments/E9/v1/A2/csi",
        "D0": "experiments/E8/v1/D0/csi", "D2": "experiments/E8/v1/D2/csi", "A2-H15": "experiments/R_E4/A2_H15/csi",
        "P15": "experiments/R_E1/P15/csi", "P60": "experiments/R_E1/P60/csi"}
STRATA = ("overall", "sunny", "cloudy", "ramp", "non-ramp")
TAU = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())


def load(name, s, k, keep=None):
    d = dict(np.load(Path(ROOT[name]) / f"seed{s}" / f"fold_{k}" / "preds_val.npz"))
    if keep is None:
        return d
    m = np.isin(d["sample_ids"], keep)
    return {kk: (v[m] if np.ndim(v) and len(v) == len(m) else v) for kk, v in d.items()}


def gated(direct, resid, params, tau):
    w = apply_gate(params, raw_features(direct, resid, tau))
    return w[:, None] * direct["pred_q"] + (1 - w[:, None]) * resid["pred_q"]


def collect(names, s, keep, folds_doc):
    q = {n: [] for n in names}
    meta, rm = [], {n: [] for n in names}
    for k in range(5):
        tau = TAU[str(k)]["training_labels_fit_days"]
        base = load("E6-S", s, k, keep)
        for n in names:
            if n == "G3":
                p = json.loads(Path(f"experiments/E7b_gate/v1/seed{s}/fold_{k}/params.json").read_text())["G3"]
                d = load("E4-B-fit", s, k, keep)
                qq = gated(d, base, p, tau)
            elif n == "G3''":
                p = json.loads(Path(f"experiments/E9/v1/G3pp_params/seed{s}/fold_{k}/params.json").read_text())
                d = load("A2", s, k, keep)
                qq = gated(d, base, p, tau)
            else:
                d = load(n, s, k, keep)
                qq = d["pred_q"]
            assert np.array_equal(d["sample_ids"], base["sample_ids"]), f"STOP: sample order differs ({n}, {s}, {k})"
            q[n].append(qq)
            kn = base["pv_now_kw"] / np.maximum(base["pclear_now_kw"], 1.0)
            kt = base["target"] / np.maximum(base["pclear_target_kw"], 1.0)
            rm[n].append((qq / np.maximum(base["pclear_target_kw"], 1.0)[:, None], kn, kt, np.full(len(kn), tau)))
        meta.append(base)
    cat = lambda key: np.concatenate([v[key] for v in meta])
    m = {key: cat(key) for key in ("sample_ids", "target", "clear_day", "fold", "day")}
    return {n: np.concatenate(v) for n, v in q.items()}, m, rm


def compare(names, pairs, keep, rl, folds_doc, label):
    out = {"per_seed": {}, "pairs": {}, "ramp_metrics": {}}
    for s in (42, 123, 456):
        q, m, rm = collect(names, s, keep, folds_doc)
        st = strata(m, rl["is_ramp"])
        y = m["target"]
        crps = {n: crps_quantile_interp(q[n], y, LEVELS) for n in names}
        med = {n: q[n][:, 4] for n in names}
        out["per_seed"][str(s)] = {n: {g: {"crps": float(crps[n][st[g]].mean()), "rmse": float(np.sqrt(np.mean((med[n][st[g]] - y[st[g]]) ** 2))),
                                           "mae": float(np.abs(med[n][st[g]] - y[st[g]]).mean())} for g in STRATA} for n in names}
        out["ramp_metrics"][str(s)] = {n: ramp_metrics(*[np.concatenate([x[i] for x in rm[n]]) for i in range(4)]) for n in names}
        for a, b in pairs:
            out["pairs"].setdefault(f"{a} - {b}", {})[str(s)] = {
                g: day_bootstrap((crps[a] - crps[b])[st[g]], m["day"][st[g]], m["clear_day"][st[g]]) for g in STRATA}
        print(label, s, " ".join(f"{n} {out['per_seed'][str(s)][n]['overall']['crps']:.3f}/{out['per_seed'][str(s)][n]['ramp']['crps']:.3f}" for n in names))
    out["seed_mean"] = {n: {g: float(np.mean([out["per_seed"][str(s)][n][g]["crps"] for s in (42, 123, 456)])) for g in STRATA} for n in names}
    return out


def interaction(e4_pairs_crps):
    return None


def main():
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    rl = ramp_labels("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz", folds_doc)
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = win["sample_ids"][win["eligible_common"]]
    res = {"protocols": ["revision/protocols/R-E1_pv_only.md", "revision/protocols/R-E4_history_encoder.md"]}
    res["E1_full"] = compare(("P15", "E4-B-fit", "G3"), [("P15", "E4-B-fit"), ("P15", "G3")], None, rl, folds_doc, "E1 full")
    res["E1_common"] = compare(("P60", "P15", "A2", "G3''"), [("P60", "A2"), ("P60", "G3''"), ("P15", "G3''"), ("P60", "P15")],
                               common, rl, folds_doc, "E1 common")
    res["E4_common"] = compare(("D0", "D2", "A2-H15", "A2"),
                               [("D2", "D0"), ("A2", "A2-H15"), ("A2-H15", "D0"), ("A2", "D2")], common, rl, folds_doc, "E4")
    # interaction per seed: (3D60 - 3D15) - (2D60 - 2D15), day bootstrap on the per-sample double difference
    inter = {}
    for s in (42, 123, 456):
        q, m, _ = collect(("D0", "D2", "A2-H15", "A2"), s, common, folds_doc)
        st = strata(m, rl["is_ramp"])
        c = {n: crps_quantile_interp(q[n], m["target"], LEVELS) for n in q}
        dd = (c["A2"] - c["A2-H15"]) - (c["D2"] - c["D0"])
        inter[str(s)] = {g: day_bootstrap(dd[st[g]], m["day"][st[g]], m["clear_day"][st[g]]) for g in STRATA}
    res["E4_common"]["interaction"] = inter
    # parameter counts
    res["parameters"] = {}
    for n in ("E4-B-fit", "E6-S", "A2", "D0", "D2", "A2-H15", "P15", "P60"):
        cfg = json.loads((Path(ROOT[n]) / "seed42" / "config.json").read_text())
        res["parameters"][n] = int(sum(p.numel() for p in build_model(cfg).parameters()))
    # verdicts per protocol rules
    def ci_count(key, g, pop, below=True):
        r = res[pop]["pairs"][key]
        return sum((r[s][g][2] < 0) if below else (r[s][g][1] > 0) for s in r)
    res["verdicts"] = {
        "E4 history effect within 3D (A2 - A2-H15): ramp CI<0 seeds": ci_count("A2 - A2-H15", "ramp", "E4_common"),
        "E4 history effect within 3D (A2 - A2-H15): overall CI<0 seeds": ci_count("A2 - A2-H15", "overall", "E4_common"),
        "E4 history effect within 2D (D2 - D0): ramp CI<0 seeds": ci_count("D2 - D0", "ramp", "E4_common"),
        "E4 encoder effect at 60 min (A2 - D2): ramp CI<0 seeds": ci_count("A2 - D2", "ramp", "E4_common"),
        "E4 encoder effect at 60 min (A2 - D2): overall CI<0 seeds": ci_count("A2 - D2", "overall", "E4_common"),
        "E4 encoder effect at 15 min (A2-H15 - D0): ramp CI<0 seeds": ci_count("A2-H15 - D0", "ramp", "E4_common"),
        "E1 P15 worse than E4-B-fit overall (CI>0) seeds": ci_count("P15 - E4-B-fit", "overall", "E1_full", below=False),
        "E1 P15 worse than E4-B-fit ramp (CI>0) seeds": ci_count("P15 - E4-B-fit", "ramp", "E1_full", below=False),
        "E1 P60 worse than A2 overall (CI>0) seeds": ci_count("P60 - A2", "overall", "E1_common", below=False),
        "E1 P60 worse than A2 ramp (CI>0) seeds": ci_count("P60 - A2", "ramp", "E1_common", below=False),
    }
    p = save("e1_e4_dev.json", res)
    print("saved", p)
    print(json.dumps(res["verdicts"], indent=1))
    print("params", res["parameters"])
    for pop in ("E1_full", "E1_common", "E4_common"):
        print(pop, {n: {g: round(v, 3) for g, v in d.items() if g in ("overall", "ramp", "sunny")} for n, d in res[pop]["seed_mean"].items()})


if __name__ == "__main__":
    main()
