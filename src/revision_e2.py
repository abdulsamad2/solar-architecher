"""Revision E2 (development): gate-feature and constant-weight ablation on frozen A2 + E6-S predictions.

Protocol: revision/protocols/R-E2_gate_ablation.md. Every method is fitted per seed x fold on calibration days only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from src.e8_features import FEAT, verify_frozen
from src.final_eval import day_bootstrap
from src.gate2_report import ramp_labels, strata
from src.gate7b import BlendCRPS, apply_gate, grid_quantiles, raw_features, select_and_fit
from src.metrics import crps_quantile_interp
from src.revision_common import save

A2 = Path("experiments/E9/v1/A2/csi")
E6S = Path("experiments/E6/v1/E6-S/csi")
E4B = Path("experiments/E4_calib/v1/csi")
G7B = Path("experiments/E7b_gate/v1")
METHODS = ("A2", "E6-S", "B50", "BC", "Gg", "Gd", "G3''", "G4")
STRATA = ("overall", "sunny", "cloudy", "ramp", "non-ramp")


def load(root, s, k, split, keep=None):
    d = dict(np.load(root / f"seed{s}" / f"fold_{k}" / f"preds_{split}.npz"))
    if keep is None:
        return d
    m = np.array([i in keep for i in d["sample_ids"]])
    return {kk: (v[m] if np.ndim(v) and len(v) == len(m) else v) for kk, v in d.items()}


def constant_weight(dc, rc) -> float:
    lv = rc["quantile_levels"]
    f = BlendCRPS(grid_quantiles(dc["pred_q"], lv), grid_quantiles(rc["pred_q"], lv), rc["target"])
    n = len(rc["target"])
    obj = lambda c: float(f.per_sample(torch.full((n,), float(c), dtype=torch.float64)).mean())
    r = minimize_scalar(obj, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-6})
    return float(r.x)


def main():
    verify_frozen()
    frozen = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = set(win["sample_ids"][win["eligible_common"]].tolist())
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    rl = ramp_labels("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz", folds_doc)
    recorded = json.loads(Path("reports/e9_gate.json").read_text())
    res = {"protocol": "revision/protocols/R-E2_gate_ablation.md", "per_seed": {}, "fits": {}, "weight_bins": {}}
    crps_all = {}
    for s in (42, 123, 456):
        qs = {m: [] for m in METHODS}
        metas, wts, feats = [], [], []
        for k in range(5):
            tau = frozen[str(k)]["training_labels_fit_days"]
            rc, rv = load(E6S, s, k, "cal", common), load(E6S, s, k, "val", common)
            dc, dv = load(A2, s, k, "cal", common), load(A2, s, k, "val", common)
            assert np.array_equal(dc["sample_ids"], rc["sample_ids"]) and np.array_equal(dv["sample_ids"], rv["sample_ids"])
            lv = rc["quantile_levels"]
            fc, fv = raw_features(dc, rc, tau), raw_features(dv, rv, tau)
            order = np.argsort(rc["origin_time_us"], kind="stable")
            fits = {"Gg": select_and_fit(fc, ("g",), dc["pred_q"], rc["pred_q"], rc["target"], lv, order),
                    "Gd": select_and_fit(fc, ("d",), dc["pred_q"], rc["pred_q"], rc["target"], lv, order),
                    "G3''": select_and_fit(fc, ("g", "d"), dc["pred_q"], rc["pred_q"], rc["target"], lv, order)}
            wc = constant_weight(dc, rc)
            res["fits"][f"{s}/{k}"] = {"BC_w": wc} | {n: {kk: p[kk] for kk in ("lambda", "b0", "beta")} for n, p in fits.items()}
            w = {"B50": np.full(len(rv["target"]), 0.5), "BC": np.full(len(rv["target"]), wc)}
            w |= {n: apply_gate(p, fv) for n, p in fits.items()}
            for n, ww in w.items():
                qs[n].append(ww[:, None] * dv["pred_q"] + (1 - ww[:, None]) * rv["pred_q"])
            qs["A2"].append(dv["pred_q"])
            qs["E6-S"].append(rv["pred_q"])
            # G4 reference on the same samples (frozen E7b gate, E4-B-fit + E6-S)
            ev = load(E4B, s, k, "val", common)
            rv4 = load(E6S, s, k, "val", common)
            assert np.array_equal(ev["sample_ids"], rv["sample_ids"])
            p4 = json.loads((G7B / f"seed{s}" / f"fold_{k}" / "params.json").read_text())["G4"]
            w4 = apply_gate(p4, raw_features(ev, rv4, tau))
            qs["G4"].append(w4[:, None] * ev["pred_q"] + (1 - w4[:, None]) * rv4["pred_q"])
            metas.append(rv)
            wts.append(w["G3''"])
            feats.append(fv)
        cat = lambda key: np.concatenate([v[key] for v in metas])
        m = {key: cat(key) for key in ("sample_ids", "target", "clear_day", "fold", "day")}
        st = strata(m, rl["is_ramp"])
        lv = metas[0]["quantile_levels"]
        crps = {n: crps_quantile_interp(np.concatenate(qs[n]), m["target"], lv) for n in METHODS}
        for g in ("overall", "ramp", "sunny", "cloudy"):
            want = recorded["per_seed"][str(s)]["raw_crps"]["G3''"][g]
            assert abs(float(crps["G3''"][st[g]].mean()) - want) < 1e-9, f"STOP: G3'' not reproduced ({s}, {g})"
        crps_all[s] = crps
        out = {"crps": {n: {g: float(crps[n][st[g]].mean()) for g in STRATA} for n in METHODS}, "diff_G3pp_minus": {}}
        for n in METHODS:
            if n == "G3''":
                continue
            out["diff_G3pp_minus"][n] = {g: day_bootstrap((crps["G3''"] - crps[n])[st[g]], m["day"][st[g]], m["clear_day"][st[g]])
                                         for g in STRATA}
        res["per_seed"][str(s)] = out
        # mean G3'' weight on A2 by decile of forecast-time g and d (validation)
        wa = np.concatenate(wts)
        for fname in ("g", "d"):
            x = np.concatenate([f[fname] for f in feats])
            edges = np.quantile(x, np.linspace(0, 1, 11))
            b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 9)
            res["weight_bins"].setdefault(fname, {})[str(s)] = {
                "edges": edges.tolist(), "mean_w": [float(wa[b == i].mean()) for i in range(10)],
                "ramp_share": [float(st["ramp"][b == i].mean()) for i in range(10)]}
        print(f"seed {s}: G3'' reproduced; " + "  ".join(f"{n} {out['crps'][n]['overall']:.4f}/{out['crps'][n]['ramp']:.3f}" for n in METHODS))
    res["seed_mean"] = {n: {g: {"mean": float(np.mean([res["per_seed"][str(s)]["crps"][n][g] for s in (42, 123, 456)])),
                               "sd": float(np.std([res["per_seed"][str(s)]["crps"][n][g] for s in (42, 123, 456)], ddof=1))}
                           for g in STRATA} for n in METHODS}
    verdict = {}
    for n in ("BC", "B50", "Gg", "Gd"):
        ramp_ok = [res["per_seed"][str(s)]["diff_G3pp_minus"][n]["ramp"][2] < 0 for s in (42, 123, 456)]
        over_ok = [res["per_seed"][str(s)]["diff_G3pp_minus"][n]["overall"][0] <= 0.005 for s in (42, 123, 456)]
        verdict[f"G3'' better than {n} on ramps (CI<0) and overall <= +0.005"] = f"{sum(a and b for a, b in zip(ramp_ok, over_ok))}/3"
    res["decision"] = verdict
    p = save("e2_gate_ablation_dev.json", res)
    print("saved", p, verdict)


if __name__ == "__main__":
    main()
