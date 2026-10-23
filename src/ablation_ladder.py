"""Ablation ladder on DEVELOPMENT out-of-fold predictions (post hoc, descriptive; no training, no test data).

Where does G3''-F's gain come from? Each rung adds one ingredient; every number is held-out (a sample is predicted
only by members that never trained on it). "Single member" = per-sample mean CRPS over its 3 out-of-fold members
(the expected score of one member); "3-member" = level-wise quantile average of those members. The 15-member test
ensemble cannot be formed out-of-fold on development data.
Output: reports/ablation_ladder.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.final_eval import FOLDS, SEEDS, blend, gate_params, day_bootstrap
from src.metrics import crps_quantile_interp

LV = np.array([.05, .1, .2, .3, .5, .7, .8, .9, .95])
ROOTS = {"kW single (E4-A)": "experiments/E4_quantile/v1/kw", "K single (E4-B)": "experiments/E4_quantile/v1/csi",
         "E4-B-fit": "experiments/E4_calib/v1/csi", "E6-S": "experiments/E6/v1/E6-S/csi"}


def load(root, s, k):
    return dict(np.load(f"{root}/seed{s}/fold_{k}/preds_val.npz"))


def main():
    z = np.load("experiments/phaseC/oof_streams.npz")
    sid = z["sample_ids"]
    pos = {s: i for i, s in enumerate(sid)}
    n = len(sid)
    y = z["y"].astype(float)
    crps_single = {m: np.zeros(n) for m in ROOTS}
    q3 = {m: np.zeros((n, 9)) for m in ("E4-B-fit", "E6-S", "avg50 E4+E6", "G4 (gated, 15-min)", "G3 (gated, 15-min)")}
    cnt = np.zeros(n)
    tau = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    for k in FOLDS:
        for s in SEEDS:
            d = {m: load(r, s, k) for m, r in ROOTS.items()}
            ids = d["E4-B-fit"]["sample_ids"]
            for m in ROOTS:
                assert np.array_equal(d[m]["sample_ids"], ids), f"STOP: sample order {m}"
            idx = np.array([pos[i] for i in ids])
            assert np.allclose(d["E4-B-fit"]["target"], y[idx])
            for m in ROOTS:
                crps_single[m][idx] += crps_quantile_interp(d[m]["pred_q"], y[idx], LV)
            e4, e6 = d["E4-B-fit"], d["E6-S"]
            t_k = tau[str(k)]["training_labels_fit_days"]
            q3["E4-B-fit"][idx] += e4["pred_q"]
            q3["E6-S"][idx] += e6["pred_q"]
            q3["avg50 E4+E6"][idx] += 0.5 * (e4["pred_q"] + e6["pred_q"])
            q3["G4 (gated, 15-min)"][idx] += blend(e4, e6, gate_params("G4", s, k), t_k)
            q3["G3 (gated, 15-min)"][idx] += blend(e4, e6, gate_params("G3", s, k), t_k)
            cnt[idx] += 1
    assert (cnt == 3).all()
    crps = {f"{m}: 1 member": crps_single[m] / 3 for m in ROOTS}
    for m, a in q3.items():
        crps[f"{m}: 3-member" if m in ("E4-B-fit", "E6-S") else f"{m}: 3-member"] = crps_quantile_interp(a / 3, y, LV)
    dq = np.abs(z["q_E4"] - q3["E4-B-fit"] / 3).max(1)  # recomputed vs stored predictions (amendment-2 criterion)
    print(f"stream check: max {dq.max():.2e}, mean {dq.mean():.2e}, share<=1e-4 {np.mean(dq <= 1e-4):.4f}")
    assert dq.max() <= 5e-3 and np.mean(dq <= 1e-4) >= 0.999, "STOP: OOF stream mismatch"
    crps["G3''-F (+60-min 3D-CNN expert, routed): 3-member"] = crps_quantile_interp(z["q_G3ppF"], y, LV)
    pers = np.load("experiments/E4_calib/v1/csi/seed42/fold_0/preds_val.npz")  # persistence is per-sample, member-independent
    p_all = np.zeros(n)
    for k in FOLDS:
        d = load(ROOTS["E4-B-fit"], 42, k)
        p_all[[pos[i] for i in d["sample_ids"]]] = np.abs(d["persistence"] - d["target"])
    crps = {"smart persistence": p_all, **crps}
    ramp, sunny, day = z["ramp"].astype(bool), z["sunny"].astype(bool), z["day"]
    strata = {"overall": np.ones(n, bool), "sunny": sunny, "cloudy": ~sunny, "ramp": ramp, "non-ramp": ~ramp}
    table = {m: {g: float(c[v].mean()) for g, v in strata.items()} for m, c in crps.items()}
    steps = [("K target vs kW target (1 member)", "K single (E4-B): 1 member", "kW single (E4-A): 1 member"),
             ("3-member ensemble vs 1 member (E4-B-fit)", "E4-B-fit: 3-member", "E4-B-fit: 1 member"),
             ("add calm expert, plain 50/50 average (no gate)", "avg50 E4+E6: 3-member", "E4-B-fit: 3-member"),
             ("gate vs plain 50/50 average (G3)", "G3 (gated, 15-min): 3-member", "avg50 E4+E6: 3-member"),
             ("gate vs plain 50/50 average (G4)", "G4 (gated, 15-min): 3-member", "avg50 E4+E6: 3-member"),
             ("60-min 3D-CNN ramp expert (G3''-F vs G3)", "G3''-F (+60-min 3D-CNN expert, routed): 3-member", "G3 (gated, 15-min): 3-member")]
    boot = {}
    for name, a, b in steps:
        boot[name] = {g: day_bootstrap(crps[a][v] - crps[b][v], day[v], sunny[v]) for g, v in strata.items()}
    out = {"note": __doc__, "n": n, "crps": table, "steps_vs_previous": boot}
    Path("reports/ablation_ladder.json").write_text(json.dumps(out, indent=1))
    print(f"{'rung':58s} overall  sunny  cloudy   ramp  non-ramp")
    for m, r in table.items():
        print(f"{m:58s} {r['overall']:.3f}  {r['sunny']:.3f}  {r['cloudy']:.3f}  {r['ramp']:.3f}  {r['non-ramp']:.3f}")
    for name, b in boot.items():
        print(name, {g: [round(x, 3) for x in v] for g, v in b.items() if g in ("overall", "ramp", "sunny")})


if __name__ == "__main__":
    main()
