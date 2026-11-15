"""Revision R-R6: change target versus event-mixture head for the calm-condition gain
(protocol revision/protocols/R-R6_target_vs_head.md). Development out-of-fold forecasts; CPU only, no training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.ramp_scoring import ramp_weighted_crps
from src.revision_common import day_bootstrap, save
from src.revision_r1_gate_proper_score import band

ROOT = {"direct": Path("experiments/E4_calib/v1/csi"), "change-target quantile": Path("experiments/E6/v1/E6-QR/csi"), "calm (event mixture)": Path("experiments/E6/v1/E6-S/csi")}
TAU = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
KEYS = ("crps", "tw_ramp", "tw_calm")


def main():
    res = {"protocol": "revision/protocols/R-R6_target_vs_head.md", "per_seed": {}, "contrasts": {}}
    shares = []
    for s in (42, 123, 456):
        sc = {m: {k: [] for k in KEYS} for m in ROOT}
        days, sun = [], []
        for k in range(5):
            tau = TAU[str(k)]["training_labels_fit_days"]
            p = {m: dict(np.load(r / f"seed{s}" / f"fold_{k}" / "preds_val.npz")) for m, r in ROOT.items()}
            ref = p["direct"]
            lo, hi = band(ref, tau)
            for m, d in p.items():
                assert np.array_equal(d["sample_ids"], ref["sample_ids"]) and np.allclose(d["target"], ref["target"])
                for key, v in zip(KEYS, ramp_weighted_crps(d["pred_q"], d["quantile_levels"], d["target"], lo, hi)):
                    sc[m][key].append(v)
            days.append(ref["day"]); sun.append(ref["clear_day"])
        day, sunny = np.concatenate(days), np.concatenate(sun).astype(bool)
        sc = {m: {k: np.concatenate(v) for k, v in d.items()} for m, d in sc.items()}
        res["per_seed"][str(s)] = {m: {"crps": float(v["crps"].mean()), "sunny": float(v["crps"][sunny].mean()), "cloudy": float(v["crps"][~sunny].mean()),
                                       "tw_ramp": float(v["tw_ramp"].mean()), "tw_calm": float(v["tw_calm"].mean())} for m, v in sc.items()}
        r = res["per_seed"][str(s)]
        shares.append((r["direct"]["sunny"] - r["change-target quantile"]["sunny"]) / (r["direct"]["sunny"] - r["calm (event mixture)"]["sunny"]))
        for a, b in (("change-target quantile", "direct"), ("calm (event mixture)", "change-target quantile"), ("calm (event mixture)", "direct")):
            res["contrasts"].setdefault(f"{a} - {b}", {})[str(s)] = {
                "overall": day_bootstrap(sc[a]["crps"] - sc[b]["crps"], day, sunny), "sunny": day_bootstrap((sc[a]["crps"] - sc[b]["crps"])[sunny], day[sunny], sunny[sunny]),
                "cloudy": day_bootstrap((sc[a]["crps"] - sc[b]["crps"])[~sunny], day[~sunny], sunny[~sunny]),
                "tw_ramp": day_bootstrap(sc[a]["tw_ramp"] - sc[b]["tw_ramp"], day, sunny), "tw_calm": day_bootstrap(sc[a]["tw_calm"] - sc[b]["tw_calm"], day, sunny)}
        print(f"seed {s}: " + " | ".join(f"{m}: overall {v['crps']:.4f} sunny {v['sunny']:.4f} cloudy {v['cloudy']:.4f} ramp region {v['tw_ramp']:.4f} calm region {v['tw_calm']:.4f}"
                                         for m, v in r.items()), flush=True)
    share = float(np.mean(shares))
    res["share_of_sunny_gain_recovered_by_change_target"] = {"per_seed": [float(x) for x in shares], "mean": share}
    res["reading"] = "mainly the change target (persistence anchoring)" if share >= 2 / 3 else "mainly the event-mixture head" if share <= 1 / 3 else "both"
    f = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"
    for name, c in res["contrasts"].items():
        for s, v in c.items():
            print(f"{name:48s} seed {s}: overall {f(v['overall'])} sunny {f(v['sunny'])} cloudy {f(v['cloudy'])} ramp {f(v['tw_ramp'])} calm {f(v['tw_calm'])}")
    print(f"share of the calm expert's sunny-day gain recovered by the change-target quantile expert: {[round(x, 2) for x in shares]} mean {share:.2f} => {res['reading']}")
    print("saved", save("r6_target_vs_head.json", res))


if __name__ == "__main__":
    main()
