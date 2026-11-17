"""Revision R-R14: evaluation of the image controls of the calm-condition expert
(protocol revision/protocols/R-R14_calm_expert_image_controls.md). Development out-of-fold forecasts; CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.e8_features import FEAT
from src.ramp_scoring import ramp_weighted_crps
from src.revision_common import LEVELS, RESULTS, day_bootstrap, save
from src.revision_m1_method_v2 import ROOT, TAU
from src.revision_r1_gate_proper_score import band
from src.revision_r13_explanatory import apply_gate, fit_gate
from src.revision_x2_linear_pool import load

KEYS = ("crps", "tw_ramp", "tw_calm")
SEEDS = (42, 123, 456)
CALM, SHUF, PV = "calm expert (E6-S, real images)", "calm expert, images shuffled (C-shuf)", "calm expert, PV-only encoder (C-pv)"
SINGLE_PAIRS = ((CALM, SHUF), (SHUF, PV), (CALM, PV), ("direct expert (E4-B-fit)", "direct expert, images shuffled (R-R8)"))
f3 = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="D:/skippd_hires/R_R14")
    a = ap.parse_args()
    root = {CALM: ROOT["C"], SHUF: Path(a.runs) / "shuffled" / "csi", PV: Path(a.runs) / "pvonly" / "csi",
            "direct expert (E4-B-fit)": ROOT["D"], "direct expert, images shuffled (R-R8)": Path("D:/skippd_hires/R_R8/shuffled/csi")}
    calm = (CALM, SHUF, PV)
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = win["sample_ids"][win["eligible_common"]]
    m1 = json.loads((RESULTS / "m1_method_v2_development.json").read_text())["per_seed"]
    res = {"protocol": "revision/protocols/R-R14_calm_expert_image_controls.md",
           "single": {"per_seed": {}, "contrasts": {f"{x} - {y}": {} for x, y in SINGLE_PAIRS}},
           "mixture_with_60min_ramp_expert": {"per_seed": {}, "contrasts": {f"mixture with {CALM} - mixture with {m}": {} for m in (SHUF, PV)}}}
    for s in SEEDS:
        sc = {m: {k: [] for k in KEYS} for m in root}
        mq = {m: [] for m in calm}
        days, sun, my, mlo, mhi, mday, msun = [], [], [], [], [], [], []
        for k in range(5):
            tau = TAU[str(k)]["training_labels_fit_days"]
            p = {m: dict(np.load(r / f"seed{s}" / f"fold_{k}" / "preds_val.npz")) for m, r in root.items()}
            ref = p[CALM]
            lo, hi = band(ref, tau)
            for m, d in p.items():
                assert np.array_equal(d["sample_ids"], ref["sample_ids"]) and np.allclose(d["target"], ref["target"]), f"STOP: samples differ ({m})"
                for key, v in zip(KEYS, ramp_weighted_crps(d["pred_q"], d["quantile_levels"], d["target"], lo, hi)):
                    sc[m][key].append(v)
            days.append(ref["day"]); sun.append(ref["clear_day"])
            cr, vr = load(ROOT["R"], s, k, "cal", common), load(ROOT["R"], s, k, "val", common)
            for m in calm:
                cc, vc = load(root[m], s, k, "cal", common), load(root[m], s, k, "val", common)
                assert np.array_equal(cc["sample_ids"], cr["sample_ids"]) and np.array_equal(vc["sample_ids"], vr["sample_ids"]), f"STOP: 60-min samples differ ({m})"
                mq[m].append(apply_gate(fit_gate({"X": cr, "C": cc}, tau, ("X", "C"), (0, 1)), {"X": vr, "C": vc}, tau, ("X", "C")))
            vc = load(root[CALM], s, k, "val", common)
            l2, h2 = band(vc, tau)
            my.append(vc["target"]); mlo.append(l2); mhi.append(h2); mday.append(vc["day"]); msun.append(vc["clear_day"])
            print(f"seed {s} fold {k} done", flush=True)
        cat = np.concatenate
        day, sunny = cat(days), cat(sun).astype(bool)
        sc = {m: {k: cat(v) for k, v in d.items()} for m, d in sc.items()}
        summ = lambda v, sn: {"crps": float(v["crps"].mean()), "sunny": float(v["crps"][sn].mean()), "cloudy": float(v["crps"][~sn].mean()),
                              "tw_ramp": float(v["tw_ramp"].mean()), "tw_calm": float(v["tw_calm"].mean())}
        res["single"]["per_seed"][str(s)] = {m: summ(v, sunny) for m, v in sc.items()}
        for x, y in SINGLE_PAIRS:
            res["single"]["contrasts"][f"{x} - {y}"][str(s)] = {key: day_bootstrap(sc[x][key] - sc[y][key], day, sunny) for key in KEYS}
        y_, lo_, hi_, d_, s_ = cat(my), cat(mlo), cat(mhi), cat(mday), cat(msun).astype(bool)
        ms = {m: dict(zip(KEYS, ramp_weighted_crps(cat(v), LEVELS, y_, lo_, hi_))) for m, v in mq.items()}
        assert abs(ms[CALM]["crps"].mean() - m1[str(s)]["M-RC"]["crps"]) < 1e-9, "STOP: M-RC (R-M1) not reproduced"
        res["mixture_with_60min_ramp_expert"]["per_seed"][str(s)] = {m: summ(v, s_) for m, v in ms.items()}
        for m in (SHUF, PV):
            res["mixture_with_60min_ramp_expert"]["contrasts"][f"mixture with {CALM} - mixture with {m}"][str(s)] = {key: day_bootstrap(ms[CALM][key] - ms[m][key], d_, s_) for key in KEYS}
        print(f"seed {s}: single " + " | ".join(f"{m}: {v['crps']:.4f} (ramp {v['tw_ramp']:.4f}, calm {v['tw_calm']:.4f})" for m, v in res["single"]["per_seed"][str(s)].items()), flush=True)
        print(f"seed {s}: mixture " + " | ".join(f"{m}: {v['crps']:.4f}" for m, v in res["mixture_with_60min_ramp_expert"]["per_seed"][str(s)].items()), flush=True)
    for part in ("single", "mixture_with_60min_ramp_expert"):
        for name, c in res[part]["contrasts"].items():
            print(name)
            for s, v in c.items():
                print(f"   seed {s}: overall {f3(v['crps'])}  ramp region {f3(v['tw_ramp'])}  calm region {f3(v['tw_calm'])}")
    n_neg = lambda c, key: int(sum(v[key][2] < 0 for v in c.values()))
    n_pos = lambda c, key: int(sum(v[key][1] > 0 for v in c.values()))
    cs = res["single"]["contrasts"][f"{CALM} - {SHUF}"]
    mx = res["mixture_with_60min_ramp_expert"]["contrasts"]
    need = [n_neg(mx[f"mixture with {CALM} - mixture with {m}"], "crps") for m in (SHUF, PV)]
    res["reading"] = {
        "calm expert, real - shuffled, calm region: seeds with interval above zero (images hurt)": n_pos(cs, "tw_calm"),
        "calm expert, real - shuffled, calm region: seeds with interval below zero (images help)": n_neg(cs, "tw_calm"),
        "calm expert, real - shuffled, ramp region: seeds with interval below zero (uses image information on ramps)": n_neg(cs, "tw_ramp"),
        "mixture with real-image calm expert better than with C-shuf / C-pv: seeds with interval below zero": need,
        "the calm expert needs the camera as a component (3/3 seeds against both controls)": bool(need[0] == 3 and need[1] == 3)}
    print(json.dumps(res["reading"], indent=1))
    print("saved", save("r14_calm_expert_image_controls.json", res))


if __name__ == "__main__":
    main()
