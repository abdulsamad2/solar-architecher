"""Revision R-R8: evaluate the shuffled-image and single-frame controls of the direct expert
(protocol revision/protocols/R-R8_image_controls.md). Development out-of-fold forecasts; CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.ramp_scoring import ramp_weighted_crps
from src.revision_common import day_bootstrap, save
from src.revision_r1_gate_proper_score import band

TAU = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
KEYS = ("crps", "tw_ramp", "tw_calm")
PAIRS = (("real sequence (E4-B-fit)", "shuffled images"), ("shuffled images", "PV-only (P15)"), ("real sequence (E4-B-fit)", "single frame"),
         ("single frame", "shuffled images"), ("real sequence (E4-B-fit)", "PV-only (P15)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="D:/skippd_hires/R_R8")
    a = ap.parse_args()
    root = {"real sequence (E4-B-fit)": Path("experiments/E4_calib/v1/csi"), "shuffled images": Path(a.runs) / "shuffled" / "csi",
            "single frame": Path(a.runs) / "single" / "csi", "PV-only (P15)": Path("experiments/R_E1/P15/csi")}
    res = {"protocol": "revision/protocols/R-R8_image_controls.md", "per_seed": {}, "contrasts": {p: {} for p in (f"{x} - {y}" for x, y in PAIRS)}}
    for s in (42, 123, 456):
        sc = {m: {k: [] for k in KEYS} for m in root}
        days, sun = [], []
        for k in range(5):
            tau = TAU[str(k)]["training_labels_fit_days"]
            p = {m: dict(np.load(r / f"seed{s}" / f"fold_{k}" / "preds_val.npz")) for m, r in root.items()}
            ref = p["real sequence (E4-B-fit)"]
            lo, hi = band(ref, tau)
            for m, d in p.items():
                assert np.array_equal(d["sample_ids"], ref["sample_ids"]) and np.allclose(d["target"], ref["target"]), f"STOP: samples differ ({m})"
                for key, v in zip(KEYS, ramp_weighted_crps(d["pred_q"], d["quantile_levels"], d["target"], lo, hi)):
                    sc[m][key].append(v)
            days.append(ref["day"]); sun.append(ref["clear_day"])
        day, sunny = np.concatenate(days), np.concatenate(sun).astype(bool)
        sc = {m: {k: np.concatenate(v) for k, v in d.items()} for m, d in sc.items()}
        res["per_seed"][str(s)] = {m: {"crps": float(v["crps"].mean()), "sunny": float(v["crps"][sunny].mean()), "cloudy": float(v["crps"][~sunny].mean()),
                                       "tw_ramp": float(v["tw_ramp"].mean()), "tw_calm": float(v["tw_calm"].mean())} for m, v in sc.items()}
        for x, y in PAIRS:
            res["contrasts"][f"{x} - {y}"][str(s)] = {key: day_bootstrap(sc[x][key] - sc[y][key], day, sunny) for key in KEYS}
        print(f"seed {s}: " + " | ".join(f"{m}: {v['crps']:.4f} (ramp region {v['tw_ramp']:.4f}, calm {v['tw_calm']:.4f})" for m, v in res["per_seed"][str(s)].items()), flush=True)
    f = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"
    for name, c in res["contrasts"].items():
        print(name)
        for s, v in c.items():
            print(f"   seed {s}: overall {f(v['crps'])}  ramp region {f(v['tw_ramp'])}  calm region {f(v['tw_calm'])}")
    neg3 = lambda name, key: bool(all(v[key][2] < 0 for v in res["contrasts"][name].values()))
    mean = lambda name, key: float(np.mean([v[key][0] for v in res["contrasts"][name].values()]))
    info, cap = mean("real sequence (E4-B-fit) - shuffled images", "tw_ramp"), mean("shuffled images - PV-only (P15)", "tw_ramp")
    res["reading"] = {
        "images add information on ramps, capacity held fixed (ramp region CI<0 in 3/3 seeds)": neg3("real sequence (E4-B-fit) - shuffled images", "tw_ramp"),
        "ramp-region difference to PV-only: part from image information (kW)": info, "part from capacity and optimisation (kW)": cap,
        "share from image information": float(info / (info + cap)) if (info + cap) != 0 else None,
        "the 15-min expert uses the sequence beyond the last frame (ramp region CI<0 in 3/3 seeds)": neg3("real sequence (E4-B-fit) - single frame", "tw_ramp"),
        "single frame better than shuffled images on ramps (3/3 seeds)": neg3("single frame - shuffled images", "tw_ramp")}
    print(json.dumps(res["reading"], indent=1))
    print("saved", save("r8_image_controls.json", res))


if __name__ == "__main__":
    main()
