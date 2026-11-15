"""Revision R-R9: method v2 (gated probability mixtures, rule of R-M1) on the official test set
(protocol revision/protocols/R-R9_method_v2_official_test.md). Post hoc; per-member dump; CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.e8_features import FEAT
from src.ramp_scoring import ramp_weighted_crps
from src.revision_common import LEVELS, day_bootstrap, frozen_test_predictions, save, test_context
from src.revision_m1_method_v2 import ROOT
from src.revision_posthoc_test import TAU, mdict, members, sub
from src.revision_r2_day_level_inference import sign_flip, t_interval
from src.revision_r7_forward_v2 import apply_mix, fit_mix
from src.revision_z_common import ramp_region

KEYS = ("crps", "tw_ramp", "tw_calm")


def main():
    c = test_context()
    h60, y, day, sunny = c["h60"], c["y"], c["day"], c["sunny"]
    _, lo, hi = ramp_region(c)
    d, r, a = members("E4-B-fit"), members("E6-S"), members("A2")
    common = np.load(FEAT / "e8_windows_v1.npz")
    common = common["sample_ids"][common["eligible_common"]]
    cal = lambda e, s, k: dict(np.load(ROOT[e] / f"seed{s}" / f"fold_{k}" / "preds_cal.npz"))
    acc = []
    for i, (s, k) in enumerate(r["members"]):
        tau = TAU[str(k)]["training_labels_fit_days"]
        cd, cc, cr = cal("D", s, k), cal("C", s, k), cal("R", s, k)
        assert np.array_equal(cd["sample_ids"], cc["sample_ids"])
        p_dc = fit_mix({"D": cd, "C": cc}, tau, ("D", "C"))
        keep_c, keep_r = np.isin(cc["sample_ids"], common), np.isin(cr["sample_ids"], common)
        cc2, cr2 = sub(cc, keep_c), sub(cr, keep_r)
        assert np.array_equal(cc2["sample_ids"], cr2["sample_ids"])
        p_rc = fit_mix({"R": cr2, "C": cc2}, tau, ("R", "C"))
        md, mr, ma = mdict(d, i), mdict(r, i), mdict(a, i)
        q = apply_mix(p_dc, {"D": md, "C": mr}, tau, ("D", "C"))
        q[h60] = apply_mix(p_rc, {"R": sub(ma, h60), "C": sub(mr, h60)}, tau, ("R", "C"))
        acc.append(q)
        print(f"member {i + 1}/15 (seed {s}, fold {k}) done", flush=True)
    fz = frozen_test_predictions(c["sample_ids"])
    qs = {"method v2": np.mean(acc, 0), "G3''-F": fz["G3''-F"], "G4": fz["G4"], "E6-S": fz["E6-S"], "E4-B-fit": fz["E4-B-fit"],
          "PV-only P15": np.nanmean(members("P15")["pred_q"], 0)}
    sc = {m: dict(zip(KEYS, ramp_weighted_crps(v, LEVELS, y, lo, hi))) for m, v in qs.items()}
    z2 = json.loads(Path("revision/results/z2_ramp_weighted_crps.json").read_text())["scores"]
    for m in ("G3''-F", "G4", "E6-S", "E4-B-fit"):
        assert abs(sc[m]["crps"].mean() - z2[m]["crps"]) < 1e-9, f"STOP: {m} not reproduced"
    ramp = c["ramp"]
    res = {"protocol": "revision/protocols/R-R9_method_v2_official_test.md", "n": int(len(y)),
           "scores": {m: {kk: float(v[kk].mean()) for kk in KEYS} | {"sunny": float(v["crps"][sunny].mean()), "cloudy": float(v["crps"][~sunny].mean()),
                                                                       "ramp_outcomes": float(v["crps"][ramp].mean()), "nonramp_outcomes": float(v["crps"][~ramp].mean())}
                      for m, v in sc.items()}, "v2_minus": {}}
    days, inv = np.unique(day, return_inverse=True)
    sday = np.zeros(len(days), bool)
    sday[inv] = sunny
    f = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"
    for m in qs:
        if m == "method v2":
            continue
        dd = {kk: sc["method v2"][kk] - sc[m][kk] for kk in KEYS}
        S, n = np.bincount(inv, weights=dd["crps"]), np.bincount(inv).astype(float)
        rr = {kk: day_bootstrap(dd[kk], day, sunny) for kk in KEYS} | {"sign_flip_p": sign_flip(S, n)[0], "t_interval_day_means": t_interval(S / n, sday)[0],
                                                                       "days_v2_better": int((S < 0).sum()), "skill": float(1 - sc["method v2"]["crps"].mean() / sc[m]["crps"].mean())}
        rr["better"] = bool(rr["crps"][2] < 0 and rr["sign_flip_p"] < 0.05)
        res["v2_minus"][m] = rr
        print(f"v2 - {m:12s} CRPS {res['scores'][m]['crps']:.4f}: {f(rr['crps'])} skill {rr['skill']:+.1%} | ramp region {f(rr['tw_ramp'])} | calm {f(rr['tw_calm'])} | "
              f"sign-flip p {rr['sign_flip_p']:.5f} ({rr['days_v2_better']}/20 days) => {'BETTER' if rr['better'] else 'not shown'}")
    refs = json.loads(Path("revision/results/r3_probabilistic_references.json").read_text())["references"]
    v2 = res["scores"]["method v2"]["crps"]
    res["skill_vs_naive_references"] = {k: float(1 - v2 / refs[k]["crps"]) for k in ("SP-dressed adaptive", "PeEn-120", "PeEn-30", "SP-dressed static", "CH-PeEn")}
    res["skill_vs_official_smart_persistence_point"] = float(1 - v2 / 1.707)
    print(f"method v2 CRPS {v2:.4f} (sunny {res['scores']['method v2']['sunny']:.4f}, cloudy {res['scores']['method v2']['cloudy']:.4f}); skill vs naive references "
          f"{ {k: round(v, 3) for k, v in res['skill_vs_naive_references'].items()} }")
    print("saved", save("r9_method_v2_official_test.json", res))


if __name__ == "__main__":
    main()
