"""Revision R-R11: interval reliability and point error of method v2 on the official test set (descriptive, post hoc;
protocol revision/protocols/R-R11_test_reliability_point_error.md). No training; gates fitted on calibration days
only, as in R-M1 / R-R9. Run: python -m src.revision_r11_test_reliability
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.cal2 import ALPHA, conformity, interval_score
from src.calibrate import base_interval
from src.e8_features import FEAT
from src.final_eval import rolling_conformal_daily
from src.metrics import crps_quantile_interp
from src.ramp_scoring import ramp_weighted_crps, selection_null
from src.revision_c07 import interval_table, past_only_init
from src.revision_common import LEVELS, day_bootstrap, frozen_test_predictions, save, test_context
from src.revision_m1_method_v2 import ROOT, TAU
from src.revision_posthoc_test import sub
from src.revision_r7_forward_v2 import P, apply_mix, fit_mix
from src.revision_z_common import calibrated_intervals, ramp_region

SEEDS, W, CAP = (42, 123, 456), 4620, 30.1
V2, FROZEN = "method v2", ("G3''-F", "G4", "E6-S", "E4-B-fit")
EVAL = Path("D:/skippd_hires/R_M3/eval")
GROUPS = ("overall", "sunny", "cloudy", "ramp", "non-ramp")


def dev_oof_v2():
    """Development out-of-fold quantiles of the routed method v2 (three-seed level-wise average) in the order of the
    frozen development calibration file."""
    ref = np.load("experiments/final/dev_oof_calibration_v1.npz")
    sid, org = ref["sample_ids"], ref["origin_us"].astype(np.int64)
    order = np.argsort(sid, kind="stable")
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = win["sample_ids"][win["eligible_common"]]
    ld = lambda e, s, k, part: dict(np.load(ROOT[e] / f"seed{s}" / f"fold_{k}" / f"preds_{part}.npz"))
    acc, cnt, y = np.zeros((len(sid), len(LEVELS))), np.zeros(len(sid), int), np.full(len(sid), np.nan)
    for s in SEEDS:
        for k in range(5):
            tau = TAU[str(k)]["training_labels_fit_days"]
            cd, cc, cr = (ld(e, s, k, "cal") for e in "DCR")
            p_dc = fit_mix({"D": cd, "C": cc}, tau, ("D", "C"))
            cc2, cr2 = (sub(x, np.isin(x["sample_ids"], common)) for x in (cc, cr))
            assert np.array_equal(cc2["sample_ids"], cr2["sample_ids"])
            p_rc = fit_mix({"R": cr2, "C": cc2}, tau, ("R", "C"))
            vd, vc, vr = (ld(e, s, k, "val") for e in "DCR")
            assert np.array_equal(vd["sample_ids"], vc["sample_ids"])
            q = apply_mix(p_dc, {"D": vd, "C": vc}, tau, ("D", "C"))
            m = np.isin(vc["sample_ids"], common)
            vr2 = sub(vr, np.isin(vr["sample_ids"], common))
            assert np.array_equal(vr2["sample_ids"], vc["sample_ids"][m]), "STOP: 60-min validation samples differ"
            q[m] = apply_mix(p_rc, {"R": vr2, "C": sub(vc, m)}, tau, ("R", "C"))
            pos = order[np.searchsorted(sid[order], vc["sample_ids"])]
            assert np.array_equal(sid[pos], vc["sample_ids"]), "STOP: validation samples not in the development calibration file"
            acc[pos] += q; cnt[pos] += 1; y[pos] = vc["target"]
            print(f"development seed {s} fold {k} done", flush=True)
    assert (cnt == len(SEEDS)).all(), "STOP: not every development sample has three seeds"
    return acc / len(SEEDS), y, org


def point(med, y, masks, pers):
    out = {}
    for g in GROUPS:
        m = masks[g]
        rm, ma = float(np.sqrt(np.mean((med[m] - y[m]) ** 2))), float(np.mean(np.abs(med[m] - y[m])))
        out[g] = {"rmse_kw": rm, "mae_kw": ma, "nrmse_pct_capacity": 100 * rm / CAP, "nmae_pct_capacity": 100 * ma / CAP,
                  "rmse_skill_vs_smart_persistence": 1 - rm / pers[g]["rmse_median_kw"], "mae_skill_vs_smart_persistence": 1 - ma / pers[g]["mae_median_kw"]}
    return out


def main():
    c = test_context()
    y, day, sunny, masks = c["y"], c["day"], c["sunny"], c["masks"]
    pc, lo_r, hi_r = ramp_region(c)
    z = np.load(EVAL / "test_quantiles.npz")
    assert np.array_equal(z["sample_ids"], c["sample_ids"])
    q = {V2: z["method_v2"]} | {m: v for m, v in frozen_test_predictions(c["sample_ids"]).items() if m in FROZEN}
    r9 = json.loads(Path("revision/results/r9_method_v2_official_test.json").read_text())
    assert abs(ramp_weighted_crps(q[V2], LEVELS, y, lo_r, hi_r)[0].mean() - r9["scores"][V2]["crps"]) < 1e-9, "STOP: method v2 (R-R9) not reproduced"
    frozen = json.loads(Path("reports/test_results.json").read_text())
    pers = frozen["scores"]["persistence"]
    for m in FROZEN:  # the frozen medians reproduce their stored point errors
        assert abs(np.sqrt(np.mean((q[m][:, 4] - y) ** 2)) - frozen["scores"][m]["overall"]["rmse_median_kw"]) < 1e-9, f"STOP: {m} point error not reproduced"

    # development scores of method v2
    qd, yd, org_d = dev_oof_v2()
    dev_scores = {nm: conformity(*base_interval(qd, LEVELS, nm), yd) for nm in ALPHA}
    res = {"protocol": "revision/protocols/R-R11_test_reliability_point_error.md", "n": int(len(y)), "W": {V2: W},
           "development_oof_method_v2": {"n": int(len(yd)), "crps_kw": float(crps_quantile_interp(qd, yd, LEVELS).mean()),
                                         "raw_coverage": {nm: float(np.mean(dev_scores[nm] <= 0)) for nm in ALPHA}},
           "intervals": {}, "quantile_reliability": {}, "point": {}, "v2_minus": {}}

    # calibrated intervals
    days = np.unique(day)
    start = {d: c["org"][day == d].min() for d in days}
    ci = {V2: {}}
    for nm, a in ALPHA.items():
        l, h = base_interval(q[V2], LEVELS, nm)
        init = {d: past_only_init(org_d, dev_scores[nm], start[d], W)[0] for d in days}
        ci[V2][nm] = rolling_conformal_daily(c["org"], c["tgt"], day, l, h, y, a, W, init)
    for m in FROZEN:
        ci[m] = calibrated_intervals(c, q[m], m)  # asserts the stored R-C07 coverage
    gm = {g: masks[g] for g in GROUPS}
    for m in q:
        res["intervals"][m] = {nm: {"raw": interval_table(*base_interval(q[m], LEVELS, nm), y, a, gm), "calibrated": interval_table(*ci[m][nm], y, a, gm)} for nm, a in ALPHA.items()}
        res["quantile_reliability"][m] = {g: [float(np.mean(y[masks[g]] <= q[m][masks[g], j])) for j in range(len(LEVELS))] for g in ("overall", "cloudy", "sunny")}
        res["point"][m] = point(q[m][:, 4], y, masks, pers)
        i9 = res["intervals"][m]["90"]
        print(f"{m:10s} 90%: raw {i9['raw']['overall']['coverage']:.3f} | calibrated {i9['calibrated']['overall']['coverage']:.3f} (ramps {i9['calibrated']['ramp']['coverage']:.3f}, "
              f"sunny {i9['calibrated']['sunny']['coverage']:.3f}, cloudy {i9['calibrated']['cloudy']['coverage']:.3f}) width {i9['calibrated']['overall']['width_kw']:.2f} "
              f"score {i9['calibrated']['overall']['interval_score']:.3f} | median RMSE {res['point'][m]['overall']['rmse_kw']:.3f} MAE {res['point'][m]['overall']['mae_kw']:.3f}", flush=True)
    res["levels"] = LEVELS.tolist()
    res["smart_persistence_point"] = {g: {"rmse_kw": pers[g]["rmse_median_kw"], "mae_kw": pers[g]["mae_median_kw"]} for g in GROUPS}

    # ramp coverage of method v2 against the selection reference (as R-Z1 / R-R7)
    nodes = np.sort(np.stack([ci[V2]["90"][0], ci[V2]["80"][0], ci[V2]["50"][0], q[V2][:, 4], ci[V2]["50"][1], ci[V2]["80"][1], ci[V2]["90"][1]], 1), 1)
    res["ramp_coverage_selection_reference"] = {}
    for nm in ALPHA:
        null = selection_null(nodes, P, ci[V2][nm][0], ci[V2][nm][1], lo_r, hi_r, y=y, cap=1.5 * pc, seed=20261117)
        res["ramp_coverage_selection_reference"][nm] = {"ramp_observed": null["observed_ramp_coverage"], "ramp_selection_reference": null["null_ramp_coverage"],
                                                        "reference_range": null["null_ramp_coverage_range"], "gap": null["gap"]}
        print(f"{nm}%: ramp coverage observed {null['observed_ramp_coverage']:.3f}, selection reference {null['null_ramp_coverage']:.3f}")

    # paired differences (descriptive)
    for m in ("G3''-F", "G4"):
        d90 = interval_score(*ci[V2]["90"], y, 0.1) - interval_score(*ci[m]["90"], y, 0.1)
        dae = np.abs(q[V2][:, 4] - y) - np.abs(q[m][:, 4] - y)
        dse = (q[V2][:, 4] - y) ** 2 - (q[m][:, 4] - y) ** 2
        res["v2_minus"][m] = {"interval_score_90": day_bootstrap(d90, day, sunny), "abs_error_median": day_bootstrap(dae, day, sunny), "sq_error_median": day_bootstrap(dse, day, sunny)}
        print(f"v2 - {m}: 90% interval score {res['v2_minus'][m]['interval_score_90']} | abs error {res['v2_minus'][m]['abs_error_median']}")
    np.savez_compressed(EVAL / "test_intervals_v2.npz", sample_ids=c["sample_ids"], y=y, q=q[V2], **{f"lo{nm}": ci[V2][nm][0] for nm in ALPHA}, **{f"hi{nm}": ci[V2][nm][1] for nm in ALPHA})
    print("saved", save("r11_test_reliability_point_error.json", res))


if __name__ == "__main__":
    main()
