"""Revision R-Y3: operational value - upward reserve at 95% reliability and the economic value of ramp warnings
(protocol revision/protocols/R-Y3_operational_value.md). Development for calibration scores and thresholds, post hoc test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.cal2 import conformity
from src.calibrate import base_interval
from src.clear_sky import ClearSkyPV
from src.events import implied_event_probability
from src.final_eval import blend, gate_params, rolling_conformal_daily
from src.revision_c07 import past_only_init
from src.revision_common import LEVELS, day_bootstrap, frozen_test_predictions, save, test_context
from src.revision_posthoc_test import test_k
from src.revision_x2_linear_pool import TAU
from src.revision_y2_conditional_conformal import FOLDS, PP, SEEDS, _sub, _val
from src.revision_y2_conditional_conformal import development as g3ppf_development

MODELS = (PP, "E4-B-fit", "E6-S", "G4")
MED = list(LEVELS).index(0.5)
ALPHAS = np.round(np.arange(0.01, 1.0, 0.01), 2)


def dev_quantiles():
    """Out-of-fold development quantiles (3 members per sample) of E4-B-fit, E6-S and G4, plus targets and E6-S g."""
    acc, cnt, meta = {m: {} for m in MODELS[1:]}, {}, {}
    gacc = {}
    for s in SEEDS:
        for k in FOLDS:
            tau = TAU[str(k)]["training_labels_fit_days"]
            e4, e6 = _val("E4-B-fit", s, k), _val("E6-S", s, k)
            assert np.array_equal(e4["sample_ids"], e6["sample_ids"])
            q = {"E4-B-fit": e4["pred_q"], "E6-S": e6["pred_q"], "G4": blend(e4, e6, gate_params("G4", s, k), tau)}
            kn = e6["pv_now_kw"] / np.maximum(e6["pclear_now_kw"], 1.0)
            qk = e6["pred_q"] / np.maximum(e6["pclear_target_kw"], 1.0)[:, None]
            g = np.concatenate([implied_event_probability(qk[i:i + 8192], LEVELS, kn[i:i + 8192], tau) for i in range(0, len(kn), 8192)])
            kt = e6["target"] / np.maximum(e6["pclear_target_kw"], 1.0)
            for i, sid in enumerate(e6["sample_ids"].tolist()):
                for m in q:
                    acc[m][sid] = acc[m].get(sid, 0) + q[m][i]
                gacc[sid] = gacc.get(sid, 0) + g[i]
                cnt[sid] = cnt.get(sid, 0) + 1
                meta[sid] = (e6["target"][i], e6["persistence"][i], bool(abs(kt[i] - kn[i]) >= tau))
    return acc, gacc, cnt, meta


def value_curve(warn, lab):
    """Relative economic value V(alpha) of binary warnings (Richardson 2000), loss L = 1, cost C = alpha."""
    s = lab.mean()
    h, f = (warn & lab).mean(), (warn & ~lab).mean()
    miss = (~warn & lab).mean()
    out = []
    for a in ALPHAS:
        e_f = a * (h + f) + miss
        e_c, e_p = min(a, s), a * s
        out.append(float((e_c - e_f) / (e_c - e_p)) if e_c > e_p else 0.0)
    return np.array(out)


def main():
    cal = np.load("experiments/final/dev_oof_calibration_v1.npz")
    Wm = json.loads(Path("experiments/final/dev_oof_calibration_v1.json").read_text())["W"]
    acc, gacc, cnt, meta = dev_quantiles()
    sid_all = cal["sample_ids"].tolist()
    assert all(cnt[x] == 3 for x in sid_all), "STOP: not 3 out-of-fold members"
    y_all = np.array([meta[x][0] for x in sid_all], float)
    org_all = cal["origin_us"].astype(np.int64)
    dev = {m: np.stack([acc[m][x] / 3 for x in sid_all]) for m in MODELS[1:]}
    repro = {}
    for m in MODELS[1:]:
        diff = np.abs(conformity(*base_interval(dev[m], LEVELS, "90"), y_all) - cal[f"{m}|90"])
        repro[m] = {"max": float(diff.max()), "share_within_1e-4": float((diff <= 1e-4).mean())}
        assert (diff <= 1e-4).mean() >= 0.999 and diff.max() <= 5e-3, f"STOP: dev {m} not reproduced ({diff.max()})"
    g3 = g3ppf_development()  # G3''-F (reproduced in R-Y2; excludes 140 samples without stored A2 predictions)
    repro[PP] = g3["repro_max_abs_kw"]["90"]
    dev_sets = {m: (dev[m], y_all, org_all) for m in MODELS[1:]} | {PP: (g3["q"], g3["y"], g3["org"])}
    print("dev reproduction:", repro, flush=True)

    c = test_context()
    q = frozen_test_predictions(c["sample_ids"])
    y, day, org, tgt, sunny = c["y"], c["day"], c["org"], c["tgt"], c["sunny"]
    from src.baselines import official_persistence
    pers = official_persistence(c["ds"].origin_time.astype(object), c["pv"][c["ds"].hist_rows[:, -1]])
    lines = Path("paper/data/t4_main_test.csv").read_text(encoding="utf-8").splitlines()
    ci = lines[0].split(",").index("crps_kw")
    t4 = {row.split(",")[0]: float(row.split(",")[ci]) for row in lines[1:]}
    days = np.unique(day)
    start = {d: org[day == d].min() for d in days}
    res = {"protocol": "revision/protocols/R-Y3_operational_value.md", "dev_reproduction": repro, "reserve": {}, "warnings": {}}
    # --- reserve: dispatch at the median, upward reserve r = median - calibrated one-sided 95% lower bound
    reserves, rel = {}, {}
    masks = {"overall": np.ones(len(y), bool), "cloudy": ~sunny, "sunny": sunny, "ramp": c["ramp"]}
    for m in MODELS:
        dq, dy, dorg = dev_sets[m]
        dlo = base_interval(dq, LEVELS, "90")[0]
        dsc = dlo - dy  # one-sided score
        lo = base_interval(q[m], LEVELS, "90")[0]
        init = {d: past_only_init(dorg, dsc, start[d], Wm[m])[0] for d in days}
        clo, _ = rolling_conformal_daily(org, tgt, day, lo, lo + 1e6, y, 0.05, Wm[m], init)
        med = q[m][:, MED]
        rr = np.maximum(med - clo, 0)
        static = float(np.quantile(dq[:, MED] - dy, 0.95))
        reserves[m], rel[m] = rr, y >= med - rr
        res["reserve"][m] = {g: {"reliability": float(rel[m][v].mean()), "mean_reserve_kw": float(rr[v].mean()),
                                 "p95_reserve_kw": float(np.quantile(rr[v], 0.95)),
                                 "unserved_kw": float(np.maximum(med[v] - rr[v] - y[v], 0).mean()),
                                 "static_reserve_kw": static, "static_reliability": float((y[v] >= med[v] - static).mean())}
                             for g, v in masks.items()}
    dpers = np.array([meta[x][1] for x in sid_all], float)
    ps = float(np.quantile(dpers - y_all, 0.95))
    pers_mae = float(np.abs(pers - y).mean())
    assert abs(pers_mae - t4["Persistence"]) < 1e-3, f"STOP: persistence not reproduced ({pers_mae} vs {t4['Persistence']})"
    res["reserve"]["persistence (static)"] = {g: {"reliability": float((y[v] >= pers[v] - ps).mean()), "mean_reserve_kw": ps,
                                                  "unserved_kw": float(np.maximum(pers[v] - ps - y[v], 0).mean())} for g, v in masks.items()}
    res["reserve_diff_G3ppF_minus"] = {m: {g: day_bootstrap((reserves[PP] - reserves[m])[v], day[v], sunny[v]) for g, v in masks.items() if g != "sunny"}
                                       for m in MODELS[1:]}
    # --- warnings: endpoint ramp at the frozen test threshold
    rd = json.loads(Path("reports/test_ramp_definition_frozen.json").read_text())
    tau = rd["tau_test"]
    kn, kt = c["k_now"], c["k_now"] + c["dk"]
    lab = np.abs(kt - kn) >= tau
    qk_r = test_k(c, q["E6-S"])
    g = np.concatenate([implied_event_probability(qk_r[i:i + 8192], LEVELS, kn[i:i + 8192], tau) for i in range(0, len(kn), 8192)])
    # alpha-optimal probability thresholds from development (out-of-fold g, fold-threshold labels)
    gd = np.array([gacc[x] / 3 for x in sid_all])
    labd = np.array([meta[x][2] for x in sid_all])
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    vdev = np.stack([value_curve(gd >= t, labd) for t in grid])  # [threshold, alpha]
    thr = grid[vdev.argmax(0)]
    v_prob = np.array([value_curve(g >= thr[i], lab)[i] for i in range(len(ALPHAS))])
    curves = {"probability g (dev-optimal threshold)": v_prob}
    for m in MODELS:
        dhat = test_k(c, q[m])[:, MED] - kn
        curves[f"median {m}"] = value_curve(np.abs(dhat) >= tau, lab)
    pos = lambda v: [float(ALPHAS[v > 0].min()), float(ALPHAS[v > 0].max())] if (v > 0).any() else None
    res["warnings"] = {"base_rate": float(lab.mean()), "tau_test": tau,
                       "curves": {k: {"max_value": float(v.max()), "alpha_at_max": float(ALPHAS[v.argmax()]), "alpha_range_positive": pos(v),
                                      "value": v.tolist()} for k, v in curves.items()},
                       "dev_thresholds": thr.tolist()}
    both = np.zeros(len(ALPHAS), bool)
    for k, v in curves.items():
        if k.startswith("median"):
            both |= v > 0
    both |= v_prob > 0
    res["warnings"]["prob_beats_best_median_share_of_alpha"] = float(np.mean(v_prob[both] > np.max([v for k, v in curves.items() if k.startswith("median")], 0)[both]))
    p = save("y3_operational_value.json", res)
    print("saved", p)
    for m in list(MODELS) + ["persistence (static)"]:
        o = res["reserve"][m]["overall"]
        cl = res["reserve"][m]["cloudy"]
        print(f"reserve {m:22s} overall rel {o['reliability']:.3f} mean {o['mean_reserve_kw']:.2f} kW | cloudy rel {cl['reliability']:.3f} mean {cl['mean_reserve_kw']:.2f}")
    print("G3''-F minus:", {m: [round(x, 3) for x in v["overall"]] for m, v in res["reserve_diff_G3ppF_minus"].items()})
    for k, v in res["warnings"]["curves"].items():
        print(f"value {k:40s} max {v['max_value']:.3f} at alpha {v['alpha_at_max']}  positive {v['alpha_range_positive']}")
    print("prob > best median on share of alpha:", res["warnings"]["prob_beats_best_median_share_of_alpha"])


if __name__ == "__main__":
    main()
