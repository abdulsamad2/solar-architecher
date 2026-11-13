"""Revision post hoc test analyses from the per-member dump (experiments/R_posthoc).

  c19: 3- vs 15-member ensemble calibration sensitivity (protocol R-C07 step 5)
  e2:  gate / constant-weight ablation on the 60-min-eligible test samples (protocol R-E2, post hoc part)
  c18: event-probability scoring and median ramp metrics on test (protocol R-C18, post hoc part)
  e1e4: PV-only and history x encoder arms on test (protocols R-E1, R-E4, post hoc part)
All labelled POST HOC; the frozen Gate 5 evaluation is unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.cal2 import ALPHA
from src.calibrate import base_interval
from src.clear_sky import ClearSkyPV
from src.events import implied_event_probability
from src.final_eval import blend, gate_params, rolling_conformal_daily
from src.gate7b import apply_gate, raw_features, select_and_fit
from src.metrics import crps_quantile_interp
from src.revision_c07 import interval_table, past_only_init
from src.revision_c18 import prob_scores, ramp_metrics
from src.revision_common import LEVELS, day_bootstrap, frozen_test_predictions, save, test_context

DUMP = Path("experiments/R_posthoc")
TAU = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())


def members(name):
    p = DUMP / f"test_members_{name}.npz"
    return dict(np.load(p)) if p.exists() else None


def mdict(d, i):
    return {"pred_q": d["pred_q"][i], "pclear_target_kw": d["pclear_target_kw"][i], "pclear_now_kw": d["pclear_now_kw"][i],
            "pv_now_kw": d["pv_now_kw"], "quantile_levels": LEVELS}


def sub(d, m):
    return {k: (v[m] if np.ndim(v) and len(v) == len(m) else v) for k, v in d.items()}


def g3ppf_members(h60):
    d, r, a = members("E4-B-fit"), members("E6-S"), members("A2")
    out = []
    for i, (s, k) in enumerate(d["members"]):
        tau = TAU[str(k)]["training_labels_fit_days"]
        md, mr, ma = mdict(d, i), mdict(r, i), mdict(a, i)
        q = blend(md, mr, gate_params("G3", s, k), tau)
        q[h60] = blend(sub(ma, h60), sub(mr, h60), gate_params("G3''", s, k), tau)
        out.append(q)
    return np.stack(out), d["members"]


def c19(c):
    q15 = frozen_test_predictions(c["sample_ids"])["G3''-F"]
    mq, mem = g3ppf_members(c["h60"])
    assert np.abs(mq.mean(0) - q15).max() < 1e-9
    cal = np.load("experiments/final/dev_oof_calibration_v1.npz")
    W = json.loads(Path("experiments/final/dev_oof_calibration_v1.json").read_text())["W"]["G3''-F"]
    y, day, org, tgt, masks = c["y"], c["day"], c["org"], c["tgt"], c["masks"]
    start = {d: org[day == d].min() for d in np.unique(day)}
    res = {}
    ens = {"15 members": q15} | {f"3 members (fold {k})": mq[mem[:, 1] == k].mean(0) for k in range(5)}
    for name, q in ens.items():
        res[name] = {"crps_overall": float(crps_quantile_interp(q, y, LEVELS).mean())}
        for nm, a in ALPHA.items():
            lo, hi = base_interval(q, LEVELS, nm)
            init = {d: past_only_init(cal["origin_us"].astype(np.int64), cal[f"G3''-F|{nm}"], start[d], W)[0] for d in start}
            clo, chi = rolling_conformal_daily(org, tgt, day, lo, hi, y, a, W, init)
            sel = {g: masks[g] for g in ("overall", "ramp", "sunny", "cloudy")}
            res[name][nm] = {"raw": interval_table(lo, hi, y, a, sel), "past_only_calibrated": interval_table(clo, chi, y, a, sel)}
    return res


def e2(c):
    """Refit each blend per member on its fold's calibration days (identical to the development fits), apply on test."""
    h60 = c["h60"]
    y = c["y"][h60]
    r, a, d = members("E6-S"), members("A2"), members("E4-B-fit")
    exp = lambda root, s, k: dict(np.load(Path(root) / f"seed{s}" / f"fold_{k}" / "preds_cal.npz"))
    from src.e8_features import FEAT
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = win["sample_ids"][win["eligible_common"]]
    from scipy.optimize import minimize_scalar
    import torch
    from src.gate7b import BlendCRPS, grid_quantiles
    acc = {m: [] for m in ("A2", "E6-S", "B50", "BC", "Gg", "Gd", "G3''", "G4")}
    for i, (s, k) in enumerate(r["members"]):
        tau = TAU[str(k)]["training_labels_fit_days"]
        rc, dc = exp("experiments/E6/v1/E6-S/csi", s, k), exp("experiments/E9/v1/A2/csi", s, k)
        keep = np.isin(rc["sample_ids"], common)
        rc, dc = sub(rc, keep), sub(dc, np.isin(dc["sample_ids"], common))
        assert np.array_equal(rc["sample_ids"], dc["sample_ids"])
        fc = raw_features(dc, rc, tau)
        order = np.argsort(rc["origin_time_us"], kind="stable")
        mr, ma = sub(mdict(r, i), h60), sub(mdict(a, i), h60)
        fv = raw_features(ma, mr, tau)
        fb = BlendCRPS(grid_quantiles(dc["pred_q"], LEVELS), grid_quantiles(rc["pred_q"], LEVELS), rc["target"])
        n = len(rc["target"])
        wc = minimize_scalar(lambda x: float(fb.per_sample(torch.full((n,), float(x), dtype=torch.float64)).mean()),
                             bounds=(0, 1), method="bounded", options={"xatol": 1e-6}).x
        w = {"B50": np.full(h60.sum(), 0.5), "BC": np.full(h60.sum(), wc)}
        for nm, names in (("Gg", ("g",)), ("Gd", ("d",))):
            w[nm] = apply_gate(select_and_fit(fc, names, dc["pred_q"], rc["pred_q"], rc["target"], LEVELS, order), fv)
        w["G3''"] = apply_gate(gate_params("G3''", s, k), fv)  # frozen G3'' parameters
        for nm, ww in w.items():
            acc[nm].append(ww[:, None] * ma["pred_q"] + (1 - ww[:, None]) * mr["pred_q"])
        acc["A2"].append(ma["pred_q"]); acc["E6-S"].append(mr["pred_q"])
        acc["G4"].append(blend(sub(mdict(d, i), h60), mr, gate_params("G4", s, k), tau))
    masks = {g: v[h60] for g, v in c["masks"].items() if g != "sunrise ramp"}
    day, sunny = c["day"][h60], c["sunny"][h60]
    crps = {m: crps_quantile_interp(np.mean(v, 0), y, LEVELS) for m, v in acc.items()}
    res = {"n": int(h60.sum()), "crps": {m: {g: float(cc[v].mean()) for g, v in masks.items()} for m, cc in crps.items()},
           "G3pp_minus": {m: {g: day_bootstrap((crps["G3''"] - crps[m])[v], day[v], sunny[v]) for g, v in masks.items()}
                          for m in crps if m != "G3''"}}
    return res


def test_k(c, q_kw):
    rd = json.loads(Path("reports/test_ramp_definition_frozen.json").read_text())
    pc = np.maximum(ClearSkyPV(rd["anchor_coef"])(c["rec_t"]), 1.0)[c["ds"].target_rows]
    return q_kw / pc[:, None]


def c18(c):
    r = members("E6-S")
    rd = json.loads(Path("reports/test_ramp_definition_frozen.json").read_text())
    tau = rd["tau_test"]
    from src.events import event_labels, future_rows
    m = np.load("data/manifests/forecast_tol30_v2.npz")
    rec_us = c["rec_t"].astype(np.int64)
    fr = future_rows(rec_us, c["org"])
    k_rec = c["pv"] / np.maximum(ClearSkyPV(rd["anchor_coef"])(c["rec_t"]), 1.0)
    k_fut = np.where(fr >= 0, k_rec[np.maximum(fr, 0)], np.nan)
    kn, kt = c["k_now"], c["k_now"] + c["dk"]
    ev = event_labels(kn, k_fut, kt, tau)
    lab_c, valid = ev["time_class"] != 3, ev["event_valid"]
    lab_e = np.abs(kt - kn) >= tau
    p_cross = 1 - r["time_probs"][:, :, 3].mean(0)
    p_mix = r["p_abs_change_ge_tau"].mean(0)
    q = frozen_test_predictions(c["sample_ids"])
    qk_r = test_k(c, q["E6-S"])
    g = np.concatenate([implied_event_probability(qk_r[i:i + 8192], LEVELS, kn[i:i + 8192], tau) for i in range(0, len(kn), 8192)])
    # out-of-sample climatology: development-period base rates (frozen fold thresholds) are not on the test tau scale,
    # so the test-period base rate of the OTHER test days is used (leave-one-day-out)
    day = c["day"]
    clim_loo = lambda lab, v: float(np.mean([lab[v & (day != d)].mean() for d in np.unique(day[v])]))
    fold_taus = [TAU[str(k)]["training_labels_fit_days"] for k in range(5)]
    res = {"tau_test": tau, "fold_taus": fold_taus,
           "crossing: P_cross": prob_scores(p_cross[valid], lab_c[valid], clim_loo(lab_c, valid)),
           "endpoint: g (ensemble quantiles)": prob_scores(g, lab_e, clim_loo(lab_e, np.ones(len(g), bool))),
           "endpoint: p_mix (member mean)": prob_scores(p_mix, lab_e, clim_loo(lab_e, np.ones(len(g), bool))),
           "ramp_metrics": {}}
    for mdl in ("G3''-F", "E4-B-fit", "E6-S", "G4"):
        res["ramp_metrics"][mdl] = ramp_metrics(test_k(c, q[mdl]), kn, kt, tau)
    res["ramp_metrics"]["persistence (no change)"] = ramp_metrics(np.repeat(kn[:, None], 9, 1), kn, kt, tau)
    return res


def e1e4(c):
    h60 = c["h60"]
    y, day, sunny = c["y"], c["day"], c["sunny"]
    q = frozen_test_predictions(c["sample_ids"])
    arms = {}
    for n in ("E4-B-fit", "A2", "D0", "D2", "A2-H15", "P15", "P60"):
        d = members(n)
        if d is not None:
            arms[n] = np.nanmean(d["pred_q"], 0) if n not in ("A2", "D2", "P60") else np.mean(d["pred_q"], 0)
    arms["G3''-F"] = q["G3''-F"]
    res = {"available": sorted(arms), "full": {}, "h60": {}, "diffs": {}}
    for pop, sel in (("full", np.ones(len(y), bool)), ("h60", h60)):
        masks = {g: v[sel] for g, v in c["masks"].items() if g != "sunrise ramp" or pop == "full"}
        for n, qq in arms.items():
            if not np.isfinite(qq[sel]).all():
                continue
            cc = crps_quantile_interp(qq[sel], y[sel], LEVELS)
            med = qq[sel][:, 4]
            res[pop][n] = {g: {"crps": float(cc[v].mean()), "rmse": float(np.sqrt(np.mean((med[v] - y[sel][v]) ** 2))),
                               "mae": float(np.abs(med[v] - y[sel][v]).mean())} for g, v in masks.items() if v.any()}
    pairs = [("P15", "E4-B-fit", "full"), ("P15", "G3''-F", "full"), ("P60", "A2", "h60"), ("P60", "G3''-F", "h60"),
             ("D2", "D0", "h60"), ("A2", "A2-H15", "h60"), ("A2-H15", "D0", "h60"), ("A2", "D2", "h60")]
    for a_, b_, pop in pairs:
        if a_ in arms and b_ in arms:
            sel = h60 if pop == "h60" else np.ones(len(y), bool)
            da = crps_quantile_interp(arms[a_][sel], y[sel], LEVELS) - crps_quantile_interp(arms[b_][sel], y[sel], LEVELS)
            res["diffs"][f"{a_} - {b_} ({pop})"] = {g: day_bootstrap(da[v[sel]], day[sel][v[sel]], sunny[sel][v[sel]])
                                                   for g, v in c["masks"].items() if g in ("overall", "ramp", "non-ramp", "sunny", "cloudy")}
    return res


def main(parts):
    c = test_context()
    for p in parts:
        r = {"c19": c19, "e2": e2, "c18": c18, "e1e4": e1e4}[p](c)
        out = save(f"posthoc_test_{p}.json", {"label": "POST HOC (test); frozen evaluation unchanged", **r})
        print("saved", out)
        if p == "c19":
            for n, v in r.items():
                print(f"{n:22s} CRPS {v['crps_overall']:.3f}  raw90 {v['90']['raw']['overall']['coverage']:.3f}  cal90 {v['90']['past_only_calibrated']['overall']['coverage']:.3f}")
        if p == "e2":
            print({m: (round(v["overall"], 4), round(v["ramp"], 3)) for m, v in r["crps"].items()})
        if p == "c18":
            print({k: (round(v["brier_skill"], 3), round(v["auprc"], 3)) for k, v in r.items() if isinstance(v, dict) and "brier" in v})
            print({k: (round(v["recall"], 3), round(v["precision"], 3), round(v["direction_accuracy"], 3), round(v["change_slope_ramps"], 3)) for k, v in r["ramp_metrics"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", choices=["c19", "e2", "c18", "e1e4"])
    main(ap.parse_args().parts)
