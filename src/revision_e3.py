"""Revision E3: forward-in-time SKIPP'D evaluation (protocol revision/protocols/R-E3_forward.md).

Fit <= 2019-06-30 | selection/calibration 2019-07-02..07-31 | evaluation 2019-08-02..10-26.
Every fitted quantity (expert weights, clear-sky anchor, event threshold, gates, conformal initial scores) uses
fit or selection data only. The evaluation period is scored once after all fitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.baselines import official_persistence
from src.cal2 import ALPHA, conformity, interval_score, replay
from src.calibrate import base_interval
from src.clear_sky import ClearSkyPV, clear_sky_index
from src.final_eval import Split, day_bootstrap, rolling_conformal_daily
from src.gate7b import apply_gate, raw_features, select_and_fit
from src.metrics import crps_quantile_interp
from src.revision_common import LEVELS, save
from src.train import build_model, predict_raw

ROOT = Path("experiments/R_E3")
SEEDS = (42, 123, 456, 789, 1011)
FOLDS = json.loads(Path("revision/manifests/folds_forward_v1.json").read_text())
W = {"G3''-F": 4620, "E4-B-fit": 990, "E6-S": 4620, "G4": 4620, "P15": 990}
GAMMA, W_ACI = 0.005, 4620
MODELS = ("G3''-F", "E4-B-fit", "E6-S", "G4", "P15")


def fit_anchor(split: Split):
    """Clear-sky anchor fitted on fit-period development records only (as src/train.run_fold without hold-out)."""
    from src.final_eval import dev_arrays
    t, pv = dev_arrays()
    sel = np.isin(t.astype("datetime64[D]").astype(str), FOLDS["folds"][0]["train_days"])
    return ClearSkyPV().fit(t[sel], pv[sel])


def predict(split: Split, name: str, seed: int, idx: np.ndarray, anchor) -> dict:
    run = ROOT / name / "csi" / f"seed{seed}"
    cfg = json.loads((run / "config.json").read_text())
    fdir = run / "fold_0"
    pc = anchor(split.rec_t)
    k_rec = clear_sky_index(split.pv, pc, 1.0)
    data = split.data
    data.set_history_feature(k_rec)
    data.residual = False
    data.k_now = torch.tensor(k_rec[split.ds.hist_rows[:, -1]], dtype=torch.float32, device=split.device)
    if cfg.get("e8_window") == "H60":
        assert split.h60_ok[idx].all()
        data.set_window(split.h60_rows)
        data.set_extra(np.zeros((len(split.pv), 11), np.float32))
    else:
        data.set_window(split.ds.hist_rows)
        data.set_extra(None)
    net = build_model(cfg).to(split.device)
    net.encoder.cnn_bf16 = cfg["cnn_bf16"]
    net.load_state_dict(torch.load(fdir / "best.pt", map_location=split.device))
    net.eval()
    raw = predict_raw(net, data, torch.as_tensor(idx, device=split.device)).cpu().numpy().astype(float)
    del net
    torch.cuda.empty_cache()
    return {"pred_q": raw * np.maximum(pc[split.ds.target_rows[idx]], 1.0)[:, None], "pclear_target_kw": pc[split.ds.target_rows[idx]],
            "pclear_now_kw": pc[split.ds.hist_rows[idx, -1]], "pv_now_kw": split.pv[split.ds.hist_rows[idx, -1]], "quantile_levels": LEVELS}


def sel_preds(name, seed):
    return dict(np.load(ROOT / name / "csi" / f"seed{seed}" / "fold_0" / "preds_val.npz"))


def sub(d, m):
    return {k: (v[m] if np.ndim(v) and len(v) == len(m) else v) for k, v in d.items()}


def fit_gates(seed):
    """Gates on selection-period predictions only (frozen E7b/E9 procedure)."""
    r, d, a = sel_preds("E6-S", seed), sel_preds("E4-B-fit", seed), sel_preds("A2", seed)
    assert np.array_equal(r["sample_ids"], d["sample_ids"])
    tau = float(r["tau"])
    order = np.argsort(r["origin_time_us"], kind="stable")
    fr = raw_features(d, r, tau)
    gates = {"G3": select_and_fit(fr, ("g", "d"), d["pred_q"], r["pred_q"], r["target"], LEVELS, order),
             "G4": select_and_fit(fr, ("g", "s_D", "s_R", "d"), d["pred_q"], r["pred_q"], r["target"], LEVELS, order)}
    rc = sub(r, np.isin(r["sample_ids"], a["sample_ids"]))
    assert np.array_equal(rc["sample_ids"], a["sample_ids"])
    oc = np.argsort(rc["origin_time_us"], kind="stable")
    gates["G3''"] = select_and_fit(raw_features(a, rc, tau), ("g", "d"), a["pred_q"], rc["pred_q"], rc["target"], LEVELS, oc)
    return gates, tau


def blend(direct, resid, p, tau):
    w = apply_gate(p, raw_features(direct, resid, tau))
    return w[:, None] * direct["pred_q"] + (1 - w[:, None]) * resid["pred_q"]


def ensemble_on(split, idx, anchor, gates_by_seed):
    h60 = split.h60_ok[idx]
    acc = {m: [] for m in MODELS + ("A2",)}
    for s in SEEDS:
        gates, tau = gates_by_seed[s]
        e4, e6, p15 = (predict(split, n, s, idx, anchor) for n in ("E4-B-fit", "E6-S", "P15"))
        a2 = predict(split, "A2", s, idx[h60], anchor)
        q = blend(e4, e6, gates["G3"], tau)
        q[h60] = blend(a2, sub(e6, h60), gates["G3''"], tau)
        acc["G3''-F"].append(q)
        acc["G4"].append(blend(e4, e6, gates["G4"], tau))
        acc["E4-B-fit"].append(e4["pred_q"]); acc["E6-S"].append(e6["pred_q"]); acc["P15"].append(p15["pred_q"])
        a2q = np.full((len(idx), 9), np.nan); a2q[h60] = a2["pred_q"]; acc["A2"].append(a2q)
    out = {m: np.mean(v, 0) for m, v in acc.items()}
    for m in MODELS:
        assert np.isfinite(out[m]).all() and (np.diff(out[m], axis=1) >= -1e-9).all(), f"STOP: invalid ensemble {m}"
    return out, h60


def selection_ensemble(gates_by_seed):
    """5-seed selection-period ensembles (the gates are in-sample here; disclosed) for conformal initial scores."""
    acc = {m: [] for m in MODELS}
    for s in SEEDS:
        gates, tau = gates_by_seed[s]
        r, d, a, p = sel_preds("E6-S", s), sel_preds("E4-B-fit", s), sel_preds("A2", s), sel_preds("P15", s)
        h = np.isin(r["sample_ids"], a["sample_ids"])
        q = blend(d, r, gates["G3"], tau)
        q[h] = blend(a, sub(r, h), gates["G3''"], tau)
        acc["G3''-F"].append(q); acc["G4"].append(blend(d, r, gates["G4"], tau))
        acc["E4-B-fit"].append(d["pred_q"]); acc["E6-S"].append(r["pred_q"]); acc["P15"].append(p["pred_q"])
    r = sel_preds("E6-S", SEEDS[0])
    return {m: np.mean(v, 0) for m, v in acc.items()}, r["target"], r["origin_time_us"]


def main():
    dev = Split("dev", torch.device("cuda"))
    anchor = fit_anchor(dev)
    day = dev.ds.day.astype(str)
    idx = np.flatnonzero(np.isin(day, FOLDS["eval_days"]))
    gates_by_seed = {s: fit_gates(s) for s in SEEDS}
    q, h60 = ensemble_on(dev, idx, anchor, gates_by_seed)
    y = dev.pv[dev.ds.target_rows[idx]]
    d_ = day[idx]
    sunny = np.array([FOLDS["day_labels"][x] for x in d_])
    tau = gates_by_seed[SEEDS[0]][1]
    k = dev.pv / np.maximum(anchor(dev.rec_t), 1.0)
    dk = k[dev.ds.target_rows[idx]] - k[dev.ds.hist_rows[idx, -1]]
    ramp = np.abs(dk) >= tau
    masks = {"overall": np.ones(len(y), bool), "sunny": sunny, "cloudy": ~sunny, "ramp": ramp, "non-ramp": ~ramp,
             "rising ramp": ramp & (dk > 0), "falling ramp": ramp & (dk < 0), "sunrise ramp (fallback)": ramp & ~h60}
    pers = official_persistence(dev.ds.origin_time[idx].astype(object), dev.pv[dev.ds.hist_rows[idx, -1]])
    crps = {m: crps_quantile_interp(q[m], y, LEVELS) for m in MODELS}
    crps["persistence"] = np.abs(pers - y)
    res = {"protocol": "revision/protocols/R-E3_forward.md", "periods": FOLDS["periods"], "n_eval": int(len(idx)),
           "n_eval_days": int(len(np.unique(d_))), "eval_days": sorted(np.unique(d_).tolist()), "n_sunny_days": int(len(set(d_[sunny]))),
           "ramp_tau": tau, "routing_share_G3pp": float(h60.mean()),
           "gates": {str(s): {g: {kk: p[kk] for kk in ("names", "lambda", "b0", "beta")} for g, p in gs[0].items()} for s, gs in gates_by_seed.items()},
           "crps": {m: {g: float(c[v].mean()) for g, v in masks.items() if v.any()} for m, c in crps.items()},
           "rmse_median": {m: {g: float(np.sqrt(np.mean((q[m][v, 4] - y[v]) ** 2))) for g, v in masks.items() if v.any()} for m in MODELS},
           "n": {g: int(v.sum()) for g, v in masks.items()}, "bootstrap": {}, "intervals": {}}
    res["rmse_median"]["persistence"] = {g: float(np.sqrt(np.mean((pers[v] - y[v]) ** 2))) for g, v in masks.items() if v.any()}
    for comp in ("E4-B-fit", "E6-S", "G4", "P15", "persistence"):
        dd = crps["G3''-F"] - crps[comp]
        res["bootstrap"][f"G3''-F - {comp}"] = {g: day_bootstrap(dd[v], d_[v], sunny[v]) for g, v in masks.items()
                                                if g in ("overall", "sunny", "cloudy", "ramp", "non-ramp")}
    res["verdicts"] = {"F1 G3''-F - E4-B-fit overall CI<0": res["bootstrap"]["G3''-F - E4-B-fit"]["overall"][2] < 0,
                       "F2 G3''-F - persistence overall CI<0": res["bootstrap"]["G3''-F - persistence"]["overall"][2] < 0}
    # calibration methods on the same raw forecasts; initial scores = latest selection-period scores (all matured before)
    qs, ys, os_ = selection_ensemble(gates_by_seed)
    org = dev.ds.origin_time.astype("datetime64[us]").astype(np.int64)[idx]
    tgt = dev.ds.target_time.astype("datetime64[us]").astype(np.int64)[idx]
    order_s = np.argsort(os_, kind="stable")
    assert os_.max() + 16 * 60e6 < org.min(), "STOP: selection scores not matured before evaluation"
    for m in MODELS:
        res["intervals"][m] = {}
        for nm, a in ALPHA.items():
            lo_s, hi_s = base_interval(qs[m], LEVELS, nm)
            sc = conformity(lo_s, hi_s, ys)[order_s]
            lo, hi = base_interval(q[m], LEVELS, nm)
            ww = W[m]
            meth = {"none": (lo, hi),
                    "daily-reset rolling": rolling_conformal_daily(org, tgt, d_, lo, hi, y, a, ww, {x: sc[-ww:] for x in np.unique(d_)}),
                    "continuous rolling": tuple(replay(org, tgt, lo, hi, y, a, "rolling", ww, sc[-ww:])[k_] for k_ in ("lo", "hi")),
                    "ACI": tuple(replay(org, tgt, lo, hi, y, a, "aci", GAMMA, sc[-W_ACI:])[k_] for k_ in ("lo", "hi"))}
            res["intervals"][m][nm] = {mm: {g: {"coverage": float(np.mean((y[v] >= l_[v]) & (y[v] <= h_[v]))),
                                                "width_kw": float(np.mean(h_[v] - l_[v])),
                                                "winkler": float(interval_score(l_[v], h_[v], y[v], a).mean())}
                                            for g, v in masks.items() if v.any()} for mm, (l_, h_) in meth.items()}
    res["verdicts"]["F3 90% coverage within 0.90 +- 0.03 (G3''-F)"] = {
        mm: 0.87 <= res["intervals"]["G3''-F"]["90"][mm]["overall"]["coverage"] <= 0.93 for mm in res["intervals"]["G3''-F"]["90"]}
    res["init_provenance"] = {"selection_first_origin": str(np.datetime64(int(os_.min()), "us")),
                              "selection_last_origin": str(np.datetime64(int(os_.max()), "us")), "n_selection_scores": int(len(ys))}
    p = save("e3_forward.json", {"label": "FORWARD (new evaluation; not pre-registered at the original freeze)", **res})
    print("saved", p, "n", len(idx), "days", res["n_eval_days"])
    print({m: (round(v["overall"], 3), round(v["ramp"], 3), round(v["sunny"], 3)) for m, v in res["crps"].items()})
    print({k_: tuple(round(x, 3) for x in v["overall"]) for k_, v in res["bootstrap"].items()})
    print(res["verdicts"])
    print({mm: (round(v["overall"]["coverage"], 3), round(v["ramp"]["coverage"], 3), round(v["overall"]["winkler"], 2))
           for mm, v in res["intervals"]["G3''-F"]["90"].items()})


if __name__ == "__main__":
    main()
