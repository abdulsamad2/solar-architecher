"""E9 analysis (reports/e9_prereg.md): stage S screen selection, stage C confirmation vs A0 (= E8 D2), stage G gating.

python -m src.e9_report screen    -> reports/e9_screen.json   (seed 42; selects <= 2 arms)
python -m src.e9_report confirm   -> reports/e9_confirm.json  (seeds 42/123/456 for selected arms)
python -m src.e9_report gate      -> reports/e9_gate.json     (G3'' vs G3', CPU)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.cal2 import ALPHA, interval_score
from src.calibrate import base_interval
from src.compare import bootstrap_diff
from src.e8_features import FEAT, verify_frozen
from src.events import implied_event_probability
from src.gate2_report import ramp_labels, strata
from src.gate3_report import conformal
from src.gate7b import apply_gate, average_precision, raw_features, select_and_fit
from src.metrics import crps_quantile_interp

FOLDS = range(5)
ARMS = ("L1", "A1", "A2", "A3", "A4", "A5")
A0 = Path("experiments/E8/v1/D2/csi")
E9 = Path("experiments/E9/v1")
RESID = Path("experiments/E6/v1/E6-S/csi")
D_O, D_S = 0.02, 0.015
GROUPS = ("overall", "sunny", "cloudy", "ramp", "non-ramp")


def arm_dir(arm):
    return A0 if arm == "A0" else E9 / arm / "csi"


def load(arm, seed, split="val"):
    return [dict(np.load(arm_dir(arm) / f"seed{seed}" / f"fold_{k}" / f"preds_{split}.npz")) for k in FOLDS]


def cat(parts, key):
    return np.concatenate([p[key] for p in parts])


def evaluate(arms, seed, rl, with_intervals=False):
    ref = load("A0", seed)
    ids = cat(ref, "sample_ids")
    meta = {k: cat(ref, k) for k in ("sample_ids", "target", "clear_day", "fold", "day", "origin_time_us")}
    st = strata(meta, rl["is_ramp"])
    y, lv = meta["target"], ref[0]["quantile_levels"]
    dayk = np.char.add(meta["fold"].astype(str), np.char.add("_", meta["day"].astype(str)))
    out, crps = {"raw_crps": {}, "contrasts": {}, "event": {}, "intervals": {}}, {}
    for a in ("A0",) + tuple(arms):
        parts = load(a, seed)
        assert np.array_equal(cat(parts, "sample_ids"), ids), f"STOP: {a} samples differ from A0"
        q = cat(parts, "pred_q")
        assert np.isfinite(q).all() and (np.diff(q, axis=1) >= -1e-9).all(), f"STOP: invalid quantiles {a}"
        crps[a] = crps_quantile_interp(q, y, lv)
        out["raw_crps"][a] = {g: float(crps[a][st[g]].mean()) for g in GROUPS}
        prs = []
        for k, p in zip(FOLDS, parts):
            qk = p["pred_q"] / np.maximum(p["pclear_target_kw"], 1.0)[:, None]
            kn = p["pv_now_kw"] / np.maximum(p["pclear_now_kw"], 1.0)
            prs.append(implied_event_probability(qk, lv, kn, rl["tau_ramp"][k]))
        pr = np.concatenate(prs)
        out["event"][a] = {"brier": float(np.mean((pr - st["ramp"]) ** 2)), "auprc": average_precision(pr, st["ramp"])}
        if with_intervals:
            ci = conformal(load(a, seed, "cal"), parts)
            out["intervals"][a] = {g: {n: {"coverage": float(np.mean((y[st[g]] >= ci[n][0][st[g]]) & (y[st[g]] <= ci[n][1][st[g]]))),
                                           "width_kw": float(np.mean(ci[n][1][st[g]] - ci[n][0][st[g]])),
                                           "interval_score": float(interval_score(ci[n][0][st[g]], ci[n][1][st[g]], y[st[g]], ALPHA[n]).mean())}
                                       for n in ALPHA} for g in ("overall", "ramp", "non-ramp")}
    for a in arms:
        out["contrasts"][f"{a} - A0"] = {g: bootstrap_diff(crps[a][st[g]], crps["A0"][st[g]], dayk[st[g]], meta["clear_day"][st[g]], kind="mean")
                                         for g in GROUPS}
    return out


def params_and_cost(arm):
    cfg = json.loads((arm_dir(arm) / "seed42" / "config.json").read_text())
    import re

    import torch

    from src.train import build_model
    n = sum(p.numel() for p in build_model(cfg).parameters())
    log = (arm_dir(arm) / "seed42" / "train.log").read_text(encoding="utf-8")
    ep = [float(x) for x in re.findall(r"epoch\s+\d+ .*\((\d+)s\)", log)]
    return {"parameters": int(n), "epochs_seed42": len(ep), "train_s_seed42": float(sum(ep)), "encoder": cfg.get("encoder", "cnn_gru"),
            "loss_weight_alpha": cfg.get("loss_weight_alpha", 0.0)}


def screen(rl):
    arms = [a for a in ARMS if (E9 / a / "csi" / "seed42").exists()]
    assert set(arms) == set(ARMS), f"screen incomplete: {arms}"
    r = evaluate(arms, 42, rl)
    a0 = r["raw_crps"]["A0"]
    elig = {a: (r["raw_crps"][a]["overall"] - a0["overall"] <= D_O) and (r["raw_crps"][a]["sunny"] - a0["sunny"] <= D_S)
            and (r["raw_crps"][a]["ramp"] < a0["ramp"]) for a in arms}
    cost = {a: params_and_cost(a) for a in arms}
    cand = sorted([a for a in arms if elig[a]], key=lambda a: (r["raw_crps"][a]["ramp"], cost[a]["parameters"]))
    r.update({"eligible": elig, "selected": cand[:2], "cost": cost, "stop": len(cand) == 0})
    Path("reports/e9_screen.json").write_text(json.dumps(r, indent=1, default=float))
    print(json.dumps({"eligible": elig, "selected": r["selected"]}, indent=1))


def confirm(rl):
    sel = json.loads(Path("reports/e9_screen.json").read_text())["selected"]
    res = {"selected": sel, "per_seed": {}}
    for s in (42, 123, 456):
        res["per_seed"][s] = evaluate(sel, s, rl, with_intervals=True)
        res["per_seed"][s]["rule"] = {a: {"ramp_ci_below_0": bool(res["per_seed"][s]["contrasts"][f"{a} - A0"]["ramp"][2] < 0),
                                          "overall_noninferior": bool(res["per_seed"][s]["raw_crps"][a]["overall"] - res["per_seed"][s]["raw_crps"]["A0"]["overall"] <= D_O),
                                          "sunny_noninferior": bool(res["per_seed"][s]["raw_crps"][a]["sunny"] - res["per_seed"][s]["raw_crps"]["A0"]["sunny"] <= D_S)}
                                      for a in sel}
    res["decision"] = {a: {"seeds_passing": int(sum(all(res["per_seed"][s]["rule"][a].values()) for s in (42, 123, 456))),
                           "clean_replication_seeds_passing": int(sum(all(res["per_seed"][s]["rule"][a].values()) for s in (123, 456)))}
                       for a in sel}
    for a in sel:
        res["decision"][a]["improves_ramp_expert"] = res["decision"][a]["seeds_passing"] >= 2
    passing = [a for a in sel if res["decision"][a]["improves_ramp_expert"]]
    res["best_confirmed"] = min(passing, key=lambda a: np.mean([res["per_seed"][s]["raw_crps"][a]["ramp"] for s in (42, 123, 456)])) if passing else None
    Path("reports/e9_confirm.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"decision": res["decision"], "best_confirmed": res["best_confirmed"]}, indent=1))


def gate(rl):
    best = json.loads(Path("reports/e9_confirm.json").read_text())["best_confirmed"]
    assert best, "no confirmed arm: stage G not run"
    frozen = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    win = np.load(FEAT / "e8_windows_v1.npz")
    common_ids = set(win["sample_ids"][win["eligible_common"]])
    res = {"best": best, "per_seed": {}}
    for s in (42, 123, 456):
        vals, meta = {"G3'": [], "G3''": []}, []
        cals = {"G3'": [], "G3''": []}
        for k in FOLDS:
            tau = frozen[str(k)]["training_labels_fit_days"]
            sp = {}
            for split in ("cal", "val"):
                r = dict(np.load(RESID / f"seed{s}" / f"fold_{k}" / f"preds_{split}.npz"))
                keep = np.array([i in common_ids for i in r["sample_ids"]])
                r = {kk: (v[keep] if np.ndim(v) and len(v) == len(keep) else v) for kk, v in r.items()}
                ex = {a: dict(np.load(arm_dir(a) / f"seed{s}" / f"fold_{k}" / f"preds_{split}.npz")) for a in ("A0", best)}
                for a, d in ex.items():
                    assert np.array_equal(d["sample_ids"], r["sample_ids"]), f"STOP: {a} {split} samples differ"
                sp[split] = (r, ex)
            (rc, exc), (rv, exv) = sp["cal"], sp["val"]
            lv, order = rc["quantile_levels"], np.argsort(rc["origin_time_us"], kind="stable")
            for name, a in (("G3'", "A0"), ("G3''", best)):
                fc, fv = raw_features(exc[a], rc, tau), raw_features(exv[a], rv, tau)
                p = select_and_fit(fc, ("g", "d"), exc[a]["pred_q"], rc["pred_q"], rc["target"], lv, order)
                wv, wc = apply_gate(p, fv), apply_gate(p, fc)
                qv = wv[:, None] * exv[a]["pred_q"] + (1 - wv[:, None]) * rv["pred_q"]
                qc = wc[:, None] * exc[a]["pred_q"] + (1 - wc[:, None]) * rc["pred_q"]
                vals[name].append({**{x: rv[x] for x in ("origin_time_us", "target_time_us", "target", "quantile_levels")}, "pred_q": qv})
                cals[name].append({**{x: rc[x] for x in ("origin_time_us", "target_time_us", "target", "quantile_levels")}, "pred_q": qc})
            meta.append(rv)
        m = {key: cat(meta, key) for key in ("sample_ids", "target", "clear_day", "fold", "day", "origin_time_us")}
        st = strata(m, rl["is_ramp"])
        y, lv = m["target"], meta[0]["quantile_levels"]
        dayk = np.char.add(m["fold"].astype(str), np.char.add("_", m["day"].astype(str)))
        crps = {n: crps_quantile_interp(np.concatenate([v["pred_q"] for v in vals[n]]), y, lv) for n in vals}
        raw = {n: {g: float(crps[n][st[g]].mean()) for g in GROUPS} for n in vals}
        d = {g: bootstrap_diff(crps["G3''"][st[g]], crps["G3'"][st[g]], dayk[st[g]], m["clear_day"][st[g]], kind="mean") for g in GROUPS}
        iv = {}
        for n in vals:
            ci = conformal(cals[n], vals[n])
            iv[n] = {g: {"coverage": float(np.mean((y[st[g]] >= ci["90"][0][st[g]]) & (y[st[g]] <= ci["90"][1][st[g]]))),
                         "interval_score": float(interval_score(ci["90"][0][st[g]], ci["90"][1][st[g]], y[st[g]], 0.1).mean())}
                     for g in ("overall", "ramp", "non-ramp")}
        res["per_seed"][s] = {"raw_crps": raw, "G3'' - G3'": d, "rolling_conformal_90": iv,
                              "rule": {"ramp_ci_below_0": bool(d["ramp"][2] < 0),
                                       "overall_within_0.005": bool(raw["G3''"]["overall"] - raw["G3'"]["overall"] <= 0.005)}}
    n = sum(all(res["per_seed"][s]["rule"].values()) for s in (42, 123, 456))
    res["decision"] = {"G3pp_retained": bool(n >= 2), "seeds_passing": int(n)}
    Path("reports/e9_gate.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["decision"], indent=1))


if __name__ == "__main__":
    verify_frozen()
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    rl = ramp_labels("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz", folds_doc)
    {"screen": screen, "confirm": confirm, "gate": gate}[sys.argv[1]](rl)
