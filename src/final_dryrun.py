"""Development-only dry run of the final test pipeline (no official test data is loaded).

(a) NUMERICAL REPRODUCTION (not validation): every stored fold-validation prediction of every expert member
    (E4-B-fit, E6-S, A2, D4 x 3 seeds x 5 folds) and of the frozen G3/G4 gates is recomputed from checkpoints.
(b) OUT-OF-FOLD ASSEMBLY (genuinely held out): member (s, k) predicts only fold k's validation samples; the
    3-member out-of-fold ensemble exercises the full G3''-F routing (incl. fallback) on development data.
(c) CALIBRATION FREEZE: out-of-fold conformity scores and rolling-window W per model (chosen on development
    data only) are saved for the test-time initialisation; a daily-reset sanity run on development data.
(d) CHECKPOINT INVENTORY with SHA-256.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.cal2 import ALPHA, conformity, interval_score, select_param
from src.calibrate import base_interval
from src.e8_features import FEAT, verify_frozen
from src.final_eval import (EXPERTS, FOLDS, MODELS, SEEDS, TOL_KW, Split, assemble, blend, gate_params, member_predict,
                            nearest_init, rolling_conformal_daily, score, sha256)
from src.gate2_report import ramp_labels

OUT = Path("experiments/final")


def main():
    verify_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    tau = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    rec = np.load(FEAT / "e8_record_features_v1.npz")
    split = Split("dev", torch.device("cuda"), np.concatenate([rec["P"], rec["M"]], 1).astype(np.float32))
    pos = {s: i for i, s in enumerate(split.ds.sample_ids)}
    report = {"reproduction": {}, "gates": {}}
    # (a) numerical reproduction of stored fold predictions
    repro_cache, member_stats = {}, {}
    report["reproduction_members"] = member_stats
    for model, (root, window) in EXPERTS.items():
        worst = 0.0
        for s in SEEDS:
            for k in FOLDS:
                fold = folds_doc["folds"][k]
                stored = [dict(np.load(root / f"seed{s}" / f"fold_{k}" / "preds_val.npz"))]
                extra = root / f"seed{s}" / f"fold_{k}" / "preds_val_extra.npz"
                if extra.exists():
                    stored.append(dict(np.load(extra)))
                ids = np.concatenate([d["sample_ids"] for d in stored])
                q_stored = np.concatenate([d["pred_q"] for d in stored])
                idx = np.array([pos[i] for i in ids])
                got = member_predict(split, model, s, fold, idx)
                per = np.abs(got["pred_q"] - q_stored).max(1)
                stats = {"max": float(per.max()), "mean": float(per.mean()), "share_within_tol": float((per <= TOL_KW).mean()),
                         "n_over_tol": int((per > TOL_KW).sum()), "n": int(len(per))}
                worst = max(worst, stats["max"])
                member_stats[f"{model}|{s}|{k}"] = stats
                repro_cache[(model, s, k)] = (ids, got)
                ok = stats["share_within_tol"] >= 0.999 and stats["max"] <= 5e-3 and stats["mean"] <= 1e-6
                assert ok, f"STOP: {model} seed {s} fold {k} reproduction {stats}"
        report["reproduction"][model] = {"members": 15, "max_abs_diff_kw": worst, "tolerance_kw": TOL_KW,
                                         "criterion": ">=99.9% within 1e-4 kW, max <= 5e-3 kW, mean <= 1e-6 kW (amendment 2)",
                                         "n_over_tol_total": int(sum(v["n_over_tol"] for kk, v in member_stats.items() if kk.startswith(model + "|"))),
                                         "pass": True}
        print(f"(a) {model}: 15 members reproduced, max |diff| {worst:.2e} kW, samples over 1e-4: "
              f"{report['reproduction'][model]['n_over_tol_total']}", flush=True)
    for gname, key in (("G3", "q_G3"), ("G4", "q_G4")):
        worst = 0.0
        for s in SEEDS:
            for k in FOLDS:
                stored = np.load(Path("experiments/E7b_gate/v1") / f"seed{s}" / f"fold_{k}" / "preds_val.npz")
                ids_d, d = repro_cache[("E4-B-fit", s, k)]
                ids_r, r = repro_cache[("E6-S", s, k)]
                assert np.array_equal(ids_d, stored["sample_ids"]) and np.array_equal(ids_r, stored["sample_ids"])
                q = blend(d, r, gate_params(gname, s, k), tau[str(k)]["training_labels_fit_days"])
                per = np.abs(q - stored[key]).max(1)
                worst = max(worst, float(per.max()))
                assert (per <= TOL_KW).mean() >= 0.999 and per.max() <= 5e-3 and per.mean() <= 1e-6, f"STOP: {gname} {s} {k}"
        report["gates"][gname] = {"max_abs_diff_kw": worst, "pass": True}
        print(f"(a) {gname}: reproduced, max |diff| {worst:.2e} kW", flush=True)
    del repro_cache
    torch.cuda.empty_cache()
    # (b) out-of-fold assembly with the full G3''-F routing on development data
    va = {k: np.flatnonzero(np.isin(split.ds.day.astype(str), folds_doc["folds"][k]["val_days"])) for k in FOLDS}
    fc = assemble(split, {(s, k): va[k] for s in SEEDS for k in FOLDS}, folds_doc, tau)
    allva = np.concatenate([va[k] for k in FOLDS])
    assert all((fc[m]["n_members"][allva] == 3).all() for m in ("G3''-F", "E4-B-fit", "E6-S", "G4"))
    rl = ramp_labels("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz", folds_doc)
    sid = split.ds.sample_ids[allva]
    y = split.pv[split.ds.target_rows[allva]]
    ramp = np.array([rl["is_ramp"][i] for i in sid])
    clear = np.array([folds_doc["day_labels"][d] for d in split.ds.day.astype(str)[allva]])
    h60 = split.h60_ok[allva]
    masks = {"overall": np.ones(len(allva), bool), "sunny (dev clear-day label)": clear, "cloudy": ~clear, "ramp": ramp,
             "non-ramp": ~ramp, "sunrise ramp (no 60-min history)": ramp & ~h60, "60-min eligible": h60, "fallback (no 60-min history)": ~h60}
    levels = np.array([.05, .1, .2, .3, .5, .7, .8, .9, .95])
    report["oof_development"] = {m: score(fc[m]["pred_q"][allva], y, levels, masks) for m in ("G3''-F", "E4-B-fit", "E6-S", "G4")}
    report["oof_development"]["D4 (60-min eligible)"] = score(fc["D4"]["pred_q"][allva][h60], y[h60], levels,
                                                              {g: v[h60] for g, v in masks.items()})
    report["routing"] = {"share_G3pp": float(h60.mean()), "share_G3_fallback": float((~h60).mean())}
    print("(b) out-of-fold G3''-F overall CRPS", round(report["oof_development"]["G3''-F"]["overall"]["crps_kw"], 4), flush=True)
    # (c) calibration freeze from out-of-fold forecasts (development only)
    org = split.ds.origin_time.astype("datetime64[us]").astype(np.int64)[allva]
    tgt = split.ds.target_time.astype("datetime64[us]").astype(np.int64)[allva]
    dayv = split.ds.day.astype(str)[allva]
    save, W, sanity = {"origin_us": org, "sample_ids": sid}, {}, {}
    for m in ("G3''-F", "E4-B-fit", "E6-S", "G4"):
        q = fc[m]["pred_q"][allva]
        bounds = {n: base_interval(q, levels, n) for n in ALPHA}
        for n in ALPHA:
            save[f"{m}|{n}"] = conformity(*bounds[n], y)
        w, _ = select_param("rolling", org, tgt, bounds, y)
        W[m] = int(w)
        lo, hi = bounds["90"]
        inits = {}
        for d in np.unique(dayv):
            other = dayv != d  # sanity run: initialise from OTHER days only
            inits[d] = nearest_init(org[other], save[f"{m}|90"][other], org[dayv == d].min(), W[m])
        clo, chi = rolling_conformal_daily(org, tgt, dayv, lo, hi, y, 0.1, W[m], inits)
        sanity[m] = {g: {"coverage90": float(np.mean((y[v] >= clo[v]) & (y[v] <= chi[v]))),
                         "interval_score90": float(interval_score(clo[v], chi[v], y[v], 0.1).mean())}
                     for g, v in masks.items() if v.any()}
    np.savez_compressed(OUT / "dev_oof_calibration_v1.npz", **save)
    (OUT / "dev_oof_calibration_v1.json").write_text(json.dumps({"W": W, "sha256": sha256(OUT / "dev_oof_calibration_v1.npz"),
                                                                 "levels": list(ALPHA)}, indent=1))
    report["calibration"] = {"W": W, "daily_reset_sanity_dev_oof": sanity}
    print("(c) calibration frozen, W =", W, flush=True)
    # (d) checkpoint inventory
    files = []
    for model, (root, _) in EXPERTS.items():
        for s in SEEDS:
            files += [root / f"seed{s}" / "config.json"]
            for k in FOLDS:
                fd = root / f"seed{s}" / f"fold_{k}"
                files += [fd / "best.pt"] + ([fd / "e8_standardisation.json"] if (fd / "e8_standardisation.json").exists() else [])
    for s in SEEDS:
        for k in FOLDS:
            files += [Path("experiments/E7b_gate/v1") / f"seed{s}" / f"fold_{k}" / "params.json",
                      Path("experiments/E9/v1/G3pp_params") / f"seed{s}" / f"fold_{k}" / "params.json"]
    files += [Path("experiments/E1_sunset_port/tol30_v2/preds_test.npz"), Path("reports/test_ramp_definition_frozen.json"),
              Path("reports/phase6_frozen_thresholds.json"), Path("reports/e8_geometry_frozen.json"),
              Path("data/manifests/e8_eligibility_v1.npz"), Path("data/manifests/forecast_tol30_v2.npz"),
              Path("data/manifests/folds_blocked5_v1.json"), OUT / "dev_oof_calibration_v1.npz"]
    inv = {"files": {str(p).replace("\\", "/"): sha256(p) for p in files}}
    Path("reports/checkpoint_inventory.json").write_text(json.dumps(inv, indent=1))
    report["inventory_files"] = len(inv["files"])
    report["runtime_s"] = time.time() - t0
    Path("reports/final_dryrun_results.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"(d) inventory: {len(inv['files'])} files hashed; dry run done in {report['runtime_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
