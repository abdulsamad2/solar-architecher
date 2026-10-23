"""Phase F evaluation (reports/phaseF_physics_prereg.md): development out-of-fold G3''-F for each physics arm.

For every arm the A2 expert's stored fold predictions (preds_cal / preds_val / preds_val_extra) are combined with the
frozen E6-S predictions; the G3'' gate is refitted on calibration days with the frozen recipe (src/freeze_g3pp.py);
samples without a 60-min history use the frozen G3 (E4-B-fit + E6-S). Members are averaged per sample (3 seeds,
out-of-fold). No test data is read.

  python scripts/evaluate_physics_ablation.py [--arms A0 A1 R A3 A4 A5]
Outputs: reports/phaseF_results.json, reports/phaseF_table.md, reports/figures/F_phys*.png|pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cal2 import ALPHA, conformity, interval_score  # noqa: E402
from src.calibrate import base_interval  # noqa: E402
from src.final_eval import blend, day_bootstrap, gate_params, nearest_init, rolling_conformal_daily  # noqa: E402
from src.gate7b import apply_gate, raw_features, select_and_fit  # noqa: E402
from src.metrics import crps_quantile_interp  # noqa: E402
from src.physics_metrics import physics_diagnostics  # noqa: E402

LV = np.array([.05, .1, .2, .3, .5, .7, .8, .9, .95])
SEEDS, FOLDS = (42, 123, 456), range(5)
E4 = ROOT / "experiments/E4_calib/v1/csi"
E6 = ROOT / "experiments/E6/v1/E6-S/csi"
ARM_DIRS = {"A0": ROOT / "experiments/E9/v1/A2/csi"}
ARM_DIRS.update({a: ROOT / f"experiments/F_physics/final_{a}/csi" for a in ("A1", "R", "A3", "A4", "A5")})
LABELS = {"A0": "Original G3''-F", "A1": "+ Physics", "R": "+ Ramp", "A3": "+ Physics + Ramp",
          "A4": "+ Physics + Ramp + Bounds", "A5": "+ Physics + Ramp + Bounds + Direction"}
W_FROZEN = json.loads((ROOT / "experiments/final/dev_oof_calibration_v1.json").read_text())["W"]["G3''-F"]
K_MAX = 1.5


def load(p: Path) -> dict:
    return dict(np.load(p))


def sub(d: dict, idx) -> dict:
    n = len(d["sample_ids"])
    return {k: (v[idx] if np.ndim(v) and len(v) == n else v) for k, v in d.items()}


def by_ids(d: dict, ids) -> dict:
    pos = {s: i for i, s in enumerate(d["sample_ids"])}
    return sub(d, np.array([pos[i] for i in ids]))


def arm_oof(arm: str, z) -> np.ndarray:
    """Out-of-fold G3''-F quantiles [n, 9] in the order of the Phase C stream."""
    tau = json.loads((ROOT / "reports/phase6_frozen_thresholds.json").read_text())
    pos = {s: i for i, s in enumerate(z["sample_ids"])}
    acc, cnt = np.zeros((len(pos), 9)), np.zeros(len(pos))
    for k in FOLDS:
        t_k = tau[str(k)]["training_labels_fit_days"]
        for s in SEEDS:
            run = ARM_DIRS[arm] / f"seed{s}" / f"fold_{k}"
            e6v, e6c = load(E6 / f"seed{s}" / f"fold_{k}" / "preds_val.npz"), load(E6 / f"seed{s}" / f"fold_{k}" / "preds_cal.npz")
            e4v = load(E4 / f"seed{s}" / f"fold_{k}" / "preds_val.npz")
            ac = load(run / "preds_cal.npz")
            rc = by_ids(e6c, ac["sample_ids"])
            order = np.argsort(rc["origin_time_us"], kind="stable")
            p = select_and_fit(raw_features(ac, rc, t_k), ("g", "d"), ac["pred_q"], rc["pred_q"], rc["target"], LV, order)
            av = load(run / "preds_val.npz")
            if (run / "preds_val_extra.npz").exists():
                ex = load(run / "preds_val_extra.npz")
                av = {kk: (np.concatenate([av[kk], ex[kk]]) if kk in ex and np.ndim(av[kk]) and len(av[kk]) == len(av["sample_ids"])
                           else av[kk]) for kk in av}
            rv = by_ids(e6v, av["sample_ids"])
            av_full = {**rv, "pred_q": av["pred_q"]}  # A2 rows share E6-S's per-sample context (same fold anchor)
            w = apply_gate(p, raw_features(av_full, rv, t_k))
            q_g3pp = w[:, None] * av["pred_q"] + (1 - w[:, None]) * rv["pred_q"]
            idx_pp = np.array([pos[i] for i in av["sample_ids"]])
            acc[idx_pp] += q_g3pp
            # fallback G3 (frozen) for validation samples without a 60-min history
            rest = np.setdiff1d(e6v["sample_ids"], av["sample_ids"])
            if len(rest):
                d4, r6 = by_ids(e4v, rest), by_ids(e6v, rest)
                acc[[pos[i] for i in rest]] += blend(d4, r6, gate_params("G3", s, k), t_k)
                cnt[[pos[i] for i in rest]] += 1
            cnt[idx_pp] += 1
    assert (cnt == 3).all(), f"STOP: {arm} member count {np.unique(cnt)}"
    return acc / 3


def context(z):
    """Per-sample fold-anchor context from stored E4-B-fit predictions (seed 42; anchor is seed-independent)."""
    pos = {s: i for i, s in enumerate(z["sample_ids"])}
    n = len(pos)
    pct, pcn, pvn, pers = (np.zeros(n) for _ in range(4))
    for k in FOLDS:
        d = load(E4 / "seed42" / f"fold_{k}" / "preds_val.npz")
        ii = [pos[i] for i in d["sample_ids"]]
        pct[ii], pcn[ii], pvn[ii], pers[ii] = d["pclear_target_kw"], d["pclear_now_kw"], d["pv_now_kw"], d["persistence"]
    return pct, pcn, pvn, pers


def evaluate(q, z, ctx) -> dict:
    y, day = z["y"].astype(float), z["day"]
    pct, pcn, pvn, pers = ctx
    k_true, k_now = y / np.maximum(pct, 1), pvn / np.maximum(pcn, 1)
    ramp, sunny, h60 = z["ramp"].astype(bool), z["sunny"].astype(bool), z["h60"].astype(bool)
    up, down = ramp & (k_true > k_now), ramp & (k_true < k_now)
    strata = {"overall": np.ones(len(y), bool), "sunny": sunny, "cloudy": ~sunny, "ramp": ramp,
              "sunrise ramp": ramp & ~h60, "rising ramp": up, "falling ramp": down}
    crps = crps_quantile_interp(q, y, LV)
    med = q[:, 4]
    rm = lambda f, m: float(np.sqrt(np.mean((f[m] - y[m]) ** 2)))
    out = {"crps": {g: float(crps[m].mean()) for g, m in strata.items()},
           "point": {"rmse": rm(med, strata["overall"]), "mae": float(np.abs(med - y).mean()),
                     "rmse_skill_vs_persistence": 1 - rm(med, strata["overall"]) / rm(pers, strata["overall"])},
           "intervals": {}, "_crps_per_sample": crps}
    org, tgt = z["origin_us"], z["target_us"]
    cov90 = None
    for nm, a in ALPHA.items():
        lo, hi = base_interval(q, LV, nm)
        sc = conformity(lo, hi, y)
        inits = {d: nearest_init(org[day != d], sc[day != d], org[day == d].min(), W_FROZEN) for d in np.unique(day)}
        clo, chi = rolling_conformal_daily(org, tgt, day, lo, hi, y, a, W_FROZEN, inits)
        cov = (y >= clo) & (y <= chi)
        isc = interval_score(clo, chi, y, a)
        out["intervals"][nm] = {g: {"coverage": float(cov[m].mean()), "width": float((chi - clo)[m].mean()),
                                    "interval_score": float(isc[m].mean())} for g, m in strata.items()}
        if nm == "90":
            cov90 = cov.astype(float)
    out["_cov90"] = cov90
    out["physics"] = physics_diagnostics(q / np.maximum(pct, 1)[:, None], k_true, k_now, y, pct, ramp, K_MAX)
    return out


def table(res: dict, arms) -> str:
    cols = [("Overall CRPS", lambda r: r["crps"]["overall"], "min"), ("Cloudy CRPS", lambda r: r["crps"]["cloudy"], "min"),
            ("Ramp CRPS", lambda r: r["crps"]["ramp"], "min"), ("RMSE", lambda r: r["point"]["rmse"], "min"),
            ("MAE", lambda r: r["point"]["mae"], "min"),
            ("90% Coverage", lambda r: r["intervals"]["90"]["overall"]["coverage"], "near90"),
            ("Ramp Coverage", lambda r: r["intervals"]["90"]["ramp"]["coverage"], "near90"),
            ("Sunrise Coverage", lambda r: r["intervals"]["90"]["sunrise ramp"]["coverage"], "near90")]
    lines = ["| Method | " + " | ".join(c[0] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    ranks = {}
    for name, f, mode in cols:
        key = (lambda a: f(res[a])) if mode == "min" else (lambda a: abs(f(res[a]) - 0.90))
        order = sorted(arms, key=key)
        ranks[name] = {order[0]: "best", order[1]: "second"} if len(order) > 1 else {order[0]: "best"}
    for a in arms:
        cells = []
        for name, f, _ in cols:
            v = f"{f(res[a]):.3f}"
            rk = ranks[name].get(a)
            cells.append(f"**{v}**" if rk == "best" else (f"*{v}*" if rk == "second" else v))
        lines.append(f"| {LABELS[a]} | " + " | ".join(cells) + " |")
    lines.append("\nBold = best, italic = second best (for coverage: closest to the 90% nominal level). "
                 "Development out-of-fold G3''-F; CRPS, RMSE, MAE in kW.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A0", "A1", "R", "A3", "A4", "A5"])
    a = ap.parse_args()
    arms = [x for x in a.arms if (ARM_DIRS[x] / "seed456" / "fold_4" / "preds_val.npz").exists()]
    print("arms available:", arms, flush=True)
    z = np.load(ROOT / "experiments/phaseC/oof_streams.npz")
    ctx = context(z)
    res, qs = {}, {}
    for arm in arms:
        qs[arm] = arm_oof(arm, z)
        res[arm] = evaluate(qs[arm], z, ctx)
        print(arm, {g: round(v, 4) for g, v in res[arm]["crps"].items()}, flush=True)
    if "A0" in res:
        a0 = res["A0"]
        dq = np.abs(qs["A0"] - z["q_G3ppF"]).max(1)  # stored-prediction rebuild vs checkpoint-recomputed stream
        print(f"A0 rebuild vs Phase C stream: max {dq.max():.2e}, mean {dq.mean():.2e}", flush=True)
        cands = [x for x in arms if x != "A0" and res[x]["crps"]["ramp"] <= a0["crps"]["ramp"]]
        best = min(cands, key=lambda x: res[x]["crps"]["overall"]) if cands else None
        boot = {}
        if best:
            day, sunny, ramp = z["day"], z["sunny"].astype(bool), z["ramp"].astype(bool)
            d = res[best]["_crps_per_sample"] - a0["_crps_per_sample"]
            boot = {"best_arm": best,
                    "dCRPS_overall": day_bootstrap(d, day, sunny),
                    "dCRPS_cloudy": day_bootstrap(d[~sunny], day[~sunny], sunny[~sunny]),
                    "dCRPS_ramp": day_bootstrap(d[ramp], day[ramp], sunny[ramp]),
                    "dCov90_ramp": day_bootstrap((res[best]["_cov90"] - a0["_cov90"])[ramp], day[ramp], sunny[ramp])}
        res["comparison_vs_A0"] = boot or {"best_arm": None, "note": "no arm with ramp CRPS <= A0"}
    clean = {k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v) for k, v in res.items()}
    (ROOT / "reports/phaseF_results.json").write_text(json.dumps(clean, indent=1, default=float))
    (ROOT / "reports/phaseF_table.md").write_text(table(res, arms))
    print(table(res, arms))
    print(json.dumps(clean.get("comparison_vs_A0", {}), indent=1, default=float))
    from scripts.physics_figures import make_figures  # noqa: E402
    make_figures(res, qs, z, ctx, arms)


if __name__ == "__main__":
    main()
