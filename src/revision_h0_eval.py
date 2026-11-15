"""Revision R-H0 (pilot): evaluate the arms on development out-of-fold forecasts
(protocol revision/protocols/R-H0_resolution_pilot.md, amendment 1). CPU only.

Primary: ramp-region threshold-weighted CRPS (R-Z2 definition) of treatment minus control at each lead, paired
day-block bootstrap; GO rule and the lead-time dissociation of the relative gains.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.clear_sky import ClearSkyPV
from src.revision_common import save
from src.revision_h0_train import lead_targets
from src.revision_z2_ramp_weighted_crps import tw_crps
from src.skippd import SkippdForecastDataset

CONTRASTS = (("FULL-I128", "FULL-I64"), ("FULL-I256", "FULL-I64"), ("SUN-hi", "SUN-ctl"), ("RING-hi", "RING-ctl"))
LEADS, FOLDS, MED = (5, 15), (0, 1, 3), 4  # folds 2 and 4 have no validation day in the pilot months
H5, MAN = "data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz"


def thresholds(runs):
    """Ramp threshold per fold and lead: 90th percentile of |K(t+lead) - K(t)| over all fit-day development samples
    with the fold's fit-day clear-sky anchor (the frozen 15-min rule; the 15-min values must reproduce the frozen file)."""
    ds = SkippdForecastDataset(H5, MAN, "dev", preload=False)
    m = np.load(MAN)
    rec = m["times_us"].astype("datetime64[us]")[m["hdf5_group"] == 0][np.argsort(m["hdf5_row"][m["hdf5_group"] == 0])]
    pv = ds.pv.numpy().astype(float)
    org = ds.origin_time.astype("datetime64[us]").astype(np.int64)
    tgt = {15: ds.target_rows, 5: lead_targets(rec.astype(np.int64), org, 5)}
    frozen = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    day = ds.day.astype(str)
    tau = {}
    for k in FOLDS:
        fd = runs / "B64" / "lead15" / "seed42" / f"fold_{k}"
        coef = json.loads((fd / "metrics.json").read_text())["clear_sky_coef"]
        fit = np.isin(day, json.loads((fd / "calibration_split.json").read_text())["fit_days"])
        kk = pv / np.maximum(ClearSkyPV(coef)(rec), 1.0)
        for L in LEADS:
            ok = fit & (tgt[L] >= 0)
            tau[(k, L)] = float(np.quantile(np.abs(kk[tgt[L][ok]] - kk[ds.hist_rows[ok, -1]]), 0.9))
        assert abs(tau[(k, 15)] - frozen[str(k)]["training_labels_fit_days"]) < 1e-5, f"STOP: frozen 15-min threshold not reproduced (fold {k})"
    return tau


def load(runs, arm, L, tau):
    parts = []
    for k in FOLDS:
        d = dict(np.load(runs / arm / f"lead{L}" / "seed42" / f"fold_{k}" / "preds_val.npz"))
        pc, kn = np.maximum(d["pclear_target_kw"], 1.0), d["pv_now_kw"] / np.maximum(d["pclear_now_kw"], 1.0)
        t = tau[(k, L)]
        crps, twr, twc = tw_crps(d["pred_q"], d["target"], (kn - t) * pc, (kn + t) * pc)
        dk = d["target"] / pc - kn
        parts.append({"sid": d["sample_ids"], "day": d["day"].astype(str), "clear": d["clear_day"].astype(bool), "crps": crps, "tw_ramp": twr, "tw_calm": twc,
                      "ramp": np.abs(dk) >= t, "pers": np.abs(d["persistence"] - d["target"]), "tgt_us": d["target_time_us"], "y": d["target"],
                      "med": d["pred_q"][:, MED], "med_ramp": np.abs(d["pred_q"][:, MED] / pc - kn) >= t, "dk_med": d["pred_q"][:, MED] / pc - kn, "dk": dk})
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def boot(stat, day, clear, n_boot=10_000, seed=20261022):
    """Day-block bootstrap (stratified sunny / cloudy days) of stat(day-index array) built from per-day sums."""
    days, inv = np.unique(day, return_inverse=True)
    sunny = np.zeros(len(days), bool)
    sunny[inv] = clear
    groups = [np.flatnonzero(sunny), np.flatnonzero(~sunny)]
    rng = np.random.default_rng(seed)
    b = np.array([stat(np.concatenate([rng.choice(g, len(g)) for g in groups if len(g)])) for _ in range(n_boot)])
    return [float(stat(np.arange(len(days)))), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))], inv, len(days)


def lag_minutes(a, max_steps=10):
    """Lag (min) at which changes of the median best match observed changes (samples 2 min apart within a day)."""
    best = {}
    for s in range(0, max_steps + 1):
        num = den_a = den_b = 0.0
        for d in np.unique(a["day"]):
            m = np.flatnonzero(a["day"] == d)
            m = m[np.argsort(a["tgt_us"][m])]
            ok = np.diff(a["tgt_us"][m]) == 120_000_000
            dy, dm = np.diff(a["y"][m])[ok], np.diff(a["med"][m])[ok]
            if len(dy) > s + 5:
                x, z = dm[s:], dy[:len(dy) - s]  # forecast change s steps later against the observed change
                num += float((x * z).sum()); den_a += float((x * x).sum()); den_b += float((z * z).sum())
        best[s * 2] = num / np.sqrt(den_a * den_b) if den_a > 0 and den_b > 0 else np.nan
    return int(max(best, key=lambda k: best[k])), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="D:/skippd_hires/runs")
    a = ap.parse_args()
    runs = Path(a.runs)
    tau = thresholds(runs)
    arms = sorted({x for c in CONTRASTS for x in c} | {"B64"})
    res = {"protocol": "revision/protocols/R-H0_resolution_pilot.md (amendment 1)", "label": "development out-of-fold, pilot months, seed 42",
           "tau": {f"fold{k}_lead{L}": v for (k, L), v in tau.items()}, "arms": {}, "contrasts": {}, "dissociation": {}, "decision": {},
           "deviations": ["controls are the same raw frame area-averaged to 64 x 64 (stricter than the benchmark frame named in the protocol; fixed before training)",
                          "ramp recall by onset-lead bin and the patchy-cloud subset are not computed in this evaluation",
                          "out-of-fold forecasts come from folds 0, 1 and 3, which contain all 147 pilot days as validation days; folds 2 and 4 "
                          "have no validation day in the pilot months and are not trained (found at the first training run, before any contrast was computed)"]}
    data = {(arm, L): load(runs, arm, L, tau) for arm in arms for L in LEADS if (runs / arm / f"lead{L}" / "seed42" / "fold_3" / "preds_val.npz").exists()}
    ref = next(iter(data.values()))
    for key, d in data.items():
        same = [k for k in data if k[1] == key[1]][0]
        assert np.array_equal(d["sid"], data[same]["sid"]), f"STOP: samples differ between arms ({key})"
        lag, curve = lag_minutes(d)
        r = d["ramp"]
        res["arms"][f"{key[0]}|lead{key[1]}"] = {
            "n": int(len(r)), "ramp_share": float(r.mean()), "crps": float(d["crps"].mean()), "tw_ramp": float(d["tw_ramp"].mean()), "tw_calm": float(d["tw_calm"].mean()),
            "crps_skill_vs_smart_persistence": float(1 - d["crps"].mean() / d["pers"].mean()), "median_lag_min": lag,
            "ramp_recall_of_median": float(d["med_ramp"][r].mean()), "ramp_magnitude_ratio": float(np.abs(d["dk_med"][r]).mean() / np.abs(d["dk"][r]).mean())}
    for trt, ctl in CONTRASTS:
        gains = {}
        for L in LEADS:
            if (trt, L) not in data or (ctl, L) not in data:
                continue
            t, c = data[(trt, L)], data[(ctl, L)]
            out = {}
            for part in ("tw_ramp", "tw_calm", "crps"):
                diff = t[part] - c[part]
                days, inv = np.unique(t["day"], return_inverse=True)
                s, n = np.bincount(inv, weights=diff), np.bincount(inv)
                out[part], _, _ = boot(lambda ii: s[ii].sum() / n[ii].sum(), t["day"], t["clear"])
            rel = -out["tw_ramp"][0] / c["tw_ramp"].mean()
            go = out["tw_ramp"][2] < 0 and rel >= 0.03 and out["crps"][0] <= 0.01 * c["crps"].mean()
            out |= {"relative_ramp_gain": float(rel), "relative_crps_change": float(out["crps"][0] / c["crps"].mean()), "go": bool(go)}
            res["contrasts"][f"{trt} - {ctl}|lead{L}"] = out
            gains[L] = (t["tw_ramp"] - c["tw_ramp"], c["tw_ramp"], t["day"], t["clear"])
        res["decision"][trt] = "GO" if any(res["contrasts"].get(f"{trt} - {ctl}|lead{L}", {}).get("go") for L in LEADS) else "NO-GO"
        if len(gains) == 2:  # relative gain at 5 min minus relative gain at 15 min (same samples and days at both leads)
            d5, c5, day, clear = gains[5]
            d15, c15, _, _ = gains[15]
            days, inv = np.unique(day, return_inverse=True)
            S = [np.bincount(inv, weights=v) for v in (d5, c5, d15, c15)]
            stat = lambda ii: -S[0][ii].sum() / S[1][ii].sum() + S[2][ii].sum() / S[3][ii].sum()
            res["dissociation"][trt] = {"gain5_minus_gain15": boot(stat, day, clear)[0]}
    for k, v in res["arms"].items():
        print(f"{k:18s} CRPS {v['crps']:.4f}  ramp region {v['tw_ramp']:.4f}  calm {v['tw_calm']:.4f}  skill {v['crps_skill_vs_smart_persistence']:.3f}  lag {v['median_lag_min']} min  recall {v['ramp_recall_of_median']:.3f}")
    f = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"
    for k, v in res["contrasts"].items():
        print(f"{k:28s} ramp region {f(v['tw_ramp'])} ({v['relative_ramp_gain']:+.1%})  CRPS {f(v['crps'])}  => {'GO' if v['go'] else 'no-go'}")
    for k, v in res["dissociation"].items():
        print(f"dissociation {k}: relative ramp gain at 5 min minus at 15 min {f(v['gain5_minus_gain15'])}")
    print("decision:", res["decision"])
    print("saved", save("h0_resolution_pilot.json", res))


if __name__ == "__main__":
    main()
