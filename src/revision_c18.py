"""Revision C15 / C18: event-probability scoring and median-based ramp metrics.

Protocol: revision/protocols/R-C18_events.md.
Development: out-of-fold validation predictions (per seed, folds concatenated). Test (post hoc): 15-member
ensembles from the revision member dump (experiments/R_posthoc), if present.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.e8_features import FEAT
from src.events import implied_event_probability
from src.gate7b import apply_gate, average_precision, k_quantiles, raw_features
from src.revision_common import LEVELS, save

ROOTS = {"E4-B-fit": Path("experiments/E4_calib/v1/csi"), "E6-S": Path("experiments/E6/v1/E6-S/csi"),
         "A2": Path("experiments/E9/v1/A2/csi")}
G3PP = Path("experiments/E9/v1/G3pp_params")


def roc_auc(score, label) -> float:
    label = np.asarray(label, bool)
    r = np.argsort(np.argsort(score, kind="stable"), kind="stable") + 1.0
    npos, nneg = label.sum(), (~label).sum()
    return float((r[label].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))


def prob_scores(p, lab, clim) -> dict:
    p, lab = np.asarray(p, float), np.asarray(lab, bool)
    bs = float(np.mean((p - lab) ** 2))
    bs_c = float(np.mean((clim - lab) ** 2))
    edges = np.quantile(p, np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    rel = [{"mean_p": float(p[b == i].mean()), "obs_rate": float(lab[b == i].mean()), "n": int((b == i).sum())}
           for i in range(10) if (b == i).any()]
    conf = {}
    for name, thr in (("0.5", 0.5), ("base_rate", clim)):
        pr = p >= thr
        conf[name] = {"threshold": float(thr), "tp": int((pr & lab).sum()), "fp": int((pr & ~lab).sum()),
                      "fn": int((~pr & lab).sum()), "tn": int((~pr & ~lab).sum())}
    # PR curve (20 recall points)
    order = np.argsort(-p, kind="stable")
    tp = np.cumsum(lab[order])
    prec, rec = tp / np.arange(1, len(p) + 1), tp / max(lab.sum(), 1)
    pts = [{"recall": float(rec[i]), "precision": float(prec[i])} for i in np.unique(np.searchsorted(rec, np.linspace(0.05, 1, 20)).clip(0, len(p) - 1))]
    return {"n": int(len(p)), "base_rate": float(lab.mean()), "climatology": float(clim), "brier": bs, "brier_climatology": bs_c,
            "brier_skill": 1 - bs / bs_c if bs_c > 0 else None, "auprc": average_precision(p, lab), "roc_auc": roc_auc(p, lab),
            "reliability": rel, "confusion": conf, "pr_curve": pts}


def ramp_metrics(q_k, k_now, k_tgt, tau) -> dict:
    """Median-based detection, direction and change slope (definitions in the protocol)."""
    dhat = q_k[:, list(LEVELS).index(0.5)] - k_now
    d = k_tgt - k_now
    obs, pred = np.abs(d) >= tau, np.abs(dhat) >= tau
    tp, fp, fn, tn = (pred & obs).sum(), (pred & ~obs).sum(), (~pred & obs).sum(), (~pred & ~obs).sum()
    dir_ok = obs & (dhat != 0)
    slope = lambda a, b: float(np.polyfit(a, b, 1)[0]) if len(a) > 2 else None
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "recall": float(tp / max(tp + fn, 1)),
            "precision": float(tp / max(tp + fp, 1)), "false_alarm_rate": float(fp / max(fp + tn, 1)),
            "direction_accuracy": float(np.mean(np.sign(dhat[dir_ok]) == np.sign(d[dir_ok]))), "direction_denominator": int(dir_ok.sum()),
            "change_slope_all": slope(d, dhat), "change_slope_ramps": slope(d[obs], dhat[obs])}


def load(root, s, k, split="val", keep=None):
    d = dict(np.load(root / f"seed{s}" / f"fold_{k}" / f"preds_{split}.npz"))
    if keep is None:
        return d
    m = np.isin(d["sample_ids"], keep)
    return {kk: (v[m] if np.ndim(v) and len(v) == len(m) else v) for kk, v in d.items()}


def development() -> dict:
    frozen = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = win["sample_ids"][win["eligible_common"]]
    out = {}
    for s in (42, 123, 456):
        acc = {key: [] for key in ("pc", "lab_c", "valid", "g", "pmix", "lab_e", "fold", "w", "lab_e_common", "g_common")}
        rm = {m: {"q": [], "kn": [], "kt": [], "tau": []} for m in ("E4-B-fit", "E6-S", "A2", "G3''")}
        cross_rates = {}
        for k in range(5):
            tau = frozen[str(k)]["training_labels_fit_days"]
            r = load(ROOTS["E6-S"], s, k)
            kn = r["pv_now_kw"] / np.maximum(r["pclear_now_kw"], 1.0)
            kt = r["target"] / np.maximum(r["pclear_target_kw"], 1.0)
            qk = k_quantiles(r)
            acc["pc"].append(1 - r["time_probs"][:, 3]); acc["lab_c"].append(r["time_class"] != 3); acc["valid"].append(r["event_valid"])
            acc["g"].append(np.concatenate([implied_event_probability(qk[i:i + 8192], LEVELS, kn[i:i + 8192], tau) for i in range(0, len(kn), 8192)]))
            acc["pmix"].append(r["p_abs_change_ge_tau"]); acc["lab_e"].append(np.abs(kt - kn) >= tau); acc["fold"].append(np.full(len(kn), k))
            cross_rates[k] = float((r["time_class"] != 3)[r["event_valid"]].mean())
            for m in ("E4-B-fit", "E6-S"):
                d = load(ROOTS[m], s, k) if m != "E6-S" else r
                assert np.array_equal(d["sample_ids"], r["sample_ids"])
                rm[m]["q"].append(k_quantiles(d)); rm[m]["kn"].append(kn); rm[m]["kt"].append(kt); rm[m]["tau"].append(np.full(len(kn), tau))
            # common population: A2, G3'' weight
            rc, ac = load(ROOTS["E6-S"], s, k, keep=common), load(ROOTS["A2"], s, k, keep=common)
            assert np.array_equal(rc["sample_ids"], ac["sample_ids"])
            p = json.loads((G3PP / f"seed{s}" / f"fold_{k}" / "params.json").read_text())
            w = apply_gate(p, raw_features(ac, rc, tau))
            knc = rc["pv_now_kw"] / np.maximum(rc["pclear_now_kw"], 1.0)
            ktc = rc["target"] / np.maximum(rc["pclear_target_kw"], 1.0)
            acc["w"].append(w); acc["lab_e_common"].append(np.abs(ktc - knc) >= tau)
            qg = w[:, None] * k_quantiles(ac) + (1 - w[:, None]) * k_quantiles(rc)
            for m, q in (("A2", k_quantiles(ac)), ("G3''", qg)):
                rm[m]["q"].append(q); rm[m]["kn"].append(knc); rm[m]["kt"].append(ktc); rm[m]["tau"].append(np.full(len(knc), tau))
        a = {key: np.concatenate(v) for key, v in acc.items() if v}
        fold = a["fold"]
        # climatology: crossing rate from the OTHER folds' validation labels; endpoint rate likewise (out-of-fold)
        ev_rate = {k: float(a["lab_e"][fold == k].mean()) for k in range(5)}
        clim_c = np.array([np.mean([cross_rates[j] for j in range(5) if j != f]) for f in fold])
        clim_e = np.array([np.mean([ev_rate[j] for j in range(5) if j != f]) for f in fold])
        v = a["valid"]
        res = {"crossing: P_cross (time-class head)": prob_scores(a["pc"][v], a["lab_c"][v], float(np.mean(clim_c[v]))),
               "endpoint: g (9 quantiles)": prob_scores(a["g"], a["lab_e"], float(np.mean(clim_e))),
               "endpoint: p_mix (mixture CDF)": prob_scores(a["pmix"], a["lab_e"], float(np.mean(clim_e))),
               "endpoint (common): G3'' weight w on A2": prob_scores(a["w"], a["lab_e_common"], float(np.mean(clim_e))),
               "cross-check: P_cross scored against endpoint label": prob_scores(a["pc"], a["lab_e"], float(np.mean(clim_e)))}
        res["ramp_metrics"] = {m: ramp_metrics(np.concatenate(x["q"]), np.concatenate(x["kn"]), np.concatenate(x["kt"]),
                                               np.concatenate(x["tau"])) for m, x in rm.items()}
        res["g_vs_pmix"] = {"corr": float(np.corrcoef(a["g"], a["pmix"])[0, 1]), "mean_abs_diff": float(np.abs(a["g"] - a["pmix"]).mean())}
        out[str(s)] = res
        print(s, {kk: (round(vv["brier_skill"], 3), round(vv["auprc"], 3)) for kk, vv in res.items() if isinstance(vv, dict) and "brier" in vv})
    return out


def main():
    res = {"protocol": "revision/protocols/R-C18_events.md", "development": development()}
    p = save("c18_events_dev.json", res)
    print("saved", p)


if __name__ == "__main__":
    main()
