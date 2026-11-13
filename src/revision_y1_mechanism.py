"""Revision R-Y1: does expert disagreement explain the probability-mixture advantage (protocol
revision/protocols/R-Y1_disagreement_mechanism.md). Development (out-of-fold, per seed) and post hoc test.

For the CRPS, a two-component linear pool satisfies CRPS(LP_w) = w CRPS(A) + (1-w) CRPS(R) - w(1-w) D(A, R),
with D the Cramer divergence (Knuppel & Kruger 2022; Kruger 2024). On the 200-atom grid, D is MixCRPS.c2.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from src.e8_features import FEAT
from src.gate2_report import ramp_labels, strata
from src.gate7b import apply_gate, grid_quantiles, raw_features
from src.revision_common import LEVELS, save
from src.revision_x2_linear_pool import A2, E6S, TAU, MixCRPS, atom_crps, g3pp_params, load

SEEDS = (42, 123, 456)
NB = 10


def terms(qa, qr, y, w):
    """Per-sample exact atom CRPS of the quantile blend (QB), the linear pool (LP), the components, and D."""
    ga, gr = grid_quantiles(qa, LEVELS), grid_quantiles(qr, LEVELS)
    mc = MixCRPS(ga, gr, y)
    lp = mc.per_sample(torch.as_tensor(w)).numpy()
    qb = atom_crps(w[:, None] * ga + (1 - w[:, None]) * gr, y)
    ca, cr = atom_crps(ga, y), atom_crps(gr, y)
    D = mc.c2.numpy()
    ident = np.abs(lp - (w * ca + (1 - w) * cr - w * (1 - w) * D)).max()
    assert ident < 1e-9, f"STOP: CRPS disagreement identity fails ({ident})"
    return {"delta": qb - lp, "Dterm": w * (1 - w) * D, "D": D, "ident": float(ident)}


def boot_top_bottom(delta, x, edges, day, clear, n_boot=10_000, seed=20261022):
    """Paired day-block bootstrap (stratified by clear/cloudy day) of mean delta in the top bin minus the bottom bin."""
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, NB - 1)
    top, bot = b == NB - 1, b == 0
    days, inv = np.unique(day, return_inverse=True)
    st, ct = np.bincount(inv, delta * top, len(days)), np.bincount(inv, top.astype(float), len(days))
    sb, cb = np.bincount(inv, delta * bot, len(days)), np.bincount(inv, bot.astype(float), len(days))
    cl = np.zeros(len(days), bool)
    cl[inv] = clear
    groups = [np.flatnonzero(cl), np.flatnonzero(~cl)]
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        ii = np.concatenate([rng.choice(g, len(g)) for g in groups if len(g)])
        bs[i] = st[ii].sum() / max(ct[ii].sum(), 1) - sb[ii].sum() / max(cb[ii].sum(), 1)
    est = st.sum() / ct.sum() - sb.sum() / cb.sum()
    return [float(est), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def decile_table(delta, x, edges, ramp):
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, NB - 1)
    rows = [{"bin": j, "n": int((b == j).sum()), "x_mean": float(x[b == j].mean()), "delta_mean_kw": float(delta[b == j].mean()),
             "ramp_share": float(ramp[b == j].mean())} for j in range(NB)]
    rho = float(spearmanr(np.arange(NB), [r["delta_mean_kw"] for r in rows]).statistic)
    return rows, rho


def analyse(t, d_feat, day, clear, ramp, edges_D=None, edges_d=None):
    edges_D = np.quantile(t["Dterm"], np.linspace(0, 1, NB + 1)) if edges_D is None else edges_D
    edges_d = np.quantile(d_feat, np.linspace(0, 1, NB + 1)) if edges_d is None else edges_d
    rows, rho = decile_table(t["delta"], t["Dterm"], edges_D, ramp)
    rows_d, rho_d = decile_table(t["delta"], d_feat, edges_d, ramp)
    order = np.argsort(-t["Dterm"])
    top20 = order[: len(order) // 5]
    tot = t["delta"].sum()
    return {"n": int(len(day)), "mean_delta_kw": float(t["delta"].mean()), "identity_max_abs": t["ident"],
            "deciles_Dterm": rows, "spearman_Dterm": rho, "top_minus_bottom_Dterm": boot_top_bottom(t["delta"], t["Dterm"], edges_D, day, clear),
            "deciles_d": rows_d, "spearman_d": rho_d, "top_minus_bottom_d": boot_top_bottom(t["delta"], d_feat, edges_d, day, clear),
            "share_of_LP_advantage_top20pct_Dterm": float(t["delta"][top20].sum() / tot) if tot != 0 else None,
            "mean_D": {"ramp": float(t["D"][ramp].mean()), "non-ramp": float(t["D"][~ramp].mean())},
            "mean_delta": {"ramp": float(t["delta"][ramp].mean()), "non-ramp": float(t["delta"][~ramp].mean())}}


def development():
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = win["sample_ids"][win["eligible_common"]]
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    rl = ramp_labels("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz", folds_doc)
    out = {}
    for s in SEEDS:
        parts, metas, dfe = [], [], []
        for k in range(5):
            tau = TAU[str(k)]["training_labels_fit_days"]
            rv, dv = load(E6S, s, k, "val", common), load(A2, s, k, "val", common)
            assert np.array_equal(dv["sample_ids"], rv["sample_ids"])
            fv = raw_features(dv, rv, tau)
            w = apply_gate(g3pp_params(s, k), fv)
            parts.append(terms(dv["pred_q"], rv["pred_q"], rv["target"], w))
            dfe.append(np.abs(dv["pred_q"].mean(1) - rv["pred_q"].mean(1)))
            metas.append(rv)
        t = {key: np.concatenate([p[key] for p in parts]) for key in ("delta", "Dterm", "D")}
        t["ident"] = max(p["ident"] for p in parts)
        m = {key: np.concatenate([v[key] for v in metas]) for key in ("sample_ids", "target", "clear_day", "fold", "day")}
        ramp = strata(m, rl["is_ramp"])["ramp"]
        out[str(s)] = analyse(t, np.concatenate(dfe), m["day"].astype(str), m["clear_day"].astype(bool), ramp)
        r = out[str(s)]
        print(f"dev seed {s}: mean delta {r['mean_delta_kw']:+.4f}  spearman(D) {r['spearman_Dterm']:.2f}  top-bottom {r['top_minus_bottom_Dterm']}", flush=True)
    sup = sum(out[str(s)]["spearman_Dterm"] > 0.8 and out[str(s)]["top_minus_bottom_Dterm"][1] > 0 for s in SEEDS)
    nos = sum(out[str(s)]["spearman_Dterm"] <= 0.5 or out[str(s)]["top_minus_bottom_Dterm"][1] <= 0 <= out[str(s)]["top_minus_bottom_Dterm"][2] for s in SEEDS)
    return {"per_seed": out, "seeds_supporting": sup, "seeds_not_supporting": nos}


def posthoc_test():
    from src.final_eval import gate_params
    from src.revision_common import test_context
    from src.revision_posthoc_test import mdict, members, sub
    c = test_context()
    h60 = c["h60"]
    y = c["y"][h60]
    r, a = members("E6-S"), members("A2")
    acc = {"delta": [], "Dterm": [], "D": []}
    dfe, ident = [], 0.0
    for i, (s, k) in enumerate(r["members"]):
        tau = TAU[str(k)]["training_labels_fit_days"]
        mr, ma = sub(mdict(r, i), h60), sub(mdict(a, i), h60)
        w = apply_gate(gate_params("G3''", s, k), raw_features(ma, mr, tau))
        t = terms(ma["pred_q"], mr["pred_q"], y, w)
        ident = max(ident, t["ident"])
        for key in acc:
            acc[key].append(t[key])
        dfe.append(np.abs(ma["pred_q"].mean(1) - mr["pred_q"].mean(1)))
    t = {key: np.mean(v, 0) for key, v in acc.items()} | {"ident": ident}
    return analyse(t, np.mean(dfe, 0), c["day"][h60], c["sunny"][h60], c["ramp"][h60])


def main():
    res = {"protocol": "revision/protocols/R-Y1_disagreement_mechanism.md", "development": development()}
    res["posthoc_test"] = {"label": "POST HOC (test; per-member exact atom CRPS averaged over 15 members)", **posthoc_test()}
    d = res["development"]
    tst = res["posthoc_test"]
    same_sign = tst["top_minus_bottom_Dterm"][0] > 0
    if d["seeds_supporting"] >= 2 and same_sign:
        verdict = "supported"
    elif d["seeds_not_supporting"] >= 2:
        verdict = "not supported"
    else:
        verdict = "partly supported"
    res["verdict"] = verdict
    p = save("y1_disagreement_mechanism.json", res)
    print("saved", p, "| verdict:", verdict)
    print("test: mean delta", round(tst["mean_delta_kw"], 4), "spearman(D)", round(tst["spearman_Dterm"], 2),
          "top-bottom", [round(x, 4) for x in tst["top_minus_bottom_Dterm"]], "ramp/non-ramp delta", tst["mean_delta"])


if __name__ == "__main__":
    main()
