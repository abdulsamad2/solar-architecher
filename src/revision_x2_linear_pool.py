"""Revision R-X2: gated linear pool (probability mixture) vs the gated quantile blend (protocol
revision/protocols/R-X2_linear_pool.md). Development (out-of-fold validation, per seed) and post hoc test.

Each expert = its 200-point grid quantile function (gate7b.grid_quantiles), i.e. 200 equally weighted atoms.
Mixture CRPS is exact and quadratic in w: CRPS(w) = c0 + c1 w + c2 w^2 (energy form E|X-y| - 0.5 E|X-X'|).
Primary score: 9 nominal quantiles (mixture CDF inverted) scored with crps_quantile_interp like every model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from src.e8_features import FEAT
from src.final_eval import day_bootstrap
from src.gate2_report import ramp_labels, strata
from src.gate7b import (LAMBDAS, TAUS, apply_gate, fit_gate, gate_weights, grid_quantiles, raw_features, standardize,
                        transform)
from src.metrics import crps_quantile_interp
from src.revision_common import LEVELS, save

A2 = Path("experiments/E9/v1/A2/csi")
E6S = Path("experiments/E6/v1/E6-S/csi")
STRATA = ("overall", "sunny", "cloudy", "ramp", "non-ramp")
METHODS = ("G3''", "LP-frozen", "LP-refit", "LP-const")
TAU = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())


# ---------- exact atom-grid CRPS ----------
def _mean_abs_self(x):  # x sorted [N, n]: mean_{i,j} |x_i - x_j|
    n = x.shape[1]
    c = 2 * torch.arange(1, n + 1, dtype=x.dtype) - n - 1
    return 2 * (x * c).sum(1) / n ** 2


def _mean_abs_cross(a, b):  # a, b sorted [N, n]: mean_{i,j} |a_i - b_j|
    n = b.shape[1]
    cs = torch.cat([torch.zeros(len(b), 1, dtype=b.dtype), b.cumsum(1)], 1)
    k = torch.searchsorted(b.contiguous(), a.contiguous(), right=True)  # #b <= a_i
    below = torch.gather(cs, 1, k)
    s = a * k - below + (cs[:, -1:] - below) - a * (n - k)
    return s.sum(1) / n ** 2


def _mean_abs_y(x, y):
    return (x - y[:, None]).abs().mean(1)


class MixCRPS:
    """Exact CRPS of the mixture w*A + (1-w)*R of two 200-atom distributions, quadratic in w."""

    def __init__(self, qa_grid, qr_grid, y):
        a = torch.as_tensor(np.sort(qa_grid, 1), dtype=torch.float64)
        r = torch.as_tensor(np.sort(qr_grid, 1), dtype=torch.float64)
        yt = torch.as_tensor(y, dtype=torch.float64)
        ay, ry = _mean_abs_y(a, yt), _mean_abs_y(r, yt)
        aa, rr, ar = _mean_abs_self(a), _mean_abs_self(r), _mean_abs_cross(a, r)
        self.c0, self.c1, self.c2 = ry - 0.5 * rr, ay - ry + rr - ar, ar - 0.5 * (aa + rr)

    def per_sample(self, w):
        return self.c0 + self.c1 * w + self.c2 * w ** 2


def atom_crps(atoms, y):
    x = torch.as_tensor(np.sort(atoms, 1), dtype=torch.float64)
    return (_mean_abs_y(x, torch.as_tensor(y, dtype=torch.float64)) - 0.5 * _mean_abs_self(x)).numpy()


# ---------- mixture quantiles ----------
def _cdf(x, q, right=True):
    """Piecewise-linear CDF of 9 quantiles with constant tails (point masses 0.05 at q05 and q95), at x [N, M]."""
    lv = LEVELS
    out = np.where(x >= q[:, -1:], 1.0, 0.0) if right else np.where(x > q[:, -1:], 1.0, 0.0)
    for j in range(len(lv) - 1):
        lo, hi = q[:, j:j + 1], q[:, j + 1:j + 2]
        sel = (x >= lo) & (x < hi) if right else (x > lo) & (x <= hi)
        val = lv[j] + (x - lo) / np.maximum(hi - lo, 1e-12) * (lv[j + 1] - lv[j])
        out = np.where(sel, val, out)
    return out


def mixture_quantiles(qa, qr, w, levels=LEVELS):
    knots = np.sort(np.concatenate([qa, qr], 1), 1)
    fl = w[:, None] * _cdf(knots, qa, False) + (1 - w[:, None]) * _cdf(knots, qr, False)
    fr = w[:, None] * _cdf(knots, qa, True) + (1 - w[:, None]) * _cdf(knots, qr, True)
    xs = np.repeat(knots, 2, 1)
    fs = np.empty_like(xs)
    fs[:, 0::2], fs[:, 1::2] = fl, fr
    fs = np.maximum.accumulate(fs, 1)
    out = np.empty((len(w), len(levels)))
    rows = np.arange(len(w))
    for i, p in enumerate(levels):
        k = np.clip((fs < p - 1e-12).sum(1), 0, xs.shape[1] - 1)  # first point with F >= p
        km = np.maximum(k - 1, 0)
        f0, f1, x0, x1 = fs[rows, km], fs[rows, k], xs[rows, km], xs[rows, k]
        t = np.where((k > 0) & (f1 > f0), (p - f0) / np.maximum(f1 - f0, 1e-15), 1.0)
        out[:, i] = x0 + np.clip(t, 0, 1) * (x1 - x0)
    return out


def self_test():
    rng = np.random.default_rng(0)
    qa = np.sort(rng.normal(10, 3, (50, 9)), 1)
    qr = np.sort(rng.normal(12, 1, (50, 9)), 1)
    assert np.allclose(mixture_quantiles(qa, qr, np.ones(50)), qa, atol=1e-9)
    assert np.allclose(mixture_quantiles(qa, qr, np.zeros(50)), qr, atol=1e-9)
    # exact mixture CRPS against brute force on the atoms
    ga, gr, y = grid_quantiles(qa, LEVELS), grid_quantiles(qr, LEVELS), rng.normal(11, 3, 50)
    w = rng.random(50)
    mc = MixCRPS(ga, gr, y).per_sample(torch.as_tensor(w)).numpy()
    for i in range(5):
        atoms = np.concatenate([ga[i], gr[i]])
        wt = np.concatenate([np.full(200, w[i] / 200), np.full(200, (1 - w[i]) / 200)])
        e1 = (wt * np.abs(atoms - y[i])).sum()
        e2 = (wt[:, None] * wt[None, :] * np.abs(atoms[:, None] - atoms[None, :])).sum()
        assert abs(mc[i] - (e1 - 0.5 * e2)) < 1e-9, (mc[i], e1 - 0.5 * e2)


# ---------- gate fitting (same machinery as gate7b.select_and_fit, mixture objective) ----------
def select_and_fit_mix(raw_cal, names, qa_cal, qr_cal, y_cal, order):
    half = len(order) // 2
    a, b = order[:half], order[half:]
    zc = transform(raw_cal, names)
    ga, gr = grid_quantiles(qa_cal, LEVELS), grid_quantiles(qr_cal, LEVELS)
    scores = {}
    for lam in LAMBDAS:
        (za, zb), _ = standardize(zc[a], zc[b])
        b0, beta = fit_gate(za, MixCRPS(ga[a], gr[a], y_cal[a]), lam)
        w = torch.as_tensor(gate_weights(zb, b0, beta))
        scores[lam] = float(MixCRPS(ga[b], gr[b], y_cal[b]).per_sample(w).mean())
    best = min(scores.values())
    lam = max(l for l in LAMBDAS if scores[l] <= best + 1e-12)
    (zall,), (mu, sd) = standardize(zc)
    b0, beta = fit_gate(zall, MixCRPS(ga, gr, y_cal), lam)
    return {"names": list(names), "lambda": lam, "lambda_scores": scores, "b0": b0, "beta": beta.tolist(),
            "mu": mu.tolist(), "sd": sd.tolist()}


def const_mix(qa_cal, qr_cal, y_cal):
    f = MixCRPS(grid_quantiles(qa_cal, LEVELS), grid_quantiles(qr_cal, LEVELS), y_cal)
    return float(minimize_scalar(lambda c: float(f.per_sample(torch.tensor(float(c), dtype=torch.float64)).mean()),
                                 bounds=(0, 1), method="bounded", options={"xatol": 1e-6}).x)


def fit_member(dc, rc, tau):
    """All fits for one (seed, fold) member from its calibration predictions (common 60-min population)."""
    fc = raw_features(dc, rc, tau)
    order = np.argsort(rc["origin_time_us"], kind="stable")
    return {"LP-refit": select_and_fit_mix(fc, ("g", "d"), dc["pred_q"], rc["pred_q"], rc["target"], order),
            "LP-const": const_mix(dc["pred_q"], rc["pred_q"], rc["target"])}


def combine(qa, qr, w_qb, fits, feats):
    """Quantiles of every method for one member on evaluation samples."""
    w_ref = apply_gate(fits["LP-refit"], feats)
    w_c = np.full(len(qa), fits["LP-const"])
    out = {"G3''": w_qb[:, None] * qa + (1 - w_qb[:, None]) * qr,
           "LP-frozen": mixture_quantiles(qa, qr, w_qb), "LP-refit": mixture_quantiles(qa, qr, w_ref),
           "LP-const": mixture_quantiles(qa, qr, w_c)}
    atoms = {"G3''": w_qb[:, None] * grid_quantiles(qa, LEVELS) + (1 - w_qb[:, None]) * grid_quantiles(qr, LEVELS)}
    return out, atoms, {"LP-refit": w_ref, "LP-const": w_c}


def load(root, s, k, split, keep):
    d = dict(np.load(root / f"seed{s}" / f"fold_{k}" / f"preds_{split}.npz"))
    m = np.isin(d["sample_ids"], keep)
    return {kk: (v[m] if np.ndim(v) and len(v) == len(m) else v) for kk, v in d.items()}


def g3pp_params(s, k):
    return json.loads(Path(f"experiments/E9/v1/G3pp_params/seed{s}/fold_{k}/params.json").read_text())


def development():
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = win["sample_ids"][win["eligible_common"]]
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    rl = ramp_labels("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz", folds_doc)
    recorded = json.loads(Path("reports/e9_gate.json").read_text())
    res = {"per_seed": {}, "fits": {}}
    for s in (42, 123, 456):
        qs = {m: [] for m in METHODS}
        atom_c = {"G3''": [], "LP-frozen": []}
        cov = {m: [] for m in METHODS}
        metas = []
        for k in range(5):
            tau = TAU[str(k)]["training_labels_fit_days"]
            rc, rv = load(E6S, s, k, "cal", common), load(E6S, s, k, "val", common)
            dc, dv = load(A2, s, k, "cal", common), load(A2, s, k, "val", common)
            assert np.array_equal(dc["sample_ids"], rc["sample_ids"]) and np.array_equal(dv["sample_ids"], rv["sample_ids"])
            fits = fit_member(dc, rc, tau)
            fv = raw_features(dv, rv, tau)
            w_qb = apply_gate(g3pp_params(s, k), fv)
            q, atoms, ws = combine(dv["pred_q"], rv["pred_q"], w_qb, fits, fv)
            res["fits"][f"{s}/{k}"] = {"LP-const_w": fits["LP-const"], "LP-refit": {kk: fits["LP-refit"][kk] for kk in ("lambda", "b0", "beta")},
                                      "mean_w": {"G3''": float(w_qb.mean()), "LP-refit": float(ws["LP-refit"].mean())}}
            for m in METHODS:
                qs[m].append(q[m])
            ga, gr = grid_quantiles(dv["pred_q"], LEVELS), grid_quantiles(rv["pred_q"], LEVELS)
            atom_c["G3''"].append(atom_crps(atoms["G3''"], rv["target"]))
            atom_c["LP-frozen"].append(MixCRPS(ga, gr, rv["target"]).per_sample(torch.as_tensor(w_qb)).numpy())
            metas.append(rv)
        cat = lambda key: np.concatenate([v[key] for v in metas])
        m = {key: cat(key) for key in ("sample_ids", "target", "clear_day", "fold", "day")}
        st = strata(m, rl["is_ramp"])
        y = m["target"]
        crps = {mm: crps_quantile_interp(np.concatenate(qs[mm]), y, LEVELS) for mm in METHODS}
        for g in ("overall", "ramp", "sunny", "cloudy"):
            want = recorded["per_seed"][str(s)]["raw_crps"]["G3''"][g]
            assert abs(float(crps["G3''"][st[g]].mean()) - want) < 1e-9, f"STOP: G3'' not reproduced ({s}, {g})"
        ac = {mm: np.concatenate(v) for mm, v in atom_c.items()}
        out = {"crps": {mm: {g: float(crps[mm][st[g]].mean()) for g in STRATA} for mm in METHODS},
               "atom_grid_crps": {mm: {g: float(ac[mm][st[g]].mean()) for g in STRATA} for mm in ac},
               "raw90": {mm: {g: {"coverage": float(np.mean((y[st[g]] >= np.concatenate(qs[mm])[st[g], 0]) & (y[st[g]] <= np.concatenate(qs[mm])[st[g], -1]))),
                                  "width_kw": float(np.mean((np.concatenate(qs[mm])[:, -1] - np.concatenate(qs[mm])[:, 0])[st[g]]))}
                             for g in ("overall", "ramp")} for mm in METHODS},
               "diff_G3pp_minus": {mm: {g: day_bootstrap((crps["G3''"] - crps[mm])[st[g]], m["day"][st[g]], m["clear_day"][st[g]])
                                        for g in STRATA} for mm in METHODS if mm != "G3''"}}
        res["per_seed"][str(s)] = out
        print(f"dev seed {s}: G3'' reproduced; " + "  ".join(f"{mm} {out['crps'][mm]['overall']:.4f}/{out['crps'][mm]['ramp']:.3f}" for mm in METHODS), flush=True)
    res["seed_mean"] = {mm: {g: float(np.mean([res["per_seed"][str(s)]["crps"][mm][g] for s in (42, 123, 456)])) for g in STRATA} for mm in METHODS}
    lp = [res["per_seed"][str(s)]["diff_G3pp_minus"]["LP-refit"]["overall"] for s in (42, 123, 456)]
    res["decision"] = {"seeds G3'' - LP-refit overall CI > 0 (LP better)": sum(c[1] > 0 for c in lp),
                       "seeds G3'' - LP-refit overall CI < 0 (QB better)": sum(c[2] < 0 for c in lp),
                       "seeds covering 0": sum(c[1] <= 0 <= c[2] for c in lp)}
    return res


def posthoc_test():
    from src.final_eval import gate_params
    from src.revision_common import day_bootstrap as db, test_context
    from src.revision_posthoc_test import mdict, members, sub
    c = test_context()
    h60 = c["h60"]
    y = c["y"][h60]
    r, a = members("E6-S"), members("A2")
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = win["sample_ids"][win["eligible_common"]]
    acc = {mm: [] for mm in METHODS}
    for i, (s, k) in enumerate(r["members"]):
        tau = TAU[str(k)]["training_labels_fit_days"]
        rc, dc = load(E6S, s, k, "cal", common), load(A2, s, k, "cal", common)
        assert np.array_equal(rc["sample_ids"], dc["sample_ids"])
        fits = fit_member(dc, rc, tau)
        mr, ma = sub(mdict(r, i), h60), sub(mdict(a, i), h60)
        fv = raw_features(ma, mr, tau)
        w_qb = apply_gate(gate_params("G3''", s, k), fv)
        q, _, _ = combine(ma["pred_q"], mr["pred_q"], w_qb, fits, fv)
        for mm in METHODS:
            acc[mm].append(q[mm])
    ens = {mm: np.mean(v, 0) for mm, v in acc.items()}
    frozen = np.load("experiments/final/test_predictions_v1.npz")
    masks = {g: v[h60] for g, v in c["masks"].items() if g in STRATA}
    day, sunny = c["day"][h60], c["sunny"][h60]
    crps = {mm: crps_quantile_interp(q, y, LEVELS) for mm, q in ens.items()}
    # the frozen G3''-F test forecast equals the G3'' blend on 60-min-eligible samples
    repro = float(np.abs(ens["G3''"] - frozen["q|G3''-F"][h60]).max())
    assert repro < 1e-6, f"STOP: G3'' test ensemble not reproduced ({repro})"
    return {"n": int(h60.sum()), "reproduction_max_abs_kw": repro,
            "crps": {mm: {g: float(cc[v].mean()) for g, v in masks.items()} for mm, cc in crps.items()},
            "raw90": {mm: {g: {"coverage": float(np.mean((y[v] >= q[v, 0]) & (y[v] <= q[v, -1]))), "width_kw": float(np.mean(q[v, -1] - q[v, 0]))}
                           for g, v in masks.items() if g in ("overall", "ramp")} for mm, q in ens.items()},
            "G3pp_minus": {mm: {g: db((crps["G3''"] - crps[mm])[v], day[v], sunny[v]) for g, v in masks.items()} for mm in METHODS if mm != "G3''"}}


def main():
    self_test()
    print("self-test passed (mixture quantiles at w=0/1, exact mixture CRPS vs brute force)", flush=True)
    res = {"protocol": "revision/protocols/R-X2_linear_pool.md", "development": development()}
    res["posthoc_test"] = {"label": "POST HOC (test)", **posthoc_test()}
    p = save("x2_linear_pool.json", res)
    print("saved", p)
    print("dev seed mean:", {mm: (round(v["overall"], 4), round(v["ramp"], 3)) for mm, v in res["development"]["seed_mean"].items()})
    print("dev decision:", res["development"]["decision"])
    t = res["posthoc_test"]
    print("test:", {mm: (round(v["overall"], 4), round(v["ramp"], 3)) for mm, v in t["crps"].items()})
    print("test G3'' minus:", {mm: tuple(round(x, 4) for x in v["overall"]) for mm, v in t["G3pp_minus"].items()})


if __name__ == "__main__":
    main()
