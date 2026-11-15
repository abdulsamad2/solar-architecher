"""Revision R-M1: development of the combination rule of the method (v2) on development out-of-fold forecasts
(protocol revision/protocols/R-M1_method_v2_development.md). Frozen expert predictions; CPU only, no training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.e8_features import FEAT, verify_frozen
from src.gate7b import LAMBDAS, apply_gate, grid_quantiles, raw_features, standardize, transform
from src.metrics import crps_quantile_interp
from src.ramp_scoring import ramp_weighted_crps
from src.revision_common import LEVELS, day_bootstrap, save
from src.revision_r1_gate_proper_score import band
from src.revision_x2_linear_pool import _cdf, _mean_abs_cross, _mean_abs_self, _mean_abs_y, g3pp_params, load

ROOT = {"D": Path("experiments/E4_calib/v1/csi"), "R": Path("experiments/E9/v1/A2/csi"), "C": Path("experiments/E6/v1/E6-S/csi")}
DESIGNS = {"M-DC": ("D", "C"), "M-RC": ("R", "C"), "M-DRC": ("D", "R", "C")}  # the calm expert is the last (reference) component
TAU = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
PARTS = ("crps", "tw_ramp", "tw_calm")


class MixN:
    """Exact CRPS of a mixture of C 200-atom distributions: sum_i w_i a_i - 0.5 sum_ij w_i w_j c_ij."""

    def __init__(self, grids, y):
        x = [torch.as_tensor(np.sort(g, 1), dtype=torch.float64) for g in grids]
        yt = torch.as_tensor(y, dtype=torch.float64)
        C = len(x)
        self.a = torch.stack([_mean_abs_y(v, yt) for v in x], 1)  # [N, C]
        self.c = torch.zeros(len(yt), C, C, dtype=torch.float64)
        for i in range(C):
            self.c[:, i, i] = _mean_abs_self(x[i])
            for j in range(i + 1, C):
                self.c[:, i, j] = self.c[:, j, i] = _mean_abs_cross(x[i], x[j])

    def per_sample(self, w):
        return (w * self.a).sum(1) - 0.5 * torch.einsum("ni,nij,nj->n", w, self.c, w)

    def subset(self, idx):
        o = object.__new__(MixN)
        o.a, o.c = self.a[idx], self.c[idx]
        return o


def weights(z, b, B):
    """Softmax gate; the last component (calm expert) has logit 0."""
    logit = b + (z @ B if B is not None else 0.0)
    logit = torch.cat([logit, torch.zeros(len(logit), 1, dtype=torch.float64)], 1)
    return torch.softmax(logit, 1)


def fit(z, f, lam, n_comp, gated=True):
    zt = torch.as_tensor(z, dtype=torch.float64)
    b = torch.zeros(1, n_comp - 1, dtype=torch.float64, requires_grad=True)
    B = torch.zeros(z.shape[1], n_comp - 1, dtype=torch.float64, requires_grad=True) if gated else None
    params = [b] + ([B] if gated else [])
    opt = torch.optim.LBFGS(params, max_iter=300, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        w = weights(zt, b.expand(len(zt), -1), B)
        loss = f.per_sample(w).mean() + (lam * (B ** 2).sum() if gated else 0.0)
        loss.backward()
        return loss

    opt.step(closure)
    return b.detach(), (B.detach() if gated else None)


def select_and_fit(z, f, order, n_comp):
    half = len(order) // 2
    a, bb = order[:half], order[half:]
    scores = {}
    for lam in LAMBDAS:
        (za, zb), _ = standardize(z[a], z[bb])
        b, B = fit(za, f.subset(a), lam, n_comp)
        scores[lam] = float(f.subset(bb).per_sample(weights(torch.as_tensor(zb), b.expand(len(zb), -1), B)).mean())
    lam = max(l for l in LAMBDAS if scores[l] <= min(scores.values()) + 1e-12)
    (zall,), (mu, sd) = standardize(z)
    b, B = fit(zall, f, lam, n_comp)
    return {"lambda": lam, "b": b, "B": B, "mu": mu, "sd": sd}


def mixture_quantiles(qs, w):
    """Nine quantiles of the mixture of piecewise-linear CDFs (constant tails), components qs [C][N, 9], weights w [N, C]."""
    knots = np.sort(np.concatenate(qs, 1), 1)
    fl = sum(w[:, i:i + 1] * _cdf(knots, q, False) for i, q in enumerate(qs))
    fr = sum(w[:, i:i + 1] * _cdf(knots, q, True) for i, q in enumerate(qs))
    xs = np.repeat(knots, 2, 1)
    fs = np.empty_like(xs)
    fs[:, 0::2], fs[:, 1::2] = fl, fr
    fs = np.maximum.accumulate(fs, 1)
    out = np.empty((len(w), len(LEVELS)))
    rows = np.arange(len(w))
    for i, p in enumerate(LEVELS):
        k = np.clip((fs < p - 1e-12).sum(1), 0, xs.shape[1] - 1)
        km = np.maximum(k - 1, 0)
        f0, f1, x0, x1 = fs[rows, km], fs[rows, k], xs[rows, km], xs[rows, k]
        t = np.where((k > 0) & (f1 > f0), (p - f0) / np.maximum(f1 - f0, 1e-15), 1.0)
        out[:, i] = x0 + np.clip(t, 0, 1) * (x1 - x0)
    return out


def features(p, tau, design):
    """Standardisable gate features: logit g (calm expert) and log disagreement of every non-calm expert with the calm expert."""
    raws = [raw_features(p[e], p["C"], tau) for e in design[:-1]]
    return np.concatenate([transform(raws[0], ("g",))] + [transform(r, ("d",)) for r in raws], 1)


def main():
    verify_frozen()
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = win["sample_ids"][win["eligible_common"]]
    stored = json.loads(Path("revision/results/e2_gate_ablation_dev.json").read_text())["per_seed"]
    methods = ["Q-RC"] + [f"{d}{s}" for d in DESIGNS for s in ("", " const")]
    res = {"protocol": "revision/protocols/R-M1_method_v2_development.md", "per_seed": {}, "mean_weights": {}}
    sc_all = {}
    for s in (42, 123, 456):
        qs = {m: [] for m in methods}
        ys, los, his, days, sun = [], [], [], [], []
        wmean = {d: [] for d in DESIGNS}
        for k in range(5):
            tau = TAU[str(k)]["training_labels_fit_days"]
            cal = {e: load(r, s, k, "cal", common) for e, r in ROOT.items()}
            val = {e: load(r, s, k, "val", common) for e, r in ROOT.items()}
            for e in ("D", "R"):
                assert np.array_equal(cal[e]["sample_ids"], cal["C"]["sample_ids"]) and np.array_equal(val[e]["sample_ids"], val["C"]["sample_ids"])
            order = np.argsort(cal["C"]["origin_time_us"], kind="stable")
            w_qb = apply_gate(g3pp_params(s, k), raw_features(val["R"], val["C"], tau))
            qs["Q-RC"].append(w_qb[:, None] * val["R"]["pred_q"] + (1 - w_qb[:, None]) * val["C"]["pred_q"])
            for d, comps in DESIGNS.items():
                f = MixN([grid_quantiles(cal[e]["pred_q"], LEVELS) for e in comps], cal["C"]["target"])
                zc, zv = features(cal, tau, comps), features(val, tau, comps)
                p = select_and_fit(zc, f, order, len(comps))
                wv = weights(torch.as_tensor((zv - p["mu"]) / p["sd"]), p["b"].expand(len(zv), -1), p["B"]).numpy()
                qs[d].append(mixture_quantiles([val[e]["pred_q"] for e in comps], wv))
                b0, _ = fit(zc, f, 0.0, len(comps), gated=False)
                wc = weights(torch.zeros(len(zv), 0, dtype=torch.float64), b0.expand(len(zv), -1), None).numpy()
                qs[f"{d} const"].append(mixture_quantiles([val[e]["pred_q"] for e in comps], wc))
                wmean[d].append(wv.mean(0))
            lo, hi = band(val["C"], tau)
            ys.append(val["C"]["target"]); los.append(lo); his.append(hi); days.append(val["C"]["day"]); sun.append(val["C"]["clear_day"])
        cat = np.concatenate
        y, lo, hi, day, sunny = cat(ys), cat(los), cat(his), cat(days), cat(sun).astype(bool)
        sc = {}
        for m in methods:
            q = cat(qs[m])
            sc[m] = dict(zip(PARTS, ramp_weighted_crps(q, LEVELS, y, lo, hi)))
            sc[m]["cov90"], sc[m]["w90"] = (y >= q[:, 0]) & (y <= q[:, -1]), q[:, -1] - q[:, 0]
            assert np.abs(sc[m]["crps"] - crps_quantile_interp(q, y, LEVELS)).max() < 1e-9
        assert abs(sc["Q-RC"]["crps"].mean() - stored[str(s)]["crps"]["G3''"]["overall"]) < 1e-9, "STOP: frozen quantile blend not reproduced"
        sc_all[s] = (sc, day, sunny)
        res["per_seed"][str(s)] = {m: {"crps": float(v["crps"].mean()), "tw_ramp": float(v["tw_ramp"].mean()), "tw_calm": float(v["tw_calm"].mean()),
                                       "sunny": float(v["crps"][sunny].mean()), "cloudy": float(v["crps"][~sunny].mean()),
                                       "raw90_coverage": float(v["cov90"].mean()), "raw90_width_kw": float(v["w90"].mean())} for m, v in sc.items()}
        res["mean_weights"][str(s)] = {d: np.mean(v, 0).tolist() for d, v in wmean.items()}
        print(f"seed {s}: " + "  ".join(f"{m} {res['per_seed'][str(s)][m]['crps']:.4f}" for m in methods), flush=True)

    def contrast(a, b):
        out = {}
        for s, (sc, day, sunny) in sc_all.items():
            out[str(s)] = {p: day_bootstrap(sc[a][p] - sc[b][p], day, sunny) for p in PARTS}
        return out

    pairs = [("M-RC", "M-DC"), ("M-DRC", "M-DC"), ("M-DRC", "M-RC")] + [(d, f"{d} const") for d in DESIGNS] + [(d, "Q-RC") for d in DESIGNS]
    res["contrasts"] = {f"{a} - {b}": contrast(a, b) for a, b in pairs}
    mean = lambda m: float(np.mean([res["per_seed"][str(s)][m]["crps"] for s in (42, 123, 456)]))
    res["seed_mean_crps"] = {m: mean(m) for m in methods}

    def qualifies(a, b):  # rule 2: a (more complex) replaces b
        c = res["contrasts"][f"{a} - {b}"]
        better = all(c[s]["crps"][2] < 0 for s in c)
        size = (mean(b) - mean(a)) / mean(b) >= 0.01
        calm_ok = not any(c[s]["tw_calm"][1] > 0 for s in c)
        return bool(better and size and calm_ok), {"3/3 seeds CI<0": bool(better), "seed-mean gain >= 1%": bool(size), "calm region not worse": bool(calm_ok),
                                                  "relative_gain": (mean(b) - mean(a)) / mean(b)}

    q_rc, d_rc = qualifies("M-RC", "M-DC")
    q_drc, d_drc = qualifies("M-DRC", "M-DC")
    q_drc_rc, d_drc_rc = qualifies("M-DRC", "M-RC")
    chosen = "M-DC"
    if q_rc:
        chosen = "M-RC"
    if q_drc and (not q_rc or q_drc_rc):
        chosen = "M-DRC"
    gc = res["contrasts"][f"{chosen} - {chosen} const"]
    gated = all(gc[s]["crps"][2] < 0 for s in gc)
    res["selection"] = {"M-RC replaces M-DC": d_rc, "M-DRC replaces M-DC": d_drc, "M-DRC replaces M-RC": d_drc_rc, "design": chosen,
                        "gate kept (beats constant weights in 3/3 seeds)": bool(gated), "method": chosen if gated else f"{chosen} const"}
    f = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"
    for name, c in res["contrasts"].items():
        print(f"{name:22s} " + " | ".join(f"seed {s}: {f(c[s]['crps'])} ramp {c[s]['tw_ramp'][0]:+.4f} calm {c[s]['tw_calm'][0]:+.4f}" for s in c), flush=True)
    print("seed-mean CRPS:", {m: round(v, 4) for m, v in res["seed_mean_crps"].items()})
    print("selection:", json.dumps(res["selection"], indent=1))
    print("saved", save("m1_method_v2_development.json", res))


if __name__ == "__main__":
    main()
