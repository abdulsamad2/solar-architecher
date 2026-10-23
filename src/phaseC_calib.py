"""Phase C (reports/phaseC_calibration_prereg.md): calibration methods M0-M6 on development out-of-fold streams.

All methods causal with 15-min maturation; hyperparameters chosen leave-one-fold-out (never on the evaluation fold);
initial scores always from other folds. Output: reports/phaseC_results.json (+ experiments/phaseC/intervals_*.npz).
"""
from __future__ import annotations

import bisect
import heapq
import json
from collections import deque
from pathlib import Path

import numpy as np

from src.cal2 import ALPHA, conformal_q, conformity, interval_score, replay, widen
from src.calibrate import base_interval
from src.final_eval import day_bootstrap, nearest_init, rolling_conformal_daily

LEVELS = np.array([.05, .1, .2, .3, .5, .7, .8, .9, .95])
W_GRID = (4620, 2310, 990, 330)          # ties -> less adaptive (larger W first)
G_GRID = (0.001, 0.005, 0.01, 0.02, 0.05)  # ties -> smaller gamma first
W_FIXED = 4620                           # fixed score-set size for ACI / DtACI (implementation detail)
METHODS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")


def dtaci(org, tgt, lo, hi, y, alpha, init_scores):
    """DtACI-style aggregation of ACI experts (prereg implementation note)."""
    fixed = np.sort(np.asarray(init_scores, float))
    gam = np.array(G_GRID)
    K = len(gam)
    a_k = np.full(K, alpha)
    logw = np.zeros(K)
    eta, sigma = 0.1 / alpha, 1 / (2 * K)
    base = conformity(lo, hi, y)
    n = len(y)
    out_lo, out_hi = np.empty(n), np.empty(n)
    order = np.argsort(org, kind="stable")
    pending = []
    for i in order:
        while pending and pending[0][0] <= org[i]:
            _, j = heapq.heappop(pending)
            beta = 1.0 - np.searchsorted(fixed, base[j], side="left") / len(fixed)  # level at which j is just covered
            loss = alpha * (beta - a_k) - np.minimum(0.0, beta - a_k)
            logw -= eta * loss
            err = (beta < a_k).astype(float)
            a_k = np.clip(a_k + gam * (alpha - err), 0.0, 1.0)
        w = np.exp(logw - logw.max())
        w = (1 - sigma) * w / w.sum() + sigma / K
        a_t = float(np.clip((w * a_k).sum(), 0.0, 1.0))
        a, b, _ = widen(lo[i], hi[i], conformal_q(fixed, 1 - a_t))
        out_lo[i], out_hi[i] = a, b
        heapq.heappush(pending, (tgt[i], i))
    return out_lo, out_hi


def mondrian(org, tgt, lo, hi, y, alpha, W, bins, init_by_bin):
    """Rolling conformal with a separate window per forecast-time regime bin (continuous, causal)."""
    base = conformity(lo, hi, y)
    win = {b: deque(np.asarray(init_by_bin[b], float)[-W:]) for b in init_by_bin}
    ordd = {b: sorted(win[b]) for b in win}
    n = len(y)
    out_lo, out_hi = np.empty(n), np.empty(n)
    pending = []
    for i in np.argsort(org, kind="stable"):
        while pending and pending[0][0] <= org[i]:
            _, j = heapq.heappop(pending)
            b = bins[j]
            win[b].append(base[j]); bisect.insort(ordd[b], base[j])
            if len(win[b]) > W:
                old = win[b].popleft(); del ordd[b][bisect.bisect_left(ordd[b], old)]
        a, c, _ = widen(lo[i], hi[i], conformal_q(ordd[bins[i]], 1 - alpha))
        out_lo[i], out_hi[i] = a, c
        heapq.heappush(pending, (tgt[i], i))
    return out_lo, out_hi


class Stream:
    def __init__(self, z, qkey):
        self.org, self.tgt, self.y = z["origin_us"], z["target_us"], z["y"].astype(float)
        self.day, self.fold = z["day"], z["fold"]
        self.q = z[qkey]
        self.spread = self.q[:, 7] - self.q[:, 1]
        self.g = z["g"]
        self.bounds = {nm: base_interval(self.q, LEVELS, nm) for nm in ALPHA}
        self.scores = {nm: conformity(*self.bounds[nm], self.y) for nm in ALPHA}

    def run(self, method, k, nm, param, exclude):
        """Intervals for fold k at level nm; initial scores / bin edges only from folds not in `exclude`."""
        sel = np.flatnonzero(self.fold == k)
        src = np.flatnonzero(~np.isin(self.fold, list(exclude)))
        lo, hi = (b[sel] for b in self.bounds[nm])
        org, tgt, y, a = self.org[sel], self.tgt[sel], self.y[sel], ALPHA[nm]
        s_org, s_sc = self.org[src], self.scores[nm][src]
        if method == "M0":
            return lo, hi
        if method == "M1":
            day = self.day[sel]
            inits = {d: nearest_init(s_org, s_sc, org[day == d].min(), param) for d in np.unique(day)}
            return rolling_conformal_daily(org, tgt, day, lo, hi, y, a, param, inits)
        start = org.min()
        if method == "M2":
            r = replay(org, tgt, lo, hi, y, a, "rolling", param, nearest_init(s_org, s_sc, start, param))
            return r["lo"], r["hi"]
        if method == "M3":
            r = replay(org, tgt, lo, hi, y, a, "aci", param, nearest_init(s_org, s_sc, start, W_FIXED))
            return r["lo"], r["hi"]
        if method == "M4":
            return dtaci(org, tgt, lo, hi, y, a, nearest_init(s_org, s_sc, start, W_FIXED))
        sig = self.spread if method == "M5" else self.g
        edges = np.quantile(sig[src], [1 / 3, 2 / 3])
        bins_all = np.digitize(sig, edges)
        inits = {b: nearest_init(s_org[bins_all[src] == b], s_sc[bins_all[src] == b], start, param) for b in range(3)}
        return mondrian(org, tgt, lo, hi, y, a, param, bins_all[sel], inits)


def grid(method):
    return {"M0": (None,), "M1": W_GRID, "M2": W_GRID, "M3": G_GRID, "M4": (None,), "M5": W_GRID, "M6": W_GRID}[method]


def select(st: Stream, method, k):
    """Leave-one-fold-out: replay each other fold f (init from folds not in {f, k}); sum IS over levels and folds."""
    cand = grid(method)
    if len(cand) == 1:
        return cand[0], {}
    folds = [f for f in range(5) if f != k]
    scores = {}
    for p in cand:
        tot = 0.0
        for f in folds:
            sel = st.fold == f
            for nm in ALPHA:
                lo, hi = st.run(method, f, nm, p, exclude={f, k})
                tot += float(interval_score(lo, hi, st.y[sel], ALPHA[nm]).mean())
        scores[p] = tot
    best = min(scores.values())
    return next(p for p in cand if scores[p] <= best + 1e-12), scores


def evaluate(st: Stream, z):
    n = len(st.y)
    iv = {m: {nm: (np.empty(n), np.empty(n)) for nm in ALPHA} for m in METHODS}
    chosen = {m: {} for m in METHODS}
    for m in METHODS:
        for k in range(5):
            p, _ = select(st, m, k)
            chosen[m][k] = p
            sel = st.fold == k
            for nm in ALPHA:
                lo, hi = st.run(m, k, nm, p, exclude={k})
                iv[m][nm][0][sel], iv[m][nm][1][sel] = lo, hi
        print(m, "chosen", chosen[m], flush=True)
    ramp, sunny = z["ramp"].astype(bool), z["sunny"].astype(bool)
    up = ramp & (st.y > st.q[:, 4])  # descriptive split only
    strata = {"overall": np.ones(n, bool), "sunny": sunny, "cloudy": ~sunny, "ramp": ramp, "non-ramp": ~ramp,
              **{f"fold {k}": st.fold == k for k in range(5)}}
    res = {"chosen": {m: {str(k): v for k, v in c.items()} for m, c in chosen.items()}, "metrics": {}, "bootstrap_vs_M1": {}}
    for m in METHODS:
        res["metrics"][m] = {}
        for nm, a in ALPHA.items():
            lo, hi = iv[m][nm]
            cov = (st.y >= lo) & (st.y <= hi)
            isc = interval_score(lo, hi, st.y, a)
            res["metrics"][m][nm] = {g: {"coverage": float(cov[v].mean()), "width": float((hi - lo)[v].mean()),
                                         "interval_score": float(isc[v].mean()), "miss_below": float((st.y < lo)[v].mean()),
                                         "miss_above": float((st.y > hi)[v].mean())} for g, v in strata.items()}
    lo1, hi1 = iv["M1"]["90"]
    cov1 = ((st.y >= lo1) & (st.y <= hi1)).astype(float)
    is1 = interval_score(lo1, hi1, st.y, 0.1)
    for m in METHODS:
        if m == "M1":
            continue
        lo, hi = iv[m]["90"]
        cov = ((st.y >= lo) & (st.y <= hi)).astype(float)
        isc = interval_score(lo, hi, st.y, 0.1)
        res["bootstrap_vs_M1"][m] = {
            "ramp_cov90_diff": day_bootstrap(cov[ramp] - cov1[ramp], st.day[ramp], sunny[ramp]),
            "overall_IS90_diff": day_bootstrap(isc - is1, st.day, sunny),
            "overall_IS90_rel": float(isc.mean() / is1.mean() - 1)}
    return res, iv


def main():
    z = np.load("experiments/phaseC/oof_streams.npz")
    out = {}
    for name, key in (("G3''-F", "q_G3ppF"), ("E4-B-fit", "q_E4")):
        print("==", name, flush=True)
        st = Stream(z, key)
        res, iv = evaluate(st, z)
        b56 = {m: res["bootstrap_vs_M1"][m] for m in ("M5", "M6")}
        # C1: pick between M5 / M6 by the LOFO selection criterion only (mean overall IS summed over levels)
        crit = {m: sum(res["metrics"][m][nm]["overall"]["interval_score"] for nm in ALPHA) for m in ("M5", "M6")}
        pick = min(crit, key=crit.get)
        d = b56[pick]["ramp_cov90_diff"]
        res["C1"] = {"picked": pick, "criterion": crit, "ramp_cov90_diff": d, "IS90_rel": b56[pick]["overall_IS90_rel"],
                     "confirmed": bool(d[0] >= 0.05 and d[1] > 0 and b56[pick]["overall_IS90_rel"] <= 0.02)}
        out[name] = res
        np.savez_compressed(f"experiments/phaseC/intervals_{key}.npz",
                            **{f"{m}|{nm}|{s}": iv[m][nm][j] for m in METHODS for nm in ALPHA for j, s in enumerate(("lo", "hi"))})
        print(json.dumps(res["C1"], indent=1), flush=True)
    Path("reports/phaseC_results.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
