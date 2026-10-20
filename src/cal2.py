"""E4-CAL2: online adaptive calibration (ACI, causal rolling-window conformal) vs fixed CQR.

Protocol: reports/e4_cal2_prereg.md. Every update is causal: a forecast's outcome becomes usable only at
its target time, and each update asserts observation_time <= current issue time.
"""
from __future__ import annotations

import bisect
import heapq
from collections import deque

import numpy as np

ALPHA = {"50": 0.5, "80": 0.2, "90": 0.1}
ACI_GAMMAS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)
ROLL_WINDOWS = (330, 990, 2310, 4620)


def conformity(lo, hi, y):
    return np.maximum(np.asarray(lo) - y, np.asarray(y) - hi)


def conformal_q(sorted_scores, level: float) -> float:
    """Rank-ceil((n+1)*level) element of ``sorted_scores``; level clipped to [0, 1], rank to [1, n]."""
    n = len(sorted_scores)
    k = int(np.ceil((n + 1) * min(max(level, 0.0), 1.0)))
    return float(sorted_scores[min(max(k, 1), n) - 1])


def widen(lo: float, hi: float, q: float):
    """Interval [lo - q, hi + q] with the pre-registered width floor (degenerate point instead of inversion)."""
    if q < -(hi - lo) / 2:
        mid = (lo + hi) / 2
        return mid, mid, True
    return lo - q, hi + q, False


def interval_score(lo, hi, y, alpha: float) -> np.ndarray:
    lo, hi, y = map(np.asarray, (lo, hi, y))
    return (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)


def replay(origin_us, target_us, lo, hi, y, alpha: float, method: str, param, init_scores) -> dict:
    """Causal chronological replay of one stream for one interval level.

    method: 'aci' (param = gamma; fixed score set = init_scores) or 'rolling' (param = W; window starts with
    init_scores, which must already be the W chosen initial scores).
    Returns per-sample issued intervals in the input order plus the update trace.
    """
    n = len(y)
    order = np.argsort(origin_us, kind="stable")
    base_scores = conformity(lo, hi, y)
    out_lo, out_hi = np.empty(n), np.empty(n)
    q_issued, alpha_issued = np.empty(n), np.full(n, np.nan)
    n_updates = np.zeros(n, np.int64)
    last_obs = np.full(n, np.iinfo(np.int64).min, np.int64)
    degenerate = np.zeros(n, bool)
    miss = np.zeros(n, bool)

    if method == "aci":
        fixed = np.sort(np.asarray(init_scores, float))
        alpha_t = alpha
    elif method == "rolling":
        window = deque(np.asarray(init_scores, float)[-int(param):])
        ordered = sorted(window)
    else:
        raise ValueError(method)

    pending: list = []
    done, latest = 0, np.iinfo(np.int64).min
    for i in order:
        t = origin_us[i]
        while pending and pending[0][0] <= t:
            tt, j = heapq.heappop(pending)
            assert tt <= t, "STOP: update uses an outcome not yet observable"
            if method == "aci":
                # amendment 1: clamp to [0, 1] (anti-windup; the level is clipped there anyway)
                alpha_t = min(max(alpha_t + param * (alpha - float(miss[j])), 0.0), 1.0)
            else:
                window.append(base_scores[j])
                bisect.insort(ordered, base_scores[j])
                if len(window) > int(param):
                    old = window.popleft()
                    del ordered[bisect.bisect_left(ordered, old)]
            done += 1
            latest = max(latest, tt)
        if method == "aci":
            q = conformal_q(fixed, 1 - alpha_t)
            alpha_issued[i] = alpha_t
        else:
            q = conformal_q(ordered, 1 - alpha)
        a, b, deg = widen(lo[i], hi[i], q)
        if not (np.isfinite(a) and np.isfinite(b)) or a > b:
            raise RuntimeError("STOP: non-finite or inverted interval")
        out_lo[i], out_hi[i], degenerate[i], q_issued[i] = a, b, deg, q
        miss[i] = not (a <= y[i] <= b)
        n_updates[i], last_obs[i] = done, latest
        assert latest <= t
        heapq.heappush(pending, (target_us[i], i))
    return {"lo": out_lo, "hi": out_hi, "q": q_issued, "alpha_t": alpha_issued, "n_updates": n_updates,
            "last_obs_us": last_obs, "degenerate": degenerate, "miss": miss}


def select_param(method: str, cal_origin, cal_target, cal_bounds: dict, cal_y) -> tuple:
    """Pick the hyperparameter on CALIBRATION data only: first half initialises, second half is replayed.

    cal_bounds: {level_name: (lo, hi)} on calibration samples. Criterion: sum over levels of mean interval score.
    Ties -> less adaptive (smaller gamma / larger W).
    """
    order = np.argsort(cal_origin, kind="stable")
    half = len(order) // 2
    a, b = order[:half], order[half:]
    grid = ACI_GAMMAS if method == "aci" else tuple(sorted(ROLL_WINDOWS, reverse=True))
    scores = {}
    for p in grid:
        total = 0.0
        for name, (lo, hi) in cal_bounds.items():
            init = conformity(lo[a], hi[a], cal_y[a])
            if method == "rolling":
                init = init[np.argsort(cal_origin[a], kind="stable")]
            r = replay(cal_origin[b], cal_target[b], lo[b], hi[b], cal_y[b], ALPHA[name], method, p, init)
            total += float(interval_score(r["lo"], r["hi"], cal_y[b], ALPHA[name]).mean())
        scores[p] = total
    best = min(scores.values())
    chosen = next(p for p in grid if scores[p] <= best + 1e-12)
    return chosen, scores
