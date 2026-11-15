"""Revision R-Z3: outcome-class (ramp-protected) conformal prediction sets and their cost
(protocol revision/protocols/R-Z3_outcome_class_conformal.md). Development initial scores, post hoc test. CPU only.
"""
from __future__ import annotations

import bisect
import heapq
from collections import deque

import numpy as np

from src.cal2 import conformal_q, conformity, widen
from src.calibrate import base_interval
from src.revision_c07 import past_only_init
from src.revision_common import LEVELS, day_bootstrap, frozen_test_predictions, save, test_context
from src.revision_y2_conditional_conformal import PP, development
from src.revision_z_common import calibrated_intervals, ramp_region

W_CLASS = {True: 660, False: 3960}  # ramp / non-ramp windows (frozen W = 4620 split by the ramp share)
LEVEL = 0.90


def class_quantiles(org, tgt, day, score, cls, init_fn):
    """Per forecast, the rolling conformal quantile of each outcome class from matured same-class scores (daily reset)."""
    qr, qn = np.empty(len(score)), np.empty(len(score))
    for d in np.unique(day):
        sel = np.flatnonzero(day == d)
        sel = sel[np.argsort(org[sel], kind="stable")]
        win = {k: deque(np.asarray(init_fn(k, org[sel].min()), float)[-W_CLASS[k]:]) for k in (True, False)}
        srt = {k: sorted(win[k]) for k in win}
        pending = []
        for i in sel:
            while pending and pending[0][0] <= org[i]:
                _, j = heapq.heappop(pending)
                k = bool(cls[j])
                win[k].append(score[j]); bisect.insort(srt[k], score[j])
                if len(win[k]) > W_CLASS[k]:
                    old = win[k].popleft(); del srt[k][bisect.bisect_left(srt[k], old)]
            qr[i], qn[i] = conformal_q(srt[True], LEVEL), conformal_q(srt[False], LEVEL)
            heapq.heappush(pending, (tgt[i], i))
    return qr, qn


def main():
    dev = development()
    c = test_context()
    q = frozen_test_predictions(c["sample_ids"])[PP]
    y, day, org, tgt, ramp = c["y"], c["day"], c["org"], c["tgt"], c["ramp"]
    _, L, U = ramp_region(c)
    ulo, uhi = calibrated_intervals(c, q, PP)["90"]
    lo, hi = base_interval(q, LEVELS, "90")
    dsc, dcls = dev["scores"]["90"], dev["ramp"].astype(bool)
    init_fn = lambda k, start: past_only_init(dev["org"][dcls == k], dsc[dcls == k], start, W_CLASS[k])[0]
    qr, qn = class_quantiles(org, tgt, day, conformity(lo, hi, y), ramp, init_fn)
    ar, br = map(np.array, zip(*[widen(a, b, v)[:2] for a, b, v in zip(lo, hi, qr)]))
    an, bn = map(np.array, zip(*[widen(a, b, v)[:2] for a, b, v in zip(lo, hi, qn)]))
    # pieces of the set: ramp-class interval below L and above U, non-ramp-class interval inside (L, U); all >= 0 kW
    pieces = [(np.maximum(ar, 0), np.minimum(br, L)), (np.maximum(np.maximum(an, L), 0), np.minimum(bn, U)), (np.maximum(np.maximum(ar, U), 0), br)]
    length = [np.maximum(b - a, 0) for a, b in pieces]
    ok = [b >= a for a, b in pieces]
    size = sum(length)
    first = np.min([np.where(o, a, np.inf) for (a, _), o in zip(pieces, ok)], 0)
    last = np.max([np.where(o, b, -np.inf) for (_, b), o in zip(pieces, ok)], 0)
    empty = ~np.any(ok, 0)
    hull = np.where(empty, 0.0, last - first)
    covered = np.where(ramp, (y >= ar) & (y <= br), (y >= an) & (y <= bn))
    ucov, uw = (y >= ulo) & (y <= uhi), uhi - ulo
    m = c["masks"]
    groups = ("overall", "ramp", "rising ramp", "falling ramp", "non-ramp", "sunny", "cloudy")
    tab = lambda cov, w: {g: {"coverage": float(cov[m[g]].mean()), "mean_size_kw": float(w[m[g]].mean()), "n": int(m[g].sum())} for g in groups}
    res = {"protocol": "revision/protocols/R-Z3_outcome_class_conformal.md", "W_class": {"ramp": W_CLASS[True], "non-ramp": W_CLASS[False]},
           "dev_excluded": dev["n_excluded"], "dev_ramp_share": float(dcls.mean()),
           "U": tab(ucov, uw), "class_conditional": tab(covered, size),
           "hull_kw": {g: float(hull[m[g]].mean()) for g in groups},
           "share_not_an_interval": float(np.mean(~empty & (hull - size > 1e-9))), "share_empty": float(empty.mean()),
           "mean_class_quantile_kw": {"ramp": float(qr.mean()), "non-ramp": float(qn.mean())},
           "size_minus_U": {g: day_bootstrap((size - uw)[m[g]], day[m[g]], c["sunny"][m[g]]) for g in ("overall", "ramp", "non-ramp")}}
    cc = res["class_conditional"]
    res["size_ratio_to_U"] = cc["overall"]["mean_size_kw"] / res["U"]["overall"]["mean_size_kw"]
    res["decision"] = "ramp protection attained" if cc["ramp"]["coverage"] >= 0.87 and cc["non-ramp"]["coverage"] >= 0.87 else "not attained"
    for g in groups:
        print(f"{g:13s} U cov {res['U'][g]['coverage']:.3f} width {res['U'][g]['mean_size_kw']:.2f} | class-conditional cov {cc[g]['coverage']:.3f} "
              f"size {cc[g]['mean_size_kw']:.2f} hull {res['hull_kw'][g]:.2f}")
    print(f"size ratio {res['size_ratio_to_U']:.2f}; size - U overall {np.round(res['size_minus_U']['overall'], 3).tolist()}; not an interval "
          f"{res['share_not_an_interval']:.3f}; empty {res['share_empty']:.4f}; mean q ramp {qr.mean():.2f} non-ramp {qn.mean():.2f} => {res['decision']}")
    print("saved", save("z3_outcome_class_conformal.json", res))


if __name__ == "__main__":
    main()
