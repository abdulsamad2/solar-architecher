"""Revision R-R2: day-level exact sign-flip tests and t intervals for the key contrasts on the official test set
(protocol revision/protocols/R-R2_day_level_inference.md). Post hoc, frozen predictions, CPU only.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from src.metrics import crps_quantile_interp
from src.ramp_scoring import ramp_weighted_crps
from src.revision_common import LEVELS, day_bootstrap, frozen_test_predictions, save, test_context
from src.revision_z_common import ramp_region

PP, DIRECT, CALM, G4 = "G3''-F", "E4-B-fit", "E6-S", "G4"
MARGIN = 0.02


def sign_flip(S, n):
    """Exact two-sided and one-sided (statistic < 0) p-values of the pooled mean sum(S)/sum(n) over all 2^D day sign patterns."""
    D = len(S)
    bits = ((np.arange(2 ** D, dtype=np.int64)[:, None] >> np.arange(D)) & 1).astype(np.int8) * 2 - 1
    t = (bits @ S) / n.sum()
    obs = S.sum() / n.sum()
    eps = 1e-12 * max(1.0, abs(obs))
    return float(np.mean(np.abs(t) >= abs(obs) - eps)), float(np.mean(t <= obs + eps))


def t_interval(dm, sunny):
    """Equal-day-weight mean of the daily means, stratified by sunny / cloudy days, with a t interval."""
    est = var = 0.0
    H = 0
    for g in (True, False):
        x = dm[sunny == g]
        if len(x) == 0:
            continue
        H += 1
        w = len(x) / len(dm)
        est += w * x.mean()
        var += w * w * (x.var(ddof=1) / len(x) if len(x) > 1 else 0.0)
    q = stats.t.ppf(0.975, len(dm) - H)
    return [float(est), float(est - q * np.sqrt(var)), float(est + q * np.sqrt(var))], int(len(dm) - H)


def analyse(diff, mask, day, sunny, margin=None):
    d, dy, sn = diff[mask], day[mask], sunny[mask]
    days, inv = np.unique(dy, return_inverse=True)
    S, n = np.bincount(inv, weights=d), np.bincount(inv).astype(float)
    sday = np.zeros(len(days), bool)
    sday[inv] = sn
    p2, _ = sign_flip(S, n)
    ti, df = t_interval(S / n, sday)
    out = {"n": int(mask.sum()), "days": int(len(days)), "pooled_mean": float(S.sum() / n.sum()), "bootstrap_percentile": day_bootstrap(d, dy, sn),
           "sign_flip_p_two_sided": p2, "t_interval_day_means": ti, "t_df": df, "days_proposed_better": int((S < 0).sum()),
           "largest_single_day_share": float(np.abs(S).max() / np.abs(S).sum())}
    if margin is not None:  # non-inferiority: H0 difference >= margin
        _, p1 = sign_flip(S - margin * n, n)
        out |= {"margin": margin, "sign_flip_p_noninferiority": p1, "t_upper_below_margin": bool(ti[2] <= margin)}
    b = out["bootstrap_percentile"]
    boot_excl, t_excl = not (b[1] <= 0 <= b[2]), not (ti[1] <= 0 <= ti[2])
    out["bootstrap_excludes_zero"], out["sign_flip_and_t_exclude_zero"] = bool(boot_excl), bool(p2 < 0.05 and t_excl)
    out["robust"] = bool(boot_excl == (p2 < 0.05) == t_excl)
    return out


def main():
    c = test_context()
    q = frozen_test_predictions(c["sample_ids"])
    y, day, sunny, ramp = c["y"], c["day"], c["sunny"], c["ramp"]
    _, lo, hi = ramp_region(c)
    crps = {m: crps_quantile_interp(q[m], y, LEVELS) for m in (PP, DIRECT, CALM, G4)}
    tw = {m: ramp_weighted_crps(q[m], LEVELS, y, lo, hi)[1] for m in (PP, DIRECT, CALM, G4)}
    al = np.ones(len(y), bool)
    spec = [("1 overall CRPS vs direct expert (H1)", crps, DIRECT, al, None), ("2 ramp CRPS vs direct expert (H3, margin 0.02)", crps, DIRECT, ramp, MARGIN),
            ("3 overall CRPS vs calm expert", crps, CALM, al, None), ("4 overall CRPS vs G4", crps, G4, al, None),
            ("5 non-ramp CRPS vs G4", crps, G4, ~ramp, None), ("6 ramp CRPS vs G4", crps, G4, ramp, None),
            ("7 ramp-region twCRPS vs direct expert", tw, DIRECT, al, None), ("8 ramp-region twCRPS vs calm expert", tw, CALM, al, None),
            ("9 ramp-region twCRPS vs G4", tw, G4, al, None)]
    res = {"protocol": "revision/protocols/R-R2_day_level_inference.md", "contrasts": {}}
    for name, sc, comp, mask, margin in spec:
        res["contrasts"][name] = r = analyse(sc[PP] - sc[comp], mask, day, sunny, margin)
        f = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"
        extra = f"  non-inferiority p {r['sign_flip_p_noninferiority']:.4f}" if margin is not None else ""
        print(f"{name:48s} boot {f(r['bootstrap_percentile'])} | t({r['t_df']}) {f(r['t_interval_day_means'])} | sign-flip p {r['sign_flip_p_two_sided']:.5f}{extra}"
              f" | better on {r['days_proposed_better']}/{r['days']} days, largest day {r['largest_single_day_share']:.0%} => {'robust' if r['robust'] else 'NOT ROBUST'}", flush=True)
    p1 = res["contrasts"][spec[0][0]]["sign_flip_p_two_sided"]
    p2 = res["contrasts"][spec[1][0]]["sign_flip_p_noninferiority"]
    lo_p, hi_p = sorted([p1, p2])
    res["holm"] = {"H1 (two-sided)": float(min(1, 2 * p1) if p1 == lo_p else min(1, max(2 * lo_p, p1))),
                   "H3 (non-inferiority)": float(min(1, 2 * p2) if p2 == lo_p and p1 != lo_p else min(1, max(2 * lo_p, p2)))}
    print("Holm-adjusted p:", res["holm"])
    print("saved", save("r2_day_level_inference.json", res))


if __name__ == "__main__":
    main()
