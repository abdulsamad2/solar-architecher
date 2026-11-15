"""Revision R-Z1: calibrated-forecast null for ramp coverage - how much of the sub-nominal coverage on ramp outcomes is
the selection effect of conditioning on the outcome (protocol revision/protocols/R-Z1_ramp_coverage_null.md).
Post hoc test, frozen predictions, CPU only.
"""
from __future__ import annotations

import numpy as np

from src.cal2 import ALPHA
from src.revision_c07 import MODELS
from src.revision_common import LEVELS, frozen_test_predictions, save, test_context
from src.revision_z_common import calibrated_intervals, ramp_region

P = np.array([.05, .10, .25, .50, .75, .90, .95])
M, CHUNK, SEED = 2000, 100, 20261115
TAILS = ("T-exp", "T-lin")
EVENTS = ("ramp", "rising ramp", "falling ramp", "non-ramp", "overall")
PBINS = (0.0, 0.05, 0.2, 0.5, 1.0000001)
MED = list(LEVELS).index(0.5)


def nodes(ci, med):
    """Calibrated quantile nodes at P (rearranged to be non-decreasing)."""
    v = np.stack([ci["90"][0], ci["80"][0], ci["50"][0], med, ci["50"][1], ci["80"][1], ci["90"][1]], 1)
    return np.sort(v, 1)


def draw(v, u, cap, tail):
    j = np.clip(np.searchsorted(P, u, side="right") - 1, 0, len(P) - 2)
    w = (np.clip(u, P[0], P[-1]) - P[j]) / (P[j + 1] - P[j])
    y = np.take_along_axis(v, j, 1) * (1 - w) + np.take_along_axis(v, j + 1, 1) * w
    s_lo, s_hi = np.maximum(v[:, 1] - v[:, 0], 1e-6)[:, None], np.maximum(v[:, 6] - v[:, 5], 1e-6)[:, None]
    t_lo, t_hi = np.clip(u / P[0], 1e-300, 1.0), np.clip((1 - u) / (1 - P[-1]), 1e-300, 1.0)  # in (0, 1] inside the tails
    e_lo, e_hi = (-np.log(t_lo), -np.log(t_hi)) if tail == "T-exp" else (1 - t_lo, 1 - t_hi)
    y = np.where(u < P[0], v[:, :1] - s_lo * e_lo, y)
    y = np.where(u > P[-1], v[:, 6:] + s_hi * e_hi, y)
    return np.clip(y, 0.0, cap[:, None])


def ratio_boot(num_o, den_o, num_n, den_n, day, sunny_day, n_boot=10_000, seed=20261022):
    """Day-block bootstrap of observed coverage, null coverage and their gap (ratios recomputed per resample)."""
    days, inv = np.unique(day, return_inverse=True)
    s = [np.bincount(inv, weights=x.astype(float)) for x in (num_o, den_o, num_n, den_n)]
    sunny = np.zeros(len(days), bool)
    sunny[inv] = sunny_day
    groups = [np.flatnonzero(sunny), np.flatnonzero(~sunny)]
    rng = np.random.default_rng(seed)
    b = np.empty((n_boot, 3))
    for i in range(n_boot):
        ii = np.concatenate([rng.choice(g, len(g)) for g in groups if len(g)])
        o, n = s[0][ii].sum() / s[1][ii].sum(), s[2][ii].sum() / s[3][ii].sum()
        b[i] = o, n, o - n
    ci = lambda k: [float(np.quantile(b[:, k], 0.025)), float(np.quantile(b[:, k], 0.975))]
    return {"c_obs_ci": ci(0), "c_null_ci_days": ci(1), "gap": float(s[0].sum() / s[1].sum() - s[2].sum() / s[3].sum()), "gap_ci": ci(2)}


def main():
    c = test_context()
    q = frozen_test_predictions(c["sample_ids"])
    y, day, sunny, kn, tau = c["y"], c["day"], c["sunny"], c["k_now"], c["tau"]
    pc, _, _ = ramp_region(c)
    cap = 1.5 * pc
    obs_ev = {"ramp": c["ramp"], "rising ramp": c["masks"]["rising ramp"], "falling ramp": c["masks"]["falling ramp"],
              "non-ramp": ~c["ramp"], "overall": np.ones(len(y), bool)}
    res = {"protocol": "revision/protocols/R-Z1_ramp_coverage_null.md", "M": M, "seed": SEED, "n": int(len(y)),
           "observed_ramp_rate": float(c["ramp"].mean()), "models": {}}
    for m in MODELS:
        ci = calibrated_intervals(c, q[m], m)
        v = nodes(ci, q[m][:, MED])
        cov_o = {nm: (y >= ci[nm][0]) & (y <= ci[nm][1]) for nm in ALPHA}
        out = {}
        for tail in TAILS:
            rng = np.random.default_rng(SEED)  # common random numbers across tail models and models
            p_ev = {e: np.zeros(len(y)) for e in EVENTS}
            p_ec = {nm: {e: np.zeros(len(y)) for e in EVENTS} for nm in ALPHA}
            rep_n = {e: [] for e in EVENTS}
            rep_c = {nm: {e: [] for e in EVENTS} for nm in ALPHA}
            for _ in range(M // CHUNK):
                ys = draw(v, rng.random((len(y), CHUNK)), cap, tail)
                dk = ys / pc[:, None] - kn[:, None]
                ev = {"ramp": np.abs(dk) >= tau, "rising ramp": dk >= tau, "falling ramp": dk <= -tau}
                ev["non-ramp"], ev["overall"] = ~ev["ramp"], np.ones_like(ev["ramp"])
                for e in EVENTS:
                    p_ev[e] += ev[e].sum(1)
                    rep_n[e].append(ev[e].sum(0))
                for nm in ALPHA:
                    cov = (ys >= ci[nm][0][:, None]) & (ys <= ci[nm][1][:, None])
                    for e in EVENTS:
                        ec = cov & ev[e]
                        p_ec[nm][e] += ec.sum(1)
                        rep_c[nm][e].append(ec.sum(0))
            t = {"null_ramp_rate": float(p_ev["ramp"].sum() / M / len(y)),
                 "null_ramp_rate_range": [float(x) for x in np.quantile(np.concatenate(rep_n["ramp"]) / len(y), [0.025, 0.975])]}
            for nm in ALPHA:
                t[nm] = {}
                for e in EVENTS:
                    rn, rc = np.concatenate(rep_n[e]), np.concatenate(rep_c[nm][e])
                    num_n, den_n = p_ec[nm][e] / M, p_ev[e] / M
                    d = {"n_obs": int(obs_ev[e].sum()), "c_obs": float(cov_o[nm][obs_ev[e]].mean()),
                         "c_null": float(num_n.sum() / den_n.sum()),
                         "c_null_range": [float(x) for x in np.quantile(rc / np.maximum(rn, 1), [0.025, 0.975])]}
                    d |= ratio_boot(cov_o[nm] & obs_ev[e], obs_ev[e], num_n, den_n, day, sunny)
                    nominal = 1 - ALPHA[nm]
                    d["selection_share"] = float((nominal - d["c_null"]) / (nominal - d["c_obs"])) if e != "overall" and nominal > d["c_obs"] else None
                    t[nm][e] = d
            pr = p_ev["ramp"] / M
            b = np.clip(np.searchsorted(PBINS, pr, side="right") - 1, 0, len(PBINS) - 2)
            t["issue_time_bins_90"] = [{"p_ramp_bin": [PBINS[j], min(PBINS[j + 1], 1.0)], "n": int((b == j).sum()),
                                        "c_obs": float(cov_o["90"][b == j].mean()) if (b == j).any() else None,
                                        "c_null": float((p_ec["90"]["overall"][b == j] / M).mean()) if (b == j).any() else None,
                                        "observed_ramp_rate": float(c["ramp"][b == j].mean()) if (b == j).any() else None,
                                        "mean_p_ramp": float(pr[b == j].mean()) if (b == j).any() else None} for j in range(len(PBINS) - 1)]
            out[tail] = t
        res["models"][m] = out
        r = out["T-exp"]["90"]["ramp"]
        print(f"{m:9s} 90% ramp: obs {r['c_obs']:.3f}  null {r['c_null']:.3f} {np.round(r['c_null_range'], 3).tolist()}  gap {r['gap']:+.3f} "
              f"{np.round(r['gap_ci'], 3).tolist()}  selection share {r['selection_share']:.2f} | null ramp rate {out['T-exp']['null_ramp_rate']:.3f} "
              f"(obs {res['observed_ramp_rate']:.3f}) | T-lin null {out['T-lin']['90']['ramp']['c_null']:.3f}", flush=True)
    r = res["models"]["G3''-F"]["T-exp"]["90"]["ramp"]
    s = r["selection_share"]
    res["decision"] = {"shortfall": "genuine shortfall" if r["gap_ci"][1] < 0 else "no shortfall beyond selection shown",
                       "gap_composition": "mostly selection" if s >= 2 / 3 else "mostly miscalibration" if s <= 1 / 3 else "both"}
    print("decision:", res["decision"])
    print("saved", save("z1_ramp_coverage_null.json", res))


if __name__ == "__main__":
    main()
