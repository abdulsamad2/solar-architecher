"""Revision R-R4: ramp warnings - value curves with uncertainty, direction, reliability and timing
(protocol revision/protocols/R-R4_ramp_warnings.md). Post hoc test, frozen predictions, CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from src.events import implied_event_probability
from src.revision_common import LEVELS, frozen_test_predictions, save, test_context
from src.revision_posthoc_test import members, test_k
from src.revision_x2_linear_pool import _cdf
from src.revision_y3_operational_value import dev_quantiles

PP, MED = "G3''-F", list(LEVELS).index(0.5)
ALPHAS = np.round(np.concatenate([np.arange(0.02, 0.30, 0.02), np.arange(0.30, 0.91, 0.05)]), 2)
CAP_DROP = 0.2 * 30.1


def isotonic(x, y):
    """Isotonic (non-decreasing) regression of y on x by pooling adjacent violators; returns a step-interpolating predictor."""
    o = np.argsort(x, kind="stable")
    xs, ys = x[o], y[o].astype(float)
    val, wt, xr = [], [], []
    for xi, yi in zip(xs, ys):
        val.append(yi); wt.append(1.0); xr.append(xi)
        while len(val) > 1 and val[-2] > val[-1]:
            w = wt[-2] + wt[-1]
            val[-2:] = [(val[-2] * wt[-2] + val[-1] * wt[-1]) / w]
            wt[-2:] = [w]
            xr[-2:] = [xr[-2]]
    xr, val = np.array(xr), np.array(val)
    return lambda p: val[np.clip(np.searchsorted(xr, p, side="right") - 1, 0, len(val) - 1)]


def counts(warn, lab, inv, nd):
    """Per-day sums of hits, false alarms, misses and samples."""
    return np.stack([np.bincount(inv, weights=x.astype(float), minlength=nd) for x in (warn & lab, warn & ~lab, ~warn & lab, np.ones_like(lab))], 1)


def value(c, a):
    h, f, m, n = c.sum(0)
    s = (h + m) / n
    e_f, e_c, e_p = (a * (h + f) + m) / n, min(a, s), a * s
    return (e_c - e_f) / (e_c - e_p) if e_c > e_p else 0.0


def boot_idx(sunny_day, n_boot=2000, seed=20261022):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(sunny_day), np.flatnonzero(~sunny_day)]
    return [np.concatenate([rng.choice(g, len(g)) for g in groups if len(g)]) for _ in range(n_boot)]


def main(forecaster=None, out_name="r4_ramp_warnings.json", protocol="revision/protocols/R-R4_ramp_warnings.md"):
    """forecaster: test quantiles (kW) that take the place of the frozen forecaster (R-R12); None reproduces R-R4."""
    c = test_context()
    q = frozen_test_predictions(c["sample_ids"])
    q["PV-only P15"] = np.nanmean(members("P15")["pred_q"], 0)
    if forecaster is not None:
        q["frozen"], q[PP] = q[PP], forecaster
    y, day, sunny, kn, dk = c["y"], c["day"], c["sunny"], c["k_now"], c["dk"]
    tau = c["tau"]
    days, inv = np.unique(day, return_inverse=True)
    sday = np.zeros(len(days), bool)
    sday[inv] = sunny
    idx = boot_idx(sday)
    pv_now = c["pv"][c["ds"].hist_rows[:, -1]]
    labels = {"any ramp": np.abs(dk) >= tau, "falling ramp": dk <= -tau, "power drop >= 20% of capacity": y <= pv_now - CAP_DROP}
    names = {"calm expert": "E6-S", "forecaster": PP, "direct expert": "E4-B-fit", "PV-only": "PV-only P15"}
    pairs = [("probability, forecaster", "median, forecaster"), ("probability, forecaster", "probability, PV-only"),
             ("probability, forecaster", "probability, calm expert"), ("probability, forecaster", "probability, direct expert")]
    if forecaster is not None:
        names["frozen forecaster"] = "frozen"
        pairs.append(("probability, forecaster", "probability, frozen forecaster"))
    qk = {n: test_k(c, q[m]) for n, m in names.items()}
    prob = {ev: {} for ev in labels}
    for n in names:
        prob["any ramp"][n] = np.concatenate([implied_event_probability(qk[n][i:i + 8192], LEVELS, kn[i:i + 8192], tau) for i in range(0, len(kn), 8192)])
        prob["falling ramp"][n] = _cdf((kn - tau)[:, None], qk[n])[:, 0]
        prob["power drop >= 20% of capacity"][n] = _cdf((pv_now - CAP_DROP)[:, None], q[names[n]])[:, 0]
    # isotonic recalibration of the calm expert's any-ramp probability on development out-of-fold forecasts
    _, gacc, cnt, meta = dev_quantiles()
    gd = np.array([gacc[k] / cnt[k] for k in gacc])
    ld = np.array([meta[k][2] for k in gacc], float)
    prob["any ramp"]["calm expert, recalibrated"] = isotonic(gd, ld)(prob["any ramp"]["calm expert"])
    med = {n: qk[n][:, MED] - kn for n in ("forecaster", "direct expert")}
    medw = {"any ramp": {n: np.abs(v) >= tau for n, v in med.items()}, "falling ramp": {n: v <= -tau for n, v in med.items()},
            "power drop >= 20% of capacity": {n: q[names[n]][:, MED] <= pv_now - CAP_DROP for n in med}}

    res = {"protocol": protocol, "alphas": ALPHAS.tolist(), "events": {}}
    for ev, lab in labels.items():
        s = float(lab.mean())
        src = {f"probability, {n}": (lambda a, p=p: p >= a) for n, p in prob[ev].items()} | {f"median, {n}": (lambda a, w=w: w) for n, w in medw[ev].items()}
        cs = {k: [counts(fn(a), lab, inv, len(days)) for a in ALPHAS] for k, fn in src.items()}
        out = {"base_rate": s, "n_events": int(lab.sum()), "sources": {}, "differences": {}}
        for k in src:
            v = [value(cs[k][j], a) for j, a in enumerate(ALPHAS)]
            b = np.array([[value(cs[k][j][ii], a) for j, a in enumerate(ALPHAS)] for ii in idx])
            ab = int(np.argmin(np.abs(ALPHAS - s)))
            h, f, m, n = cs[k][ab].sum(0)
            out["sources"][k] = {"value": v, "lo": np.quantile(b, 0.025, 0).tolist(), "hi": np.quantile(b, 0.975, 0).tolist(),
                                 "at_base_rate": {"alpha": float(ALPHAS[ab]), "value": v[ab], "hit_rate": float(h / (h + m)), "false_detection_rate": float(f / (n - h - m))}}
        for k, p in prob[ev].items():
            bins = np.clip((p * 10).astype(int), 0, 9)
            out["sources"][f"probability, {k}"] |= {
                "brier_skill": float(1 - np.mean((p - lab) ** 2) / (s * (1 - s))),
                "reliability": [{"bin": int(b_), "n": int((bins == b_).sum()), "mean_p": float(p[bins == b_].mean()), "observed": float(lab[bins == b_].mean())}
                                for b_ in range(10) if (bins == b_).any()]}

        def diff(a_, b_):
            d = np.array([[value(cs[a_][j][ii], a) - value(cs[b_][j][ii], a) for j, a in enumerate(ALPHAS)] for ii in idx])
            pt = [value(cs[a_][j], a) - value(cs[b_][j], a) for j, a in enumerate(ALPHAS)]
            return {"diff": pt, "lo": np.quantile(d, 0.025, 0).tolist(), "hi": np.quantile(d, 0.975, 0).tolist()}

        for a_, b_ in pairs:
            out["differences"][f"{a_} - {b_}"] = diff(a_, b_)
        res["events"][ev] = out
        ab = int(np.argmin(np.abs(ALPHAS - s)))
        print(f"\n== {ev}: base rate {s:.3f}, n {lab.sum()}")
        for k, v in out["sources"].items():
            bs = f"  Brier skill {v['brier_skill']:+.3f}" if "brier_skill" in v else ""
            pos = [a for a, x in zip(ALPHAS, v["lo"]) if x > 0]
            print(f"  {k:38s} V(base) {v['value'][ab]:.3f} [{v['lo'][ab]:.3f}, {v['hi'][ab]:.3f}]  hit {v['at_base_rate']['hit_rate']:.2f} false-det "
                  f"{v['at_base_rate']['false_detection_rate']:.2f}  clearly positive for a in [{min(pos) if pos else '-'}, {max(pos) if pos else '-'}]{bs}")
        for k, d in out["differences"].items():
            better = [float(a) for a, x in zip(ALPHAS, d["lo"]) if x > 0]
            worse = [float(a) for a, x in zip(ALPHAS, d["hi"]) if x < 0]
            print(f"  {k}: at base {d['diff'][ab]:+.3f} [{d['lo'][ab]:+.3f}, {d['hi'][ab]:+.3f}]; first better for a in {[min(better), max(better)] if better else None}"
                  f"; second better for a in {[min(worse), max(worse)] if worse else None}")

    # --- timing of falling-ramp warnings
    lab, p = labels["falling ramp"], prob["falling ramp"]["forecaster"]
    s = float(lab.mean())
    warn = p >= s
    rec_t = c["rec_t"].astype("datetime64[us]").astype(np.int64)
    order = np.argsort(rec_t, kind="stable")
    ts = rec_t[order]
    rd = json.loads(Path("reports/test_ramp_definition_frozen.json").read_text())
    from src.clear_sky import ClearSkyPV
    k_rec = (c["pv"] / np.maximum(ClearSkyPV(rd["anchor_coef"])(c["rec_t"]), 1.0))[order]
    t0 = c["org"]
    lead = np.full(len(y), np.nan)
    for i in np.flatnonzero(lab):
        a, b = np.searchsorted(ts, t0[i], side="right"), np.searchsorted(ts, t0[i] + 15 * 60_000_000 + 30_000_000, side="right")
        hit = np.flatnonzero(k_rec[a:b] <= kn[i] - tau)
        if len(hit):
            lead[i] = (ts[a + hit[0]] - t0[i]) / 60e6
    ok = lab & np.isfinite(lead)
    first = np.r_[True, (day[1:] != day[:-1]) | (np.diff(c["tgt"]) > 90_000_000)]
    ep = lab & (first | ~np.r_[False, lab[:-1]])
    fa = warn & ~lab
    fa_ep = fa & (first | ~np.r_[False, fa[:-1]])
    res["timing_falling_ramps"] = {
        "threshold": s, "n_falling": int(lab.sum()), "share_warned": float(warn[lab].mean()), "n_with_crossing_found": int(ok.sum()),
        "minutes_to_crossing_all": [float(x) for x in np.nanquantile(lead[ok], [0.1, 0.25, 0.5, 0.75, 0.9])],
        "minutes_to_crossing_warned": [float(x) for x in np.nanquantile(lead[ok & warn], [0.1, 0.25, 0.5, 0.75, 0.9])],
        "share_of_falling_ramps_warned_with_at_least_7_min": float((warn & ok & (lead >= 7)).sum() / lab.sum()),
        "share_of_falling_ramps_with_crossing_at_least_7_min_after_issue": float((ok & (lead >= 7)).sum() / lab.sum()),
        "episodes": int(ep.sum()), "episodes_warned": float(warn[ep].mean()),
        "false_alarm_episodes_per_cloudy_day": float(fa_ep[~sunny].sum() / len(np.unique(day[~sunny])))}
    print("\ntiming (falling ramps, forecaster probability >= base rate):", json.dumps(res["timing_falling_ramps"], indent=1))
    if out_name:
        print("saved", save(out_name, res))
    return res


if __name__ == "__main__":
    main()
