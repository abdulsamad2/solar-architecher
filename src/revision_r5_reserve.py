"""Revision R-R5: upward reserve compared at equal 95% reliability (protocol revision/protocols/R-R5_reserve_equal_reliability.md).
Post hoc test, frozen predictions, CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.baselines import official_persistence
from src.clear_sky import ClearSkyPV
from src.revision_common import H5, LEVELS, MAN, day_bootstrap, frozen_test_predictions, save, test_context
from src.revision_posthoc_test import members
from src.revision_r2_day_level_inference import sign_flip
from src.skippd import SkippdForecastDataset

PP, MED, CAP, MIN = "G3''-F", list(LEVELS).index(0.5), 30.1, 60_000_000


def equalise(base, f, y, target=0.95):
    """delta such that reliability mean(y >= f - max(base + delta, 0)) equals the target (bisection)."""
    lo, hi = -50.0, 50.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if np.mean(y >= f - np.maximum(base + mid, 0)) < target:
            lo = mid
        else:
            hi = mid
    return hi


def dressed_q05(c):
    """5% quantile and point value of smart persistence dressed with the 15-min changes of K matured in the last 120 min (R-R3)."""
    ds, pv, rec_t = c["ds"], c["pv"], c["rec_t"]
    rd = json.loads(Path("reports/test_ramp_definition_frozen.json").read_text())
    cs = ClearSkyPV(rd["anchor_coef"])
    pc = np.maximum(cs(rec_t), 1.0)
    k = pv / pc
    t = rec_t.astype("datetime64[us]").astype(np.int64)
    d = rec_t.astype("datetime64[D]")
    now, tgt = ds.hist_rows[:, -1], ds.target_rows
    dev = SkippdForecastDataset(H5, MAN, "dev", preload=False)
    m = np.load(MAN)
    sel = m["hdf5_group"] == 0
    dev_t = m["times_us"][sel][np.argsort(m["hdf5_row"][sel])].astype("datetime64[us]")
    kd = dev.pv.numpy().astype(float) / np.maximum(cs(dev_t), 1.0)
    static = np.quantile(kd[dev.target_rows] - kd[dev.hist_rows[:, -1]], 0.05)
    dk, t0, tt, dd = k[tgt] - k[now], t[now], t[tgt], d[now]
    q05 = np.empty(len(now))
    for i in range(len(now)):
        past = (dd == dd[i]) & (tt <= t0[i]) & (t0 > t0[i] - 120 * MIN)
        q05[i] = np.quantile(dk[past], 0.05) if past.sum() >= 10 else static
    return np.maximum(k[now] + q05, 0) * pc[tgt], k[now] * pc[tgt]


def main(forecaster=None, name="method v2", out="r5_reserve_equal_reliability.json", protocol="revision/protocols/R-R5_reserve_equal_reliability.md"):
    """forecaster: test quantiles (kW) of the forecaster compared with the others (R-R12; the frozen forecaster then
    becomes one of the alternatives); None reproduces R-R5."""
    c = test_context()
    q = frozen_test_predictions(c["sample_ids"])
    q["PV-only P15"] = np.nanmean(members("P15")["pred_q"], 0)
    pp, others = PP, ("E4-B-fit", "E6-S", "G4", "PV-only P15")
    if forecaster is not None:
        q[name], pp, others = forecaster, name, (PP,) + others
    y, day, sunny, dk, tau = c["y"], c["day"], c["sunny"], c["dk"], c["tau"]
    pers = official_persistence(c["ds"].origin_time.astype(object), c["pv"][c["ds"].hist_rows[:, -1]])
    sp05, spf = dressed_q05(c)
    spec = {m: (q[m][:, MED], q[m][:, MED] - q[m][:, 0]) for m in (pp,) + others}
    spec["smart persistence, dressed (120 min)"] = (spf, spf - sp05)
    spec["constant reserve, official smart persistence"] = (pers, np.zeros(len(y)))
    spec["constant reserve, forecaster median"] = (q[pp][:, MED], np.zeros(len(y)))
    strata = {"overall": np.ones(len(y), bool), "cloudy": ~sunny, "falling ramp": dk <= -tau, "rising ramp": dk >= tau, "non-ramp": np.abs(dk) < tau}
    res = {"protocol": protocol, "forecaster": pp, "target_reliability": 0.95, "methods": {}, "forecaster_minus": {}}
    r = {}
    for m, (f, base) in spec.items():
        delta = equalise(base, f, y)
        r[m] = np.maximum(base + delta, 0)
        short = np.maximum(f - r[m] - y, 0)
        res["methods"][m] = {"delta_kw": float(delta), "by_stratum": {
            g: {"reliability": float((short[v] == 0).mean()), "mean_reserve_kw": float(r[m][v].mean()), "mean_reserve_pct_capacity": float(100 * r[m][v].mean() / CAP),
                "mean_shortfall_when_short_kw": float(short[v][short[v] > 0].mean()) if (short[v] > 0).any() else 0.0,
                "p95_shortfall_kw": float(np.quantile(short[v], 0.95))} for g, v in strata.items()}}
    days, inv = np.unique(day, return_inverse=True)
    for m in spec:
        if m == pp:
            continue
        d = r[pp] - r[m]
        res["forecaster_minus"][m] = {"overall": day_bootstrap(d, day, sunny), "cloudy": day_bootstrap(d[~sunny], day[~sunny], sunny[~sunny]),
                                      "sign_flip_p": sign_flip(np.bincount(inv, weights=d), np.bincount(inv).astype(float))[0]}
    for m, v in res["methods"].items():
        o, fr = v["by_stratum"]["overall"], v["by_stratum"]["falling ramp"]
        dd = res["forecaster_minus"].get(m)
        extra = f" | forecaster minus this {dd['overall'][0]:+.2f} [{dd['overall'][1]:+.2f}, {dd['overall'][2]:+.2f}] p {dd['sign_flip_p']:.4f}" if dd else ""
        print(f"{m:46s} reserve {o['mean_reserve_kw']:.2f} kW ({o['mean_reserve_pct_capacity']:.1f}% cap), cloudy {v['by_stratum']['cloudy']['mean_reserve_kw']:.2f}; "
              f"reliability overall {o['reliability']:.3f}, falling ramps {fr['reliability']:.3f} (shortfall when short {fr['mean_shortfall_when_short_kw']:.2f} kW){extra}")
    if out:
        print("saved", save(out, res))
    return res


if __name__ == "__main__":
    main()
