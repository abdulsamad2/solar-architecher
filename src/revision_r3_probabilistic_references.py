"""Revision R-R3: probabilistic reference forecasts for the CRPS skill on the official test set
(protocol revision/protocols/R-R3_probabilistic_references.md). Post hoc, CPU only, no training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.clear_sky import ClearSkyPV
from src.metrics import crps_quantile_interp
from src.revision_common import H5, LEVELS, MAN, day_bootstrap, frozen_test_predictions, save, test_context
from src.revision_posthoc_test import members
from src.revision_r2_day_level_inference import sign_flip
from src.skippd import SkippdForecastDataset

PP = "G3''-F"
MIN = 60_000_000  # microseconds


def quant(x):
    return np.quantile(x, LEVELS)


def main():
    c = test_context()
    ds, pv, rec_t, y, day, sunny = c["ds"], c["pv"], c["rec_t"], c["y"], c["day"], c["sunny"]
    rd = json.loads(Path("reports/test_ramp_definition_frozen.json").read_text())
    cs = ClearSkyPV(rd["anchor_coef"])
    pc_rec = np.maximum(cs(rec_t), 1.0)
    k_rec = pv / pc_rec
    t_rec = rec_t.astype("datetime64[us]").astype(np.int64)
    d_rec = rec_t.astype("datetime64[D]")
    now, tgt = ds.hist_rows[:, -1], ds.target_rows
    pc_t, k_now, t0 = pc_rec[tgt], k_rec[now], t_rec[now]

    # development: 15-min changes of K (all samples) and K by hour of day (all records), same clear-sky anchor
    dev = SkippdForecastDataset(H5, MAN, "dev", preload=False)
    m = np.load(MAN)
    sel = m["hdf5_group"] == 0
    dev_t = m["times_us"][sel][np.argsort(m["hdf5_row"][sel])].astype("datetime64[us]")
    k_dev = dev.pv.numpy().astype(float) / np.maximum(cs(dev_t), 1.0)
    dk_dev = k_dev[dev.target_rows] - k_dev[dev.hist_rows[:, -1]]
    hour_dev = dev_t.astype("datetime64[h]").astype(np.int64) % 24
    q_static = quant(dk_dev)
    ch = {h: quant(k_dev[hour_dev == h]) for h in np.unique(hour_dev)}

    order = np.argsort(t_rec, kind="stable")
    ts, ks, dsd = t_rec[order], k_rec[order], d_rec[order]
    # matured 15-min changes of K on the test days (target time known), for the adaptive dressing
    s_t0, s_tt, s_dk, s_day = t0, t_rec[tgt], k_rec[tgt] - k_now, d_rec[now]

    n = len(y)
    q = {"PeEn-120": np.empty((n, 9)), "PeEn-60": np.empty((n, 9)), "PeEn-30": np.empty((n, 9)), "SP-dressed static": np.empty((n, 9)),
         "SP-dressed adaptive": np.empty((n, 9)), "CH-PeEn": np.empty((n, 9))}
    n_point, n_fallback = {w: 0 for w in (120, 60, 30)}, 0
    hour_t = t_rec[tgt].astype("datetime64[us]").astype("datetime64[h]").astype(np.int64) % 24
    for i in range(n):
        hi = np.searchsorted(ts, t0[i], side="right")
        for w in (120, 60, 30):
            lo = np.searchsorted(ts, t0[i] - w * MIN, side="right")
            mem = ks[lo:hi][dsd[lo:hi] == d_rec[now[i]]]
            if len(mem) >= 5:
                q[f"PeEn-{w}"][i] = quant(mem)
            else:
                q[f"PeEn-{w}"][i] = k_now[i]
                n_point[w] += 1
        q["SP-dressed static"][i] = k_now[i] + q_static
        past = (s_day == d_rec[now[i]]) & (s_tt <= t0[i]) & (s_t0 > t0[i] - 120 * MIN)
        if past.sum() >= 10:
            q["SP-dressed adaptive"][i] = k_now[i] + quant(s_dk[past])
        else:
            q["SP-dressed adaptive"][i] = k_now[i] + q_static
            n_fallback += 1
        q["CH-PeEn"][i] = ch[hour_t[i]] if hour_t[i] in ch else quant(k_dev)
    q = {k: np.maximum(v, 0.0) * pc_t[:, None] for k, v in q.items()}

    fz = frozen_test_predictions(c["sample_ids"])
    crps = {k: crps_quantile_interp(v, y, LEVELS) for k, v in q.items()}
    crps[PP] = crps_quantile_interp(fz[PP], y, LEVELS)
    crps["E4-B-fit"] = crps_quantile_interp(fz["E4-B-fit"], y, LEVELS)
    crps["PV-only P15 (learned)"] = crps_quantile_interp(np.nanmean(members("P15")["pred_q"], 0), y, LEVELS)
    crps["persistence of K, own clear-sky (point)"] = np.abs(k_now * pc_t - y)
    lines = Path("paper/data/t4_main_test.csv").read_text(encoding="utf-8").splitlines()
    ci = lines[0].split(",").index("crps_kw")
    ref_sp = next(float(r.split(",")[ci]) for r in lines[1:] if "persistence" in r.split(",")[0].lower())
    assert abs(crps[PP].mean() - 1.006) < 6e-4, crps[PP].mean()

    res = {"protocol": "revision/protocols/R-R3_probabilistic_references.md", "n": int(n),
           "official_smart_persistence_mae_table2": ref_sp, "peen_point_fallback": n_point, "adaptive_static_fallback": int(n_fallback),
           "dev_dk_quantiles": q_static.tolist(), "references": {}}
    days, inv = np.unique(day, return_inverse=True)
    for k, v in crps.items():
        r = {"crps": float(v.mean()), "sunny": float(v[sunny].mean()), "cloudy": float(v[~sunny].mean())}
        if k != PP:
            d = crps[PP] - v
            r |= {"skill_of_proposed": float(1 - crps[PP].mean() / v.mean()), "proposed_minus_reference": day_bootstrap(d, day, sunny),
                  "sign_flip_p": sign_flip(np.bincount(inv, weights=d), np.bincount(inv).astype(float))[0]}
        res["references"][k] = r
    naive = ("PeEn-120", "SP-dressed static", "SP-dressed adaptive", "CH-PeEn")
    best = min(naive, key=lambda k: res["references"][k]["crps"])
    res["headline_reference"] = best
    res["headline_skill"] = res["references"][best]["skill_of_proposed"]
    res["skill_vs_official_smart_persistence_point"] = float(1 - crps[PP].mean() / ref_sp)
    for k, r in sorted(res["references"].items(), key=lambda kv: -kv[1]["crps"]):
        extra = (f"  skill of proposed {r['skill_of_proposed']:.1%}  diff {r['proposed_minus_reference'][0]:+.3f} "
                 f"[{r['proposed_minus_reference'][1]:+.3f}, {r['proposed_minus_reference'][2]:+.3f}]  p {r['sign_flip_p']:.5f}") if k != PP else ""
        print(f"{k:42s} CRPS {r['crps']:.3f}  sunny {r['sunny']:.3f}  cloudy {r['cloudy']:.3f}{extra}")
    print(f"official smart persistence (point, Table 2): {ref_sp:.3f} -> skill {res['skill_vs_official_smart_persistence_point']:.1%}")
    print(f"headline reference (lowest-CRPS naive probabilistic): {best}, skill {res['headline_skill']:.1%}; PeEn point fallbacks {n_point}, adaptive fallbacks {n_fallback}")
    print("saved", save("r3_probabilistic_references.json", res))


if __name__ == "__main__":
    main()
