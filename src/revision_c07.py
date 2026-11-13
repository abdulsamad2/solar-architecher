"""Revision C07 / C19: audit and past-only rerun of the test-day conformal initialisation.

Protocol: revision/protocols/R-C07_conformal_pastonly.md. Uses only frozen predictions and frozen OOF scores.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.cal2 import ALPHA, interval_score
from src.calibrate import base_interval
from src.final_eval import nearest_init, rolling_conformal_daily
from src.revision_common import LEVELS, frozen_test_predictions, save, test_context

MATURE_US = int(15.5 * 60e6)  # target matched within +-30 s of origin + 15 min
MODELS = ("G3''-F", "E4-B-fit", "E6-S", "G4")


def past_only_init(dev_origin_us, dev_scores, day_start_us, W):
    """The W latest development scores whose target time is strictly before the day's first issue time."""
    ok = np.flatnonzero(dev_origin_us + MATURE_US < day_start_us)
    pick = ok[np.argsort(dev_origin_us[ok], kind="stable")][-W:]
    return dev_scores[pick], pick


def interval_table(clo, chi, y, a, masks):
    return {g: {"coverage": float(np.mean((y[v] >= clo[v]) & (y[v] <= chi[v]))), "width_kw": float(np.mean(chi[v] - clo[v])),
                "interval_score": float(interval_score(clo[v], chi[v], y[v], a).mean()),
                "below": float(np.mean(y[v] < clo[v])), "above": float(np.mean(y[v] > chi[v])), "n": int(v.sum())}
            for g, v in masks.items() if v.any()}


def window_composition(org, tgt, day, init_len, W):
    """Share of the window that is still initial scores at each forecast (matured test scores push them out)."""
    out = {}
    for d in np.unique(day):
        sel = np.flatnonzero(day == d)
        o = np.sort(org[sel])
        t = np.sort(tgt[sel])
        matured = np.searchsorted(t, o, side="right")  # test scores matured by each issue time
        share = np.clip(init_len[d] - np.maximum(init_len[d] + matured - W, 0), 0, None) / np.minimum(init_len[d] + matured, W)
        out[d] = {"start": float(share[0]), "median": float(np.median(share)), "end": float(share[-1]), "n_forecasts": int(len(sel))}
    return out


def main():
    c = test_context()
    q = frozen_test_predictions(c["sample_ids"])
    cal = np.load("experiments/final/dev_oof_calibration_v1.npz")
    meta = json.loads(Path("experiments/final/dev_oof_calibration_v1.json").read_text())
    frozen = json.loads(Path("reports/test_results.json").read_text())
    y, day, org, tgt, masks = c["y"], c["day"], c["org"], c["tgt"], c["masks"]
    frozen_masks = {g: masks[g] for g in ("overall", "sunny", "cloudy", "ramp", "non-ramp", "sunrise ramp")}
    dev_o = cal["origin_us"].astype(np.int64)
    days = np.unique(day)
    start = {d: org[day == d].min() for d in days}
    res = {"protocol": "revision/protocols/R-C07_conformal_pastonly.md", "W": meta["W"], "audit": {}, "frozen_reproduced": {},
           "nearest_init": {}, "past_only": {}, "raw": {}, "window_composition": {}}
    for mm in MODELS:
        W = meta["W"][mm]
        # 1) audit of the frozen rule
        aud = {}
        for d in days:
            idx = np.argsort(np.abs(dev_o - start[d]), kind="stable")[:W]
            aud[d] = {"future_origin_share": float(np.mean(dev_o[idx] > start[d])),
                      "immature_share": float(np.mean(dev_o[idx] + MATURE_US >= start[d])),
                      "span_days": float((dev_o[idx].max() - dev_o[idx].min()) / 86_400e6),
                      "latest_init_minus_start_days": float((dev_o[idx].max() - start[d]) / 86_400e6)}
        res["audit"][mm] = {"per_day": aud, "mean_future_share": float(np.mean([v["future_origin_share"] for v in aud.values()])),
                            "max_future_share": float(np.max([v["future_origin_share"] for v in aud.values()]))}
        res["nearest_init"][mm], res["past_only"][mm], res["raw"][mm] = {}, {}, {}
        for nm, a in ALPHA.items():
            lo, hi = base_interval(q[mm], LEVELS, nm)
            res["raw"][mm][nm] = interval_table(lo, hi, y, a, masks)
            # 2) frozen rule, reproduced
            init_n = {d: nearest_init(dev_o, cal[f"{mm}|{nm}"], start[d], W) for d in days}
            clo, chi = rolling_conformal_daily(org, tgt, day, lo, hi, y, a, W, init_n)
            tab = interval_table(clo, chi, y, a, masks)
            res["nearest_init"][mm][nm] = tab
            for g in frozen_masks:
                for key in ("coverage", "width_kw", "interval_score"):
                    want = frozen["intervals"][mm][nm][g][key]
                    assert abs(tab[g][key] - want) < 1e-9, f"STOP: frozen H5 not reproduced ({mm} {nm} {g} {key})"
            res["frozen_reproduced"][f"{mm}|{nm}"] = True
            # 3) strictly past-only, matured initial scores
            init_p, lens, provenance = {}, {}, {}
            for d in days:
                s, pick = past_only_init(dev_o, cal[f"{mm}|{nm}"], start[d], W)
                init_p[d], lens[d] = s, len(s)
                provenance[d] = {"n": int(len(s)), "first_origin": str(np.datetime64(int(dev_o[pick].min()), "us")),
                                 "last_origin": str(np.datetime64(int(dev_o[pick].max()), "us")),
                                 "gap_to_start_days": float((start[d] - dev_o[pick].max()) / 86_400e6)}
            clo, chi = rolling_conformal_daily(org, tgt, day, lo, hi, y, a, W, init_p)
            res["past_only"][mm][nm] = interval_table(clo, chi, y, a, masks) | {"init_provenance": provenance}
            if nm == "90":
                res["window_composition"][mm] = window_composition(org, tgt, day, lens, W)
    p = save("c07_conformal_pastonly.json", res)
    g = "G3''-F"
    print(f"saved {p}")
    print("future-dated share (G3''-F): mean", round(res["audit"][g]["mean_future_share"], 3), "max", round(res["audit"][g]["max_future_share"], 3))
    for nm in ALPHA:
        for st in ("overall", "ramp", "sunny", "cloudy"):
            a, b, r = res["nearest_init"][g][nm][st], res["past_only"][g][nm][st], res["raw"][g][nm][st]
            print(f"{nm}% {st:8s} raw {r['coverage']:.3f} | frozen {a['coverage']:.3f} w {a['width_kw']:.2f} IS {a['interval_score']:.2f}"
                  f" | past-only {b['coverage']:.3f} w {b['width_kw']:.2f} IS {b['interval_score']:.2f}")


if __name__ == "__main__":
    main()
