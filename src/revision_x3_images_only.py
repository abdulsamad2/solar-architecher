"""Revision R-X3: images-only arms (protocol revision/protocols/R-X3_images_only.md).

Development: out-of-fold validation predictions per seed, paired day-block bootstrap (as src/revision_e1_e4.py).
Post hoc test: 15-member ensembles from the per-member dump (python -m src.revision_posthoc_dump --models I15 [I60]).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import src.revision_e1_e4 as e1
from src.e8_features import FEAT
from src.gate2_report import ramp_labels
from src.metrics import crps_quantile_interp
from src.revision_common import LEVELS, day_bootstrap, save, test_context
from src.revision_posthoc_test import members
from src.train import build_model

e1.ROOT |= {"I15": "experiments/R_X3/I15/csi", "I60": "experiments/R_X3/I60/csi"}
STRATA = ("overall", "sunny", "cloudy", "ramp", "non-ramp")


def have(n):
    return all((Path(e1.ROOT[n]) / f"seed{s}" / f"fold_{k}" / "preds_val.npz").exists() for s in (42, 123, 456) for k in range(5))


def posthoc(arms_pairs):
    c = test_context()
    h60, y, day, sunny = c["h60"], c["y"], c["day"], c["sunny"]
    arms = {}
    for n in ("I15", "I60", "E4-B-fit", "A2", "P15", "P60"):
        d = members(n)
        if d is not None:
            arms[n] = np.mean(d["pred_q"], 0) if n in ("A2", "P60", "I60") else np.nanmean(d["pred_q"], 0)
    res = {"available": sorted(arms), "crps": {}, "diffs": {}}
    for pop, sel in (("full", np.ones(len(y), bool)), ("h60", h60)):
        masks = {g: c["masks"][g][sel] for g in STRATA}
        res["crps"][pop] = {}
        for n, q in arms.items():
            if np.isfinite(q[sel]).all():
                cc = crps_quantile_interp(q[sel], y[sel], LEVELS)
                med = q[sel][:, 4]
                res["crps"][pop][n] = {g: {"crps": float(cc[v].mean()), "rmse": float(np.sqrt(np.mean((med[v] - y[sel][v]) ** 2))),
                                           "mae": float(np.abs(med[v] - y[sel][v]).mean())} for g, v in masks.items()}
    for a, b, pop in arms_pairs:
        if a in arms and b in arms:
            sel = h60 if pop == "h60" else np.ones(len(y), bool)
            d = crps_quantile_interp(arms[a][sel], y[sel], LEVELS) - crps_quantile_interp(arms[b][sel], y[sel], LEVELS)
            res["diffs"][f"{a} - {b} ({pop})"] = {g: day_bootstrap(d[c["masks"][g][sel]], day[sel][c["masks"][g][sel]], sunny[sel][c["masks"][g][sel]])
                                                 for g in STRATA}
    return res


def main():
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    rl = ramp_labels("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz", folds_doc)
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = win["sample_ids"][win["eligible_common"]]
    res = {"protocol": "revision/protocols/R-X3_images_only.md", "trained": {n: have(n) for n in ("I15", "I60")}}
    pairs_t = []
    if have("I15"):
        res["dev_full"] = e1.compare(("I15", "E4-B-fit", "P15"), [("I15", "E4-B-fit"), ("I15", "P15")], None, rl, folds_doc, "X3 full")
        pairs_t += [("I15", "E4-B-fit", "full"), ("I15", "P15", "full")]
    if have("I60"):
        res["dev_common"] = e1.compare(("I60", "A2", "P60"), [("I60", "A2"), ("I60", "P60")], common, rl, folds_doc, "X3 common")
        pairs_t += [("I60", "A2", "h60"), ("I60", "P60", "h60")]
    res["parameters"] = {}
    for n in ("I15", "I60", "E4-B-fit", "A2", "P15", "P60"):
        p = Path(e1.ROOT[n]) / "seed42" / "config.json"
        if p.exists():
            res["parameters"][n] = int(sum(x.numel() for x in build_model(json.loads(p.read_text())).parameters()))
    res["posthoc_test"] = {"label": "POST HOC (test)", **posthoc(pairs_t)}
    out = save("x3_images_only.json", res)
    print("saved", out)
    for pop in ("dev_full", "dev_common"):
        if pop in res:
            print(pop, {n: {g: round(v, 3) for g, v in d.items() if g in ("overall", "ramp", "sunny", "cloudy")} for n, d in res[pop]["seed_mean"].items()})
            print("  pairs", {k: {s: tuple(round(x, 3) for x in v["overall"]) for s, v in d.items()} for k, d in res[pop]["pairs"].items()})
    print("test", {pop: {n: round(v["overall"]["crps"], 3) for n, v in d.items()} for pop, d in res["posthoc_test"]["crps"].items()})
    print("test diffs", {k: {g: tuple(round(x, 3) for x in v[g]) for g in ("overall", "ramp")} for k, v in res["posthoc_test"]["diffs"].items()})


if __name__ == "__main__":
    main()
