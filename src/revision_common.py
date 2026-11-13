"""Shared helpers for the v5-review revision analyses (CPU only; no model inference).

Builds the official test-split context exactly as src/final_eval.evaluate does (targets, days, frozen sunny/ramp
strata, 60-min routing) without loading images to the GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.clear_sky import ClearSkyPV
from src.e8_audit import history_rows
from src.skippd import SUNNY_DATES, SkippdForecastDataset

H5 = "data/raw/2017_2019_images_pv_processed.hdf5"
MAN = "data/manifests/forecast_tol30_v2.npz"
LEVELS = np.array([.05, .1, .2, .3, .5, .7, .8, .9, .95])
RESULTS = Path("revision/results")


def test_context() -> dict:
    ds = SkippdForecastDataset(H5, MAN, "test", preload=False)
    m = np.load(MAN)
    sel = m["hdf5_group"] == 1
    rec_t = m["times_us"][sel][np.argsort(m["hdf5_row"][sel])].astype("datetime64[us]")
    pv = ds.pv.numpy().astype(float)
    origin_tl = m["origin"][m["is_test"]]
    tl = history_rows(m["times_us"], origin_tl, 4)
    h60 = (tl >= 0).all(1)
    rd = json.loads(Path("reports/test_ramp_definition_frozen.json").read_text())
    k = pv / np.maximum(ClearSkyPV(rd["anchor_coef"])(rec_t), 1.0)
    dk = k[ds.target_rows] - k[ds.hist_rows[:, -1]]
    ramp = np.abs(dk) >= rd["tau_test"]
    day = ds.day.astype(str)
    sunny = np.array([np.datetime64(d).astype(object) in SUNNY_DATES for d in day])
    y = pv[ds.target_rows]
    org = ds.origin_time.astype("datetime64[us]").astype(np.int64)
    tgt = ds.target_time.astype("datetime64[us]").astype(np.int64)
    masks = {"overall": np.ones(len(y), bool), "sunny": sunny, "cloudy": ~sunny, "ramp": ramp, "non-ramp": ~ramp,
             "rising ramp": ramp & (dk > 0), "falling ramp": ramp & (dk < 0), "sunrise ramp": ramp & ~h60}
    return {"ds": ds, "sample_ids": ds.sample_ids, "y": y, "day": day, "sunny": sunny, "ramp": ramp, "dk": dk,
            "k_now": k[ds.hist_rows[:, -1]], "tau": rd["tau_test"], "h60": h60, "org": org, "tgt": tgt,
            "masks": masks, "pv": pv, "rec_t": rec_t}


def frozen_test_predictions(sample_ids: np.ndarray) -> dict:
    p = np.load("experiments/final/test_predictions_v1.npz")
    assert np.array_equal(p["sample_ids"], sample_ids), "STOP: frozen prediction order differs"
    return {k.split("|", 1)[1]: p[k] for k in p.files if k.startswith("q|")} | {"sunset": p["sunset"]}


def day_bootstrap(diff, day, sunny_day, n_boot=10_000, seed=20261022):
    from src.final_eval import day_bootstrap as _db
    return _db(np.asarray(diff, float), np.asarray(day), np.asarray(sunny_day), n_boot, seed)


def save(name: str, obj) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=1, default=float))
    return p
