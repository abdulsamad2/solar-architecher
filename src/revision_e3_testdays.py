"""Revision E3b: forward models on the official test days inside the forward window (protocol
revision/protocols/R-E3b_testdays.md). Models, anchor, gates and conformal initial scores exactly as in
src/revision_e3.py; nothing is refitted. POST HOC; point estimates and per-day differences only (3 days).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.baselines import official_persistence
from src.cal2 import ALPHA, conformity, replay
from src.calibrate import base_interval
from src.final_eval import FEAT, H5, MAN, Split, record_table_unsorted, rolling_conformal_daily
from src.metrics import crps_quantile_interp
from src.revision_common import LEVELS, SUNNY_DATES, save
from src.revision_e3 import GAMMA, MODELS, SEEDS, W, W_ACI, ensemble_on, fit_anchor, fit_gates, selection_ensemble

DAYS = ("2019-08-10", "2019-10-14", "2019-10-19")


def main():
    import h5py
    with h5py.File(H5, "r") as f:
        gray = np.round(f["test/images_log"][...].astype(np.float32).mean(-1)).astype(np.uint8)
    m = np.load(MAN)
    sel = m["hdf5_group"] == 1
    t_test = m["times_us"][sel][np.argsort(m["hdf5_row"][sel])].astype("datetime64[us]")
    test = Split("test", torch.device("cuda"), record_table_unsorted(t_test, gray, FEAT / "gray_u8_test.npy"))
    anchor = fit_anchor(test)
    day = test.ds.day.astype(str)
    idx = np.flatnonzero(np.isin(day, DAYS))
    assert set(day[idx]) == set(DAYS), f"STOP: test days found {sorted(set(day[idx]))}"
    gates_by_seed = {s: fit_gates(s) for s in SEEDS}
    q, h60 = ensemble_on(test, idx, anchor, gates_by_seed)
    y = test.pv[test.ds.target_rows[idx]]
    d_ = day[idx]
    tau = gates_by_seed[SEEDS[0]][1]
    k = test.pv / np.maximum(anchor(test.rec_t), 1.0)
    dk = k[test.ds.target_rows[idx]] - k[test.ds.hist_rows[idx, -1]]
    ramp = np.abs(dk) >= tau
    masks = {"overall": np.ones(len(y), bool), "ramp": ramp, "non-ramp": ~ramp} | {d: d_ == d for d in DAYS}
    pers = official_persistence(test.ds.origin_time[idx].astype(object), test.pv[test.ds.hist_rows[idx, -1]])
    crps = {mm: crps_quantile_interp(q[mm], y, LEVELS) for mm in MODELS}
    crps["persistence"] = np.abs(pers - y)
    # context: the frozen 15-member models on the same samples
    fz = np.load("experiments/final/test_predictions_v1.npz")
    pos = {int(s): i for i, s in enumerate(fz["sample_ids"])}
    j = np.array([pos[int(s)] for s in test.ds.sample_ids[idx]])
    frozen = {mm: crps_quantile_interp(fz[f"q|{mm}"][j], y, LEVELS) for mm in ("G3''-F", "E4-B-fit", "E6-S", "G4")}
    res = {"protocol": "revision/protocols/R-E3b_testdays.md", "label": "POST HOC (forward models on official test days)",
           "days": list(DAYS), "sunny": {d: np.datetime64(d).astype(object) in SUNNY_DATES for d in DAYS},
           "n": {g: int(v.sum()) for g, v in masks.items()}, "ramp_tau": tau, "routing_share_G3pp": float(h60.mean()),
           "crps": {mm: {g: float(c[v].mean()) for g, v in masks.items() if v.any()} for mm, c in crps.items()},
           "rmse_median": {mm: float(np.sqrt(np.mean((q[mm][:, 4] - y) ** 2))) for mm in MODELS},
           "frozen_models_same_samples_crps": {mm: {g: float(c[v].mean()) for g, v in masks.items() if v.any()} for mm, c in frozen.items()},
           "diff_G3pp_F_minus": {}, "intervals_90": {}}
    res["rmse_median"]["persistence"] = float(np.sqrt(np.mean((pers - y) ** 2)))
    for comp in ("E4-B-fit", "E6-S", "G4", "P15", "persistence"):
        dd = crps["G3''-F"] - crps[comp]
        res["diff_G3pp_F_minus"][comp] = {g: float(dd[v].mean()) for g, v in masks.items() if v.any()}
    # 90% intervals: methods and initial scores exactly as in src/revision_e3.py
    qs, ys, os_ = selection_ensemble(gates_by_seed)
    org = test.ds.origin_time.astype("datetime64[us]").astype(np.int64)[idx]
    tgt = test.ds.target_time.astype("datetime64[us]").astype(np.int64)[idx]
    o = np.argsort(org, kind="stable")  # replay in time order (the test file is not stored chronologically)
    order_s = np.argsort(os_, kind="stable")
    assert os_.max() + 16 * 60e6 < org.min(), "STOP: selection scores not matured before the test days"
    a, nm = ALPHA["90"], "90"
    for mm in MODELS:
        lo_s, hi_s = base_interval(qs[mm], LEVELS, nm)
        sc = conformity(lo_s, hi_s, ys)[order_s]
        lo, hi = base_interval(q[mm], LEVELS, nm)
        lo, hi, yy, dd_, oo, tt = lo[o], hi[o], y[o], d_[o], org[o], tgt[o]
        ww = W[mm]
        meth = {"none": (lo, hi),
                "daily-reset rolling": rolling_conformal_daily(oo, tt, dd_, lo, hi, yy, a, ww, {x: sc[-ww:] for x in np.unique(dd_)}),
                "continuous rolling": tuple(replay(oo, tt, lo, hi, yy, a, "rolling", ww, sc[-ww:])[k_] for k_ in ("lo", "hi")),
                "ACI": tuple(replay(oo, tt, lo, hi, yy, a, "aci", GAMMA, sc[-W_ACI:])[k_] for k_ in ("lo", "hi"))}
        rmask = ramp[o]
        res["intervals_90"][mm] = {k_: {"overall": float(np.mean((yy >= l_) & (yy <= h_))),
                                        "ramp": float(np.mean(((yy >= l_) & (yy <= h_))[rmask])) if rmask.any() else None}
                                   for k_, (l_, h_) in meth.items()}
    p = save("e3_testdays.json", res)
    print("saved", p, "n", len(idx))
    print("CRPS overall:", {mm: round(v["overall"], 3) for mm, v in res["crps"].items()})
    print("proposed minus:", {c: {g: round(v, 3) for g, v in dd.items()} for c, dd in res["diff_G3pp_F_minus"].items()})
    print("frozen same samples:", {mm: round(v["overall"], 3) for mm, v in res["frozen_models_same_samples_crps"].items()})
    print("90% coverage proposed:", res["intervals_90"]["G3''-F"])


if __name__ == "__main__":
    main()
