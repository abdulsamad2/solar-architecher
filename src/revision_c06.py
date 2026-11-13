"""Revision C06: issuance-time input availability audit and past-only PV sensitivity (post hoc, test).

Protocol: revision/protocols/R-C06_input_timing.md.
Facts used: the forecast origin t is itself the latest matched record (history column 0 == origin), so no image is
later than t; earlier frames are matched within +-30 s of t - j min. SKIPP'D PV values are averages over the
following minute, so PV(t) is only available at about t + 60 s. Sensitivity: every PV/K input (history, K(t) of the
residual expert, gate features, persistence) is taken from the record ~60 s earlier; images are unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.baselines import official_persistence
from src.final_eval import FEAT, H5, MAN, SEEDS, Split, blend, gate_params, record_table_unsorted
from src.metrics import crps_quantile_interp
from src.revision_common import LEVELS, day_bootstrap, frozen_test_predictions, save, test_context
from src.revision_posthoc_dump import run_member
from src.skippd import _nearest

MIN = 60_000_000


def audit() -> dict:
    m = np.load(MAN)
    us = m["times_us"]
    out = {}
    for name, sel in (("dev", ~m["is_test"]), ("test", m["is_test"])):
        o, h, t = m["origin"][sel], m["history"][sel], m["target"][sel]
        assert (h[:, 0] == o).all(), "origin is not the latest history record"
        frame_off = (us[h] - (us[o][:, None] - np.arange(h.shape[1])[None, :] * MIN)) / 1e6  # seconds
        tgt_off = (us[t] - (us[o] + 15 * MIN)) / 1e6
        latest_after_t = (us[h] > us[o][:, None]).any(1)
        lead_pv_avail = (us[t] - (us[o] + MIN)) / 60e6  # minutes from PV(t) availability to target record
        q = lambda x: {p: float(np.percentile(x, p)) for p in (0, 5, 50, 95, 100)}
        out[name] = {"n": int(sel.sum()), "any_frame_after_origin": int(latest_after_t.sum()),
                     "frame_offset_s_(j>=1)": q(frame_off[:, 1:].ravel()), "target_offset_s": q(tgt_off),
                     "lead_from_origin_min": q((us[t] - us[o]) / 60e6), "lead_from_pv_availability_min": q(lead_pv_avail)}
    return out


def prev_records(rec_t) -> np.ndarray:
    us = np.asarray(rec_t, "datetime64[us]").astype(np.int64)
    order = np.argsort(us, kind="stable")
    s = us[order]
    k = _nearest(s, s - MIN, MIN // 2)
    k = np.where((k >= 0) & (s[np.maximum(k, 0)] < s), k, -1)
    same_day = (s[np.maximum(k, 0)] // 86_400_000_000) == (s // 86_400_000_000)
    k = np.where(same_day, k, -1)
    prev = np.full(len(us), -1)
    prev[order] = np.where(k >= 0, order[np.maximum(k, 0)], -1)
    return prev


def shifted() -> dict:
    import h5py
    with h5py.File(H5, "r") as f:
        gray = np.round(f["test/images_log"][...].astype(np.float32).mean(-1)).astype(np.uint8)
    m = np.load(MAN)
    sel = m["hdf5_group"] == 1
    t_test = m["times_us"][sel][np.argsort(m["hdf5_row"][sel])].astype("datetime64[us]")
    split = Split("test", torch.device("cuda"), record_table_unsorted(t_test, gray, FEAT / "gray_u8_test.npy"))
    prev = prev_records(split.rec_t)
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    tau_f = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    h60 = split.h60_ok
    lv = LEVELS
    acc = {k: [] for k in ("G3''-F", "E4-B-fit", "E6-S", "G4")}
    pv_now_s = None
    for s in SEEDS:
        for k in range(5):
            fold = {**folds_doc["folds"][k]}
            tau = tau_f[str(k)]["training_labels_fit_days"]
            mem = {n: run_member(split, n, s, fold, tau, prev) for n in ("E4-B-fit", "E6-S", "A2")}
            for n in mem:
                mem[n]["quantile_levels"] = lv
            pv_now_s = mem["E6-S"]["pv_now_kw"]
            sub = lambda d, mm: {kk: (v[mm] if np.ndim(v) and len(v) == len(mm) else v) for kk, v in d.items()}
            q = blend(mem["E4-B-fit"], mem["E6-S"], gate_params("G3", s, k), tau)
            ok = h60 & np.isfinite(mem["A2"]["pred_q"][:, 0]) & np.isfinite(mem["E6-S"]["pred_q"][:, 0])
            q[ok] = blend(sub(mem["A2"], ok), sub(mem["E6-S"], ok), gate_params("G3''", s, k), tau)
            acc["G3''-F"].append(q)
            acc["G4"].append(blend(mem["E4-B-fit"], mem["E6-S"], gate_params("G4", s, k), tau))
            acc["E4-B-fit"].append(mem["E4-B-fit"]["pred_q"])
            acc["E6-S"].append(mem["E6-S"]["pred_q"])
            torch.cuda.empty_cache()
    q_s = {n: np.mean(v, 0) for n, v in acc.items()}
    c = test_context()
    assert np.array_equal(c["sample_ids"], split.ds.sample_ids)
    q_o = frozen_test_predictions(c["sample_ids"])
    # a sample is valid iff every PV record it uses (15-min history; 60-min window where routed to G3'') has an earlier record
    valid = (prev[split.ds.hist_rows] >= 0).all(1)
    valid &= ~h60 | (prev[split.h60_rows] >= 0).all(1)
    assert all(np.isfinite(v[valid]).all() for v in q_s.values())
    y, day, sunny, masks = c["y"], c["day"], c["sunny"], c["masks"]
    pers_o = official_persistence(c["ds"].origin_time.astype(object), c["pv"][c["ds"].hist_rows[:, -1]])
    pers_s = official_persistence(c["ds"].origin_time.astype(object), pv_now_s)
    res = {"n_valid": int(valid.sum()), "n_excluded_no_earlier_pv": int((~valid).sum()), "scores": {}, "shift_minus_original": {}}
    crps = {}
    for n in q_s:
        crps[(n, "original")] = crps_quantile_interp(q_o[n][valid], y[valid], LEVELS)
        crps[(n, "shifted")] = crps_quantile_interp(q_s[n][valid], y[valid], LEVELS)
    crps[("persistence", "original")] = np.abs(pers_o - y)[valid]
    crps[("persistence", "shifted")] = np.abs(pers_s - y)[valid]
    mv = {g: v[valid] for g, v in masks.items()}
    dv, sv = day[valid], sunny[valid]
    for (n, kind), cc in crps.items():
        res["scores"].setdefault(n, {})[kind] = {g: float(cc[v].mean()) for g, v in mv.items() if v.any()}
    for n in list(q_s) + ["persistence"]:
        d = crps[(n, "shifted")] - crps[(n, "original")]
        res["shift_minus_original"][n] = {g: day_bootstrap(d[v], dv[v], sv[v]) for g, v in mv.items() if g in ("overall", "ramp", "non-ramp")}
    for kind in ("original", "shifted"):
        d = crps[("G3''-F", kind)] - crps[("E4-B-fit", kind)]
        res[f"H1 G3''-F - E4-B-fit ({kind})"] = day_bootstrap(d, dv, sv)
        d = crps[("G3''-F", kind)] - crps[("persistence", kind)]
        res[f"G3''-F - persistence ({kind})"] = day_bootstrap(d, dv, sv)
    return res


def main():
    a = audit()
    save("c06_timing_audit.json", a)
    print(json.dumps(a["test"], indent=0)[:800])
    r = shifted()
    p = save("c06_shifted_inputs.json", {"label": "POST HOC (test)", **r})
    print("saved", p)
    print({k: v for k, v in r.items() if k.startswith("H1") or "persistence (" in k}, r["n_excluded_no_earlier_pv"])
    print({n: round(v["shifted"]["overall"] - v["original"]["overall"], 4) for n, v in r["scores"].items()})


if __name__ == "__main__":
    main()
