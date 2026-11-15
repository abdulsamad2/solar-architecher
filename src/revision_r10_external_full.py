"""Revision R-R10: the full method on SkyGPT's external test days, with 60-min histories rebuilt from the file itself
(protocol revision/protocols/R-R10_external_days_full_method.md). Post hoc, contextual; inference only.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from src.cal2 import replay
from src.clear_sky import ClearSkyPV
from src.e8_features import FEAT
from src.final_eval import SEEDS, blend, day_bootstrap, gate_params
from src.metrics import crps_quantile_interp
from src.phaseA_a2 import H5X, TIMES, load_times
from src.revision_c21 import crps_ens, sample
from src.revision_common import LEVELS, save
from src.revision_m1_method_v2 import ROOT
from src.revision_posthoc_dump import run_member
from src.revision_posthoc_test import sub
from src.revision_r7_forward_v2 import apply_mix, fit_mix
from src.train import GpuSeq

PUBLISHED = {"SkyGPT (m = 100)": 2.81, "PV-MM-Diffusion (m = 30)": 2.63, "SUNSET, SkyGPT paper (m = 10)": 3.31, "smart persistence, SkyGPT paper": 3.67}
US_MIN = 60_000_000


def build_split(device):
    """Per-day record table from the external file. Consecutive samples of a day share 15 of their 16 history records
    (checked), so the records of a day are a sequence: record j of the day = history position j - i of sample i. The
    60-min window takes, for each nominal time t - 60 + 4j min, the record nearest in time within 30 s (the matching
    rule of the benchmark); forecasts without all 16 such records keep the 15-min fallback."""
    t = np.array(load_times(TIMES), dtype="datetime64[us]")
    with h5py.File(H5X, "r") as f:
        g = f["test"]
        img, pvl, y = g["images_log"][...], g["pv_log"][...].astype(float), g["pv_pred"][:, 14].astype(float)
    n = len(t)
    tus = t.astype(np.int64)
    day = t.astype("datetime64[D]")
    hist_rows, h60 = np.empty((n, 16), np.int64), np.full((n, 16), -1, np.int64)
    images, pv, rec_t, base, max_shift = [], [], [], 0, 0.0
    for d in np.unique(day):
        ii = np.flatnonzero(day == d)
        assert (np.diff(ii) == 1).all() and (np.diff(tus[ii]) > 0).all()
        # records of the day in sequence: the 16 records of the first sample, then the new records of every later sample
        # (one per sample; two or three where a sample is missing from the file, found from the shared frames)
        im_d, pv_d, tr = [img[ii[0]]], [pvl[ii[0]]], [tus[ii[0]] - (15 - np.arange(16)) * US_MIN]
        n_d = 16
        hist_rows[ii[0]] = base + np.arange(16)
        for a, b in zip(ii[:-1], ii[1:]):
            s = next((s for s in (1, 2, 3, 4) if np.array_equal(img[b, :16 - s], img[a, s:])), None)
            assert s is not None, "STOP: consecutive samples do not share their frames"
            max_shift = max(max_shift, float(np.abs(pvl[b, :16 - s] - pvl[a, s:]).max()))
            im_d.append(img[b, 16 - s:]); pv_d.append(pvl[b, 16 - s:]); tr.append(tus[b] - (s - 1 - np.arange(s)) * US_MIN)
            n_d += s
            hist_rows[b] = base + n_d - 16 + np.arange(16)
        images.append(np.concatenate(im_d)); pv.append(np.concatenate(pv_d))
        tr = np.concatenate(tr)
        assert (np.diff(tr) > 0).all() and len(tr) == n_d
        rec_t.append(tr)
        nd = n_d - 15
        for i in ii:
            want = tus[i] - (60 - 4 * np.arange(16)) * US_MIN
            j = np.clip(np.searchsorted(tr, want), 1, len(tr) - 1)
            j = np.where(np.abs(tr[j - 1] - want) <= np.abs(tr[j] - want), j - 1, j)
            if (np.abs(tr[j] - want) <= 30_000_000).all():
                h60[i] = base + j
        base += nd + 15
    assert max_shift < 1e-6, f"STOP: consecutive samples do not share their PV history ({max_shift})"
    n_rec = base
    images = np.concatenate(images + [np.zeros((n, 64, 64, 3), np.uint8)])  # one target record per sample (no image)
    pv = np.concatenate(pv + [y])
    rec_t = np.concatenate(rec_t + [tus + 15 * US_MIN]).astype("datetime64[us]")
    target_rows = n_rec + np.arange(n)
    ok = (h60 >= 0).all(1)
    ds = SimpleNamespace(sample_ids=np.arange(n), hist_rows=hist_rows, target_rows=target_rows, images=torch.from_numpy(images),
                         pv=torch.from_numpy(pv.astype(np.float32)), rec_t=rec_t, origin_time=t, target_time=t + np.timedelta64(15, "m"), day=day)
    split = SimpleNamespace(name="external", ds=ds, device=device, table=None, rec_t=rec_t, pv=pv.astype(float), h60_ok=ok,
                            h60_rows=np.where(ok[:, None], h60, hist_rows), data=GpuSeq(ds, device))
    assert (split.h60_rows[:, -1] == hist_rows[:, -1]).all(), "STOP: window origin mismatch"
    info = {"n": int(n), "records": int(n_rec), "share_with_60min_history": float(ok.mean()),
            "n_with_60min_history_by_day": {str(d): int(ok[day == d].sum()) for d in np.unique(day)}}
    return split, t, y, info


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    split, t, y, info = build_split(torch.device("cuda"))
    print("external table:", info, flush=True)
    h60, n = split.h60_ok, len(y)
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    tau_all = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    common = np.load(FEAT / "e8_windows_v1.npz")
    common = common["sample_ids"][common["eligible_common"]]
    cal = lambda e, s, k: dict(np.load(ROOT[e] / f"seed{s}" / f"fold_{k}" / "preds_cal.npz"))
    pv_now = split.pv[split.ds.hist_rows[:, -1]]
    names = ("frozen rule, fallback only", "frozen rule, routed", "method v2, fallback only", "method v2, routed", "G4", "calm expert", "direct expert")
    acc = {m: [] for m in names}
    for s in SEEDS:
        for k in range(5):
            fold, tau = {**folds_doc["folds"][k]}, tau_all[str(k)]["training_labels_fit_days"]
            rd, rc, ra = (run_member(split, e, s, fold, tau) for e in ("E4-B-fit", "E6-S", "A2"))
            md, mr, ma = ({"pred_q": r["pred_q"], "pclear_target_kw": r["pclear_target_kw"], "pclear_now_kw": r["pclear_now_kw"], "pv_now_kw": pv_now,
                           "quantile_levels": LEVELS} for r in (rd, rc, ra))
            g3 = blend(md, mr, gate_params("G3", s, k), tau)
            acc["frozen rule, fallback only"].append(g3)
            routed = g3.copy()
            routed[h60] = blend(sub(ma, h60), sub(mr, h60), gate_params("G3''", s, k), tau)
            acc["frozen rule, routed"].append(routed)
            acc["G4"].append(blend(md, mr, gate_params("G4", s, k), tau))
            acc["calm expert"].append(mr["pred_q"]); acc["direct expert"].append(md["pred_q"])
            cd, cc, cr = cal("D", s, k), cal("C", s, k), cal("R", s, k)
            p_dc = fit_mix({"D": cd, "C": cc}, tau, ("D", "C"))
            cc2, cr2 = sub(cc, np.isin(cc["sample_ids"], common)), sub(cr, np.isin(cr["sample_ids"], common))
            p_rc = fit_mix({"R": cr2, "C": cc2}, tau, ("R", "C"))
            v = apply_mix(p_dc, {"D": md, "C": mr}, tau, ("D", "C"))
            acc["method v2, fallback only"].append(v)
            v2 = v.copy()
            v2[h60] = apply_mix(p_rc, {"R": sub(ma, h60), "C": sub(mr, h60)}, tau, ("R", "C"))
            acc["method v2, routed"].append(v2)
            print(f"member seed {s} fold {k} done", flush=True)
    q = {m: np.mean(v, 0) for m, v in acc.items()}
    a2 = np.load("experiments/phaseA/a2_predictions.npz")
    dev_ = float(np.abs(q["frozen rule, fallback only"] - a2["q|G3''-F"]).max())
    print(f"reproduction of the Phase A2 forecasts: max |diff| {dev_:.4f} kW, mean |diff| {float(np.abs(q['frozen rule, fallback only'] - a2[chr(113) + '|G3' + chr(39) * 2 + '-F']).mean()):.5f} kW", flush=True)
    assert dev_ < 0.25, f"STOP: Phase A2 forecasts not reproduced (max |diff| {dev_} kW)"
    day = t.astype("datetime64[D]").astype(str)
    zero = np.zeros(n, bool)
    rdj = json.loads(Path("reports/test_ramp_definition_frozen.json").read_text())
    anchor = ClearSkyPV(rdj["anchor_coef"])
    ramp = np.abs(y / np.maximum(anchor(t + np.timedelta64(15, "m")), 1.0) - pv_now / np.maximum(anchor(t), 1.0)) >= rdj["tau_test"]
    rng = np.random.default_rng(20261113)
    res = {"protocol": "revision/protocols/R-R10_external_days_full_method.md", **info, "phaseA2_reproduction_max_abs_kw": dev_, "days": sorted(set(day)),
           "n_ramp": int(ramp.sum()), "published": PUBLISHED, "scores": {}, "paired": {}}
    crps = {m: crps_quantile_interp(v, y, LEVELS) for m, v in q.items()}
    for m, v in q.items():
        r = {"crps_exact": float(crps[m].mean()), "ramp": float(crps[m][ramp].mean()), "non_ramp": float(crps[m][~ramp].mean()),
             "with_60min_history": float(crps[m][h60].mean()), "without": float(crps[m][~h60].mean()) if (~h60).any() else None}
        for mm in (10, 30, 100):
            r[f"standard_m{mm}"] = float(np.mean([crps_ens(sample(v, mm, rng), y, False).mean() for _ in range(20)]))
        r["fair"] = float(np.mean([crps_ens(sample(v, 100, rng), y, True).mean() for _ in range(20)]))
        res["scores"][m] = r
        print(f"{m:28s} exact {r['crps_exact']:.3f} | m=10 {r['standard_m10']:.3f}  m=30 {r['standard_m30']:.3f}  m=100 {r['standard_m100']:.3f}  fair {r['fair']:.3f} | "
              f"ramp {r['ramp']:.3f} non-ramp {r['non_ramp']:.3f} | 60-min eligible {r['with_60min_history']:.3f}", flush=True)
    f = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"
    for a, b in (("method v2, routed", "method v2, fallback only"), ("frozen rule, routed", "frozen rule, fallback only"), ("method v2, routed", "frozen rule, routed"),
                 ("method v2, routed", "G4"), ("method v2, routed", "calm expert"), ("method v2, routed", "direct expert")):
        res["paired"][f"{a} - {b}"] = day_bootstrap(crps[a] - crps[b], day, zero)
        print(f"{a} - {b}: {f(res['paired'][f'{a} - {b}'])} (five days; descriptive)")
    # 90% coverage: raw, continuous rolling and ACI started from development scores of the frozen forecaster
    cald = np.load("experiments/final/dev_oof_calibration_v1.npz")
    W = json.loads(Path("experiments/final/dev_oof_calibration_v1.json").read_text())["W"]["G3''-F"]
    init = cald["G3''-F|90"][np.argsort(cald["origin_us"].astype(np.int64), kind="stable")][-W:]
    org = t.astype(np.int64)
    tgt = org + 15 * US_MIN
    res["coverage90"] = {}
    for m in ("method v2, routed", "frozen rule, routed", "frozen rule, fallback only"):
        lo, hi = q[m][:, 0], q[m][:, -1]
        out = {"raw": (lo, hi)}
        for meth, par in (("continuous rolling", ("rolling", W)), ("ACI", ("aci", 0.005))):
            rep = replay(org, tgt, lo, hi, y, 0.1, par[0], par[1], init)
            out[meth] = (rep["lo"], rep["hi"])
        res["coverage90"][m] = {kk: {"coverage": float(np.mean((y >= l) & (y <= h))), "ramp_coverage": float(np.mean((y[ramp] >= l[ramp]) & (y[ramp] <= h[ramp]))),
                                     "width_kw": float(np.mean(h - l))} for kk, (l, h) in out.items()}
        print(f"90% interval, {m}: " + " | ".join(f"{kk} cov {v['coverage']:.3f} (ramps {v['ramp_coverage']:.3f}) width {v['width_kw']:.2f}" for kk, v in res["coverage90"][m].items()))
    print("published (standard estimator):", PUBLISHED)
    print("saved", save("r10_external_days_full_method.json", res))


if __name__ == "__main__":
    main()
