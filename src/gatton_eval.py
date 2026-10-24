"""Phase E evaluation on Gatton (reports/phaseE_gatton_prereg.md sections 4-6, amendments E-1..E-4).

  python -m src.gatton_eval prepare          development only: gate fits (F1, F0), development forecasts for every arm,
                                            calibration window W and init scores, ramp threshold tau_G, inventory
  python -m src.gatton_eval run_test TOKEN   ONE-SHOT sealed test (guarded; requires reports/gatton_test_approval.txt)

Arms: Z0 zero-shot (frozen SKIPP'D members + gates), Z1 (Z0 with vertically flipped images; descriptive),
      F1 fine-tuned from SKIPP'D (Gatton gates refitted on calibration days), F0 trained from scratch (secondary).
Models per arm: G3''-F (G3'' where a 60-min history exists, else G3), E4-B-fit, E6-S; plus smart persistence.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from src.cal2 import ALPHA, conformity, interval_score, replay, select_param
from src.calibrate import base_interval
from src.clear_sky import clear_sky_index
from src.e8_audit import history_rows
from src.final_eval import day_bootstrap
from src.folds import calibration_split
from src.gate7b import apply_gate, raw_features, select_and_fit
from src.gatton_clearsky import GattonClearSky, load_params
from src.metrics import crps_quantile_interp
from src.skippd import SkippdForecastDataset
from src.train import GpuSeq, build_model, predict_raw

H5 = "D:/solar_data/gatton/processed/gatton_skippd.hdf5"
MAN = "data/manifests/gatton_forecast_v1.npz"
FOLDS = json.loads(Path("data/manifests/gatton_folds_v1.json").read_text())
OUT = Path("experiments/E_gatton")
SEEDS, KS = (42, 123, 456), range(5)
LV = np.array([.05, .1, .2, .3, .5, .7, .8, .9, .95])
C_MIN = load_params()["c_min_kw"]
SKIPPD = {"E4-B-fit": Path("experiments/E4_calib/v1/csi"), "E6-S": Path("experiments/E6/v1/E6-S/csi"),
          "A2": Path("experiments/E9/v1/A2/csi")}
SKIPPD_TAU = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
TEST_START = np.datetime64("2020-04-16")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()


class GattonSplit:
    """Split object for member inference (records of one HDF5 group, pv may be replaced by sealed test PV)."""

    def __init__(self, name: str, device, pv_override: np.ndarray | None = None, flip: bool = False):
        self.ds = SkippdForecastDataset(H5, MAN, "dev" if name in ("dev", "trainval") else "test", preload=True)
        if flip:
            self.ds.images = torch.flip(self.ds.images, dims=[1])
        if pv_override is not None:
            self.ds.pv = torch.as_tensor(pv_override, dtype=torch.float32)
        m = np.load(MAN)
        g = 0 if name == "trainval" or name == "dev" else 1
        sel = m["hdf5_group"] == g
        self.rec_t = m["times_us"][sel][np.argsort(m["hdf5_row"][sel])].astype("datetime64[us]")
        self.pv = self.ds.pv.numpy().astype(float)
        origin_tl = m["origin"][m["is_test"] == (g == 1)]
        tl = history_rows(m["times_us"], origin_tl, 4)
        self.h60_ok = (tl >= 0).all(1)
        rows = np.where(tl >= 0, m["hdf5_row"][np.maximum(tl, 0)], -1)[:, ::-1]
        self.h60_rows = np.where(self.h60_ok[:, None], rows, self.ds.hist_rows)
        self.data = GpuSeq(self.ds, device)
        self.device = device
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


_ANCHORS = {}


def dev_records():
    m = np.load(MAN)
    sel = m["hdf5_group"] == 0
    t = m["times_us"][sel][np.argsort(m["hdf5_row"][sel])].astype("datetime64[us]")
    import h5py
    with h5py.File(H5, "r") as f:
        pv = f["trainval/pv_log"][...].astype(float)
    return t, pv


def anchor(fold_k: int | None) -> GattonClearSky:
    """Fold anchor (fit days of Gatton fold k) for Gatton-trained members; the development fit for zero-shot."""
    if fold_k not in _ANCHORS:
        if fold_k is None:
            _ANCHORS[None] = GattonClearSky.from_dev_fit()
        else:
            t, pv = dev_records()
            fit = calibration_split(FOLDS["folds"][fold_k]["train_days"])[0]
            sel = np.isin(t.astype("datetime64[D]").astype(str), fit)
            _ANCHORS[fold_k] = GattonClearSky().fit(t[sel], pv[sel])
    return _ANCHORS[fold_k]


def member_predict(split: GattonSplit, run_root: Path, seed: int, k: int, idx, window: bool, anc: GattonClearSky) -> dict:
    cfg = json.loads((run_root / f"seed{seed}" / "config.json").read_text())
    pc = anc(split.rec_t)
    k_rec = clear_sky_index(split.pv, pc, C_MIN)
    d = split.data
    d.set_history_feature(k_rec)
    d.residual = False
    d.k_now = torch.tensor(k_rec[split.ds.hist_rows[:, -1]], dtype=torch.float32, device=split.device)
    if window:
        assert split.h60_ok[idx].all()
        d.set_window(split.h60_rows)
        d.set_extra(np.zeros((len(split.pv), 11), np.float32))
    else:
        d.set_window(split.ds.hist_rows)
        d.set_extra(None)
    net = build_model(cfg).to(split.device)
    net.encoder.cnn_bf16 = cfg["cnn_bf16"]
    net.load_state_dict(torch.load(run_root / f"seed{seed}" / f"fold_{k}" / "best.pt", map_location=split.device))
    raw = predict_raw(net, d, torch.as_tensor(idx, device=split.device)).cpu().numpy().astype(float)
    pc_t = pc[split.ds.target_rows[idx]]
    return {"pred_q": raw * np.maximum(pc_t, C_MIN)[:, None], "pclear_target_kw": pc_t,
            "pclear_now_kw": pc[split.ds.hist_rows[idx, -1]], "pv_now_kw": split.pv[split.ds.hist_rows[idx, -1]],
            "quantile_levels": LV}


def blend(direct, resid, params, tau):
    w = apply_gate(params, raw_features(direct, resid, tau))
    return w[:, None] * direct["pred_q"] + (1 - w[:, None]) * resid["pred_q"]


def arm_members(arm: str, s: int, k: int):
    """(run roots, gate params G3/G3'', tau, anchor) for one member of an arm."""
    if arm in ("Z0", "Z1"):
        g3 = json.loads((Path("experiments/E7b_gate/v1") / f"seed{s}" / f"fold_{k}" / "params.json").read_text())["G3"]
        g3pp = json.loads((Path("experiments/E9/v1/G3pp_params") / f"seed{s}" / f"fold_{k}" / "params.json").read_text())
        return SKIPPD, g3, g3pp, SKIPPD_TAU[str(k)]["training_labels_fit_days"], anchor(None)
    roots = {e: OUT / arm / e / "csi" for e in ("E4-B-fit", "E6-S", "A2")}
    gp = json.loads((OUT / arm / "gates" / f"seed{s}" / f"fold_{k}" / "params.json").read_text())
    return roots, gp["G3"], gp["G3''"], gp["tau"], anchor(k)


def predict_arm(split: GattonSplit, arm: str, idx: np.ndarray, members) -> dict:
    """Ensemble over the given (seed, fold) members: G3''-F, E4-B-fit, E6-S quantiles in kW."""
    n = len(idx)
    acc = {m: np.zeros((n, 9)) for m in ("G3''-F", "E4-B-fit", "E6-S")}
    ok = split.h60_ok[idx]
    for s, k in members:
        roots, g3, g3pp, tau, anc = arm_members(arm, s, k)
        e4 = member_predict(split, roots["E4-B-fit"], s, k, idx, False, anc)
        e6 = member_predict(split, roots["E6-S"], s, k, idx, False, anc)
        q = blend(e4, e6, g3, tau)
        if ok.any():
            a2 = member_predict(split, roots["A2"], s, k, idx[ok], True, anc)
            sub = {kk: (v[ok] if np.ndim(v) and len(v) == n else v) for kk, v in e6.items()}
            q[ok] = blend(a2, sub, g3pp, tau)
        acc["G3''-F"] += q
        acc["E4-B-fit"] += e4["pred_q"]
        acc["E6-S"] += e6["pred_q"]
        torch.cuda.empty_cache()
    return {m: a / len(members) for m, a in acc.items()}


def fit_gates(arm: str):
    for s in SEEDS:
        for k in KS:
            ld = lambda e, sp: dict(np.load(OUT / arm / e / "csi" / f"seed{s}" / f"fold_{k}" / f"preds_{sp}.npz"))
            e6c, e4c, a2c = ld("E6-S", "cal"), ld("E4-B-fit", "cal"), ld("A2", "cal")
            tau = float(ld("E6-S", "val")["tau"])
            pos = {i: j for j, i in enumerate(e6c["sample_ids"])}
            sub = lambda d, ids: {kk: (v[[pos[i] for i in ids]] if np.ndim(v) and len(v) == len(e6c["sample_ids"]) else v)
                                  for kk, v in d.items()}
            order = np.argsort(e6c["origin_time_us"], kind="stable")
            g3 = select_and_fit(raw_features(e4c, e6c, tau), ("g", "d"), e4c["pred_q"], e6c["pred_q"], e6c["target"], LV, order)
            r6 = sub(e6c, a2c["sample_ids"])
            o2 = np.argsort(r6["origin_time_us"], kind="stable")
            g3pp = select_and_fit(raw_features(a2c, r6, tau), ("g", "d"), a2c["pred_q"], r6["pred_q"], r6["target"], LV, o2)
            p = OUT / arm / "gates" / f"seed{s}" / f"fold_{k}"
            p.mkdir(parents=True, exist_ok=True)
            (p / "params.json").write_text(json.dumps({"G3": g3, "G3''": g3pp, "tau": tau}, indent=1))
    print(f"{arm}: gates fitted on Gatton calibration days", flush=True)


def prepare():
    dev = GattonSplit("trainval", torch.device("cuda"))
    n = len(dev.ds.sample_ids)
    day = dev.ds.day.astype(str)
    va = {k: np.flatnonzero(np.isin(day, FOLDS["folds"][k]["val_days"])) for k in KS}
    y = dev.pv[dev.ds.target_rows]
    out = {"n_dev": n}
    streams = {}
    for arm in ("F1", "F0"):
        fit_gates(arm)
        q = {m: np.zeros((n, 9)) for m in ("G3''-F", "E4-B-fit", "E6-S")}
        for k in KS:  # out-of-fold: member (s, k) predicts only fold k's validation days
            r = predict_arm(dev, arm, va[k], [(s, k) for s in SEEDS])
            for m in q:
                q[m][va[k]] = r[m]
        streams[arm] = q
    for arm in ("Z0",):  # zero-shot: all 15 SKIPP'D members on every development sample (never trained on Gatton)
        streams[arm] = predict_arm(dev, arm, np.arange(n), [(s, k) for s in SEEDS for k in KS])
    anc = anchor(None)
    pc = anc(dev.rec_t)
    k_now = dev.pv[dev.ds.hist_rows[:, -1]] / np.maximum(pc[dev.ds.hist_rows[:, -1]], C_MIN)
    k_tgt = y / np.maximum(pc[dev.ds.target_rows], C_MIN)
    tau_g = float(np.quantile(np.abs(k_tgt - k_now), 0.9))  # prereg 3.8: development samples, frozen before testing
    pers = dev.pv[dev.ds.hist_rows[:, -1]] * np.maximum(pc[dev.ds.target_rows], C_MIN) / np.maximum(pc[dev.ds.hist_rows[:, -1]], C_MIN)
    ramp = np.abs(k_tgt - k_now) >= tau_g
    out["tau_G"] = tau_g
    out["dev_crps"] = {arm: {m: {"overall": float(crps_quantile_interp(q, y, LV).mean()),
                                 "ramp": float(crps_quantile_interp(q[ramp], y[ramp], LV).mean())} for m, q in qs.items()}
                       for arm, qs in streams.items()}
    out["dev_crps"]["smart persistence"] = {"overall": float(np.abs(pers - y).mean()), "ramp": float(np.abs(pers - y)[ramp].mean())}
    # calibration: continuous causal rolling conformal (Phase C "M2"); W chosen on the development stream; init = last W
    org = dev.ds.origin_time.astype("datetime64[us]").astype(np.int64)
    tgt = dev.ds.target_time.astype("datetime64[us]").astype(np.int64)
    cal = {"origin_us": org}
    W = {}
    for arm, qs in streams.items():
        for m, q in qs.items():
            bounds = {nm: base_interval(q, LV, nm) for nm in ALPHA}
            w, _ = select_param("rolling", org, tgt, bounds, y)
            W[f"{arm}|{m}"] = int(w)
            for nm in ALPHA:
                cal[f"{arm}|{m}|{nm}"] = conformity(*bounds[nm], y)
    np.savez_compressed(OUT / "dev_calibration_v1.npz", **cal)
    (OUT / "dev_calibration_v1.json").write_text(json.dumps({"W": W, "tau_G": tau_g, "c_min_kw": C_MIN}, indent=1))
    np.savez_compressed(OUT / "dev_streams_v1.npz", sample_ids=dev.ds.sample_ids, y=y,
                        **{f"{a}|{m}": q for a, qs in streams.items() for m, q in qs.items()})
    out["W"] = W
    files = [MAN, "data/manifests/gatton_folds_v1.json", "reports/phaseE_gatton_clearsky.json", OUT / "dev_calibration_v1.npz",
             OUT / "dev_calibration_v1.json", H5]
    for arm in ("F1", "F0"):
        for e in ("E4-B-fit", "E6-S", "A2"):
            for s in SEEDS:
                files.append(OUT / arm / e / "csi" / f"seed{s}" / "config.json")
                files += [OUT / arm / e / "csi" / f"seed{s}" / f"fold_{k}" / "best.pt" for k in KS]
        files += [OUT / arm / "gates" / f"seed{s}" / f"fold_{k}" / "params.json" for s in SEEDS for k in KS]
    inv = {str(p).replace("\\", "/"): sha(p) for p in files}
    Path("reports/phaseE_gatton_inventory.json").write_text(json.dumps(inv, indent=1))
    out["inventory_files"] = len(inv)
    Path("reports/phaseE_gatton_dev_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "W"}, indent=1))


CSV = r"D:/solar_data/gatton/bag/data/pv_2020.csv"
TEST_END = np.datetime64("2020-05-07")                     # exclusive
STANFORD_PREFIXES = ("experiments/E4_calib/v1/csi", "experiments/E6/v1/E6-S/csi", "experiments/E9/v1/A2/csi",
                     "experiments/E7b_gate/v1", "experiments/E9/v1/G3pp_params")


def verify_all_artifacts() -> dict:
    """Fail closed: every Gatton inventory file (144) and every Stanford artifact used by Z0/Z1 (100) must exist and match."""
    gat = json.loads(Path("reports/phaseE_gatton_inventory.json").read_text())
    inv5 = json.loads(Path("reports/checkpoint_inventory.json").read_text())["files"]
    stan = {p: h for p, h in inv5.items() if p.startswith(STANFORD_PREFIXES) or p == "reports/phase6_frozen_thresholds.json"}
    assert len(gat) == 144 and len(stan) == 100, f"STOP: unexpected inventory sizes {len(gat)} / {len(stan)}"
    for p, h in {**gat, **stan}.items():
        assert Path(p).exists(), f"STOP: missing artifact {p}"
        assert sha(p) == h, f"STOP: hash mismatch {p}"
    return {"gatton": len(gat), "stanford": len(stan)}


def read_pv_window(start: np.datetime64, end: np.datetime64) -> dict:
    """PV of CSV rows with start <= time < end ONLY (other rows are skipped while reading, never loaded).
    Sign convention E-3; missing -> NaN (E-1); duplicate timestamps -> first occurrence."""
    import pandas as pd
    t_all = pd.to_datetime(pd.read_csv(CSV, usecols=["DateTime"])["DateTime"], format="%d/%m/%Y %I:%M:%S %p")
    rows = set((np.flatnonzero((t_all >= str(start)) & (t_all < str(end))) + 1).tolist()) | {0}
    d = pd.read_csv(CSV, skiprows=lambda i: i not in rows, na_values=["(null)"])
    tt = pd.to_datetime(d["DateTime"], format="%d/%m/%Y %I:%M:%S %p").to_numpy().astype("datetime64[s]")
    assert len(tt) == 0 or (tt.min() >= start and tt.max() < end), "STOP: a row outside the requested window was loaded"
    gen = -pd.to_numeric(d["Power(kW)"], errors="coerce").to_numpy(float)
    gen = np.where(np.isnan(gen), np.nan, np.maximum(gen, 0.0))
    first = {}
    for ti, v in zip(tt.astype(np.int64), gen):
        first.setdefault(int(ti), v)
    return first


def run_test(token: str):
    """OFFICIAL one-shot sealed test: 21 days 2020-04-16 .. 2020-05-06 (requires a recorded approval token)."""
    assert Path("reports/gatton_test_approval.txt").read_text().strip() == token, "STOP: no recorded explicit approval"
    res_path = Path("reports/phaseE_gatton_test_results.json")
    assert not res_path.exists(), "STOP: test results already exist (one-shot)"
    assert not subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip(), "STOP: dirty tree"
    verify_all_artifacts()
    m = np.load(MAN)
    test_days = sorted({str(d) for d in m["times_us"][m["origin"][m["is_test"]]].astype("datetime64[us]").astype("datetime64[D]")})
    return evaluate_window("test", test_days, TEST_START, TEST_END, res_path, OUT / "test_predictions_v1.npz")


def rehearse():
    """DEVELOPMENT-ONLY rehearsal of the identical evaluation body on the last 21 development days (2020-03-26 .. 04-15).
    Structurally unable to read sealed rows: the window end is asserted <= the sealed start before any read."""
    dev_days = sorted({d for fd in FOLDS["folds"] for d in fd["val_days"]})
    window = dev_days[-21:]
    start, end = np.datetime64(window[0]), np.datetime64(window[-1]) + np.timedelta64(1, "D")
    assert end <= TEST_START, "STOP: rehearsal window would touch sealed days"
    verify_all_artifacts()
    return evaluate_window("dev", window, start, end, Path("reports/phaseE_gatton_rehearsal_dev.json"),
                           OUT / "rehearsal_predictions_dev.npz")


def evaluate_window(group: str, days: list, start, end, res_path: Path, pred_path: Path) -> dict:
    """Evaluation body shared by the official test and the development rehearsal (code moved from the dd39740 run_test;
    the only parameters are the HDF5 group, the window of days, the CSV row window and the output paths; calibration
    initial scores are restricted to origins before the window start, which for the official window equals all
    development scores, i.e. identical to dd39740)."""
    first = read_pv_window(start, end)
    m = np.load(MAN)
    sel = m["hdf5_group"] == (1 if group == "test" else 0)
    rec_t = m["times_us"][sel][np.argsort(m["hdf5_row"][sel])].astype("datetime64[us]").astype("datetime64[s]").astype(np.int64)
    pv_test = np.array([first.get(int(s), np.nan) for s in rec_t])
    test = GattonSplit(group, torch.device("cuda"), pv_override=np.nan_to_num(pv_test, nan=0.0))
    pv_nan = np.isnan(pv_test)
    valid = ~(pv_nan[test.ds.hist_rows].any(1) | pv_nan[test.ds.target_rows])
    if test.h60_ok.any():
        valid &= ~np.where(test.h60_ok[:, None], pv_nan[test.h60_rows], False).any(1)  # E-1 also for 60-min windows
    in_window = np.isin(test.ds.day.astype(str), days)
    valid_w = valid & in_window
    idx = np.flatnonzero(valid_w)
    fc = {"F1": predict_arm(test, "F1", idx, [(s, k) for s in SEEDS for k in KS]),
          "F0": predict_arm(test, "F0", idx, [(s, k) for s in SEEDS for k in KS]),
          "Z0": predict_arm(test, "Z0", idx, [(s, k) for s in SEEDS for k in KS])}
    flip = GattonSplit(group, torch.device("cuda"), pv_override=np.nan_to_num(pv_test, nan=0.0), flip=True)
    fc["Z1"] = predict_arm(flip, "Z1", idx, [(s, k) for s in SEEDS for k in KS])
    y = pv_test[test.ds.target_rows[idx]]
    day = test.ds.day.astype(str)[idx]
    calj = json.loads((OUT / "dev_calibration_v1.json").read_text())
    tau_g = calj["tau_G"]
    anc = anchor(None)
    pc = anc(test.rec_t)
    hr, tr = test.ds.hist_rows[idx, -1], test.ds.target_rows[idx]
    k_now, k_tgt = pv_test[hr] / np.maximum(pc[hr], C_MIN), y / np.maximum(pc[tr], C_MIN)
    pers = pv_test[hr] * np.maximum(pc[tr], C_MIN) / np.maximum(pc[hr], C_MIN)
    ramp = np.abs(k_tgt - k_now) >= tau_g
    h60 = test.h60_ok[idx]
    # test-day sunny labels (reporting only; development rule from amendment E-2)
    lab = {}
    for dd in np.unique(day):
        s_ = (rec_t.astype("datetime64[s]").astype("datetime64[D]").astype(str) == dd) & ~pv_nan
        c = np.corrcoef(pv_test[s_], pc[s_])[0, 1] if s_.sum() > 2 else np.nan
        lab[dd] = bool(c > 0.995 and np.abs(np.diff(pv_test[s_])).mean() < 0.15 * 3275 / 30.1)
    sunny = np.array([lab[dd] for dd in day])
    strata = {"overall": np.ones(len(y), bool), "sunny": sunny, "cloudy": ~sunny, "ramp": ramp, "non-ramp": ~ramp,
              "sunrise ramp": ramp & ~h60, "rising ramp": ramp & (k_tgt > k_now), "falling ramp": ramp & (k_tgt < k_now)}
    res = {"n": int(len(idx)), "n_excluded_missing_pv": int((~valid & in_window).sum()), "n_days": int(len(np.unique(day))),
           "n_sunny_days": int(sum(lab.values())), "tau_G": tau_g, "scores": {}, "intervals": {}, "hypotheses": {}, "per_day": {}}
    crps = {}
    for arm, qs in fc.items():
        for mdl, q in qs.items():
            c = crps_quantile_interp(q, y, LV)
            crps[f"{arm}|{mdl}"] = c
            med = q[:, 4]
            res["scores"][f"{arm}|{mdl}"] = {g: {"n": int(v.sum()), "crps": float(c[v].mean()),
                                                 "rmse": float(np.sqrt(np.mean((med[v] - y[v]) ** 2))),
                                                 "mae": float(np.abs(med[v] - y[v]).mean())} for g, v in strata.items() if v.any()}
    crps["persistence"] = np.abs(pers - y)
    res["scores"]["smart persistence"] = {g: {"n": int(v.sum()), "crps": float(crps["persistence"][v].mean()),
                                              "rmse": float(np.sqrt(np.mean((pers[v] - y[v]) ** 2)))} for g, v in strata.items() if v.any()}
    bt = lambda a, b, v=strata["overall"]: day_bootstrap(crps[a][v] - crps[b][v], day[v], sunny[v])
    res["hypotheses"]["E1 F1 G3''-F - F1 E4-B-fit (primary)"] = bt("F1|G3''-F", "F1|E4-B-fit")
    res["hypotheses"]["E2 F1 G3''-F - smart persistence"] = bt("F1|G3''-F", "persistence")
    res["hypotheses"]["E3 Z0 G3''-F - F1 G3''-F"] = bt("Z0|G3''-F", "F1|G3''-F")
    res["hypotheses"]["E3 Z0 G3''-F - smart persistence"] = bt("Z0|G3''-F", "persistence")
    res["hypotheses"]["F1 G3''-F - F1 E4-B-fit, ramp"] = bt("F1|G3''-F", "F1|E4-B-fit", ramp)
    res["hypotheses"]["F0 G3''-F - F1 G3''-F"] = bt("F0|G3''-F", "F1|G3''-F")
    # calibration (continuous rolling conformal, init from the last W development scores)
    cal = np.load(OUT / "dev_calibration_v1.npz")
    org = test.ds.origin_time.astype("datetime64[us]").astype(np.int64)[idx]
    tgt = test.ds.target_time.astype("datetime64[us]").astype(np.int64)[idx]
    for arm in ("F1", "Z0", "F0"):
        for mdl in ("G3''-F", "E4-B-fit"):
            key = f"{arm}|{mdl}"
            W = calj["W"][key]
            res["intervals"][key] = {}
            for nm, a in ALPHA.items():
                lo, hi = base_interval(fc[arm][mdl], LV, nm)
                o_ = np.argsort(cal["origin_us"], kind="stable")
                o_ = o_[cal["origin_us"][o_] < np.datetime64(start, "us").astype(np.int64)]  # causal: before the window
                init = cal[f"{key}|{nm}"][o_][-W:]
                r = replay(org, tgt, lo, hi, y, a, "rolling", W, init)
                cov = (y >= r["lo"]) & (y <= r["hi"])
                isc = interval_score(r["lo"], r["hi"], y, a)
                res["intervals"][key][nm] = {g: {"coverage": float(cov[v].mean()), "width": float((r["hi"] - r["lo"])[v].mean()),
                                                 "interval_score": float(isc[v].mean()),
                                                 "miss_below": float((y < r["lo"])[v].mean()), "miss_above": float((y > r["hi"])[v].mean())}
                                             for g, v in strata.items() if v.any()}
    res["hypotheses"]["E4 F1 G3''-F calibrated 90% coverage"] = res["intervals"]["F1|G3''-F"]["90"]["overall"]["coverage"]
    for dd in np.unique(day):
        s_ = day == dd
        res["per_day"][dd] = {"sunny": lab[dd], "n": int(s_.sum()), "n_ramp": int(ramp[s_].sum()),
                              **{k: float(c[s_].mean()) for k, c in crps.items()}}
    res["window"] = {"group": group, "start": str(start), "end_exclusive": str(end), "n_days": len(days)}
    res_path.write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(pred_path, sample_ids=test.ds.sample_ids[idx], y=y, persistence=pers,
                        **{f"{a}|{mm}": q for a, qs in fc.items() for mm, q in qs.items()})
    return res


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "run_test":
        run_test(sys.argv[2])
    elif sys.argv[1] == "rehearse":
        rehearse()
