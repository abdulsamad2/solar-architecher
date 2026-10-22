"""Final sealed-test evaluation pipeline (protocol: reports/test_eval_prereg.md).

Pieces:
  * member inference: rebuild each (model, seed, fold) member's exact inputs (fit-day anchor, K history, window,
    standardised features) on a split and run its frozen checkpoint
  * G3''-F routing: G3'' (A2 + E6-S, frozen disagreement gate) where a valid 60-min history exists, else G3
    (E4-B-fit + E6-S, frozen E7b gate). Routing depends only on timestamps (history availability).
  * ensemble: equal-weight level-wise quantile averaging over the 15 members (5 folds x 3 seeds)
  * calibration: empirical adaptive (causal rolling-window) conformal, initialised per test day from out-of-fold
    development scores, reset at each day boundary; no finite-sample coverage guarantee is claimed
  * scoring: crps_quantile_interp, paired test-day bootstrap stratified 10 sunny / 10 cloudy

The test entry point (run_test) is guarded and is NOT executed during development.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from src.cal2 import ALPHA, conformity, conformal_q, interval_score, widen
from src.calibrate import base_interval
from src.clear_sky import ClearSkyPV, clear_sky_index
from src.e8_audit import history_rows
from src.e8_features import FEAT, verify_frozen
from src.folds import calibration_split
from src.gate7b import apply_gate, raw_features
from src.metrics import crps_quantile_interp
from src.skippd import SkippdForecastDataset
from src.train import GpuSeq, build_model, predict_raw

H5 = "data/raw/2017_2019_images_pv_processed.hdf5"
MAN = "data/manifests/forecast_tol30_v2.npz"
SEEDS, FOLDS = (42, 123, 456), range(5)
EXPERTS = {"E4-B-fit": (Path("experiments/E4_calib/v1/csi"), None),
           "E6-S": (Path("experiments/E6/v1/E6-S/csi"), None),
           "A2": (Path("experiments/E9/v1/A2/csi"), "H60"),
           "D4": (Path("experiments/E8/v1/D4/csi"), "H60")}
GATES = {"G3": ("E4-B-fit", Path("experiments/E7b_gate/v1"), "G3"), "G4": ("E4-B-fit", Path("experiments/E7b_gate/v1"), "G4"),
         "G3''": ("A2", Path("experiments/E9/v1/G3pp_params"), None)}
TOL_KW = 1e-4  # frozen numerical-reproduction tolerance (max |difference|, kW)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


class Split:
    """GPU-resident split (dev or test) with record timeline, PV, 60-min windows and feature table."""

    def __init__(self, name: str, device, record_table: np.ndarray | None = None):
        assert name in ("dev", "test")
        self.name = name
        self.ds = SkippdForecastDataset(H5, MAN, name, preload=True)
        m = np.load(MAN)
        g = 0 if name == "dev" else 1
        sel = m["hdf5_group"] == g
        self.rec_t = m["times_us"][sel][np.argsort(m["hdf5_row"][sel])].astype("datetime64[us]")
        self.pv = self.ds.pv.numpy().astype(float)
        origin_tl = m["origin"][m["is_test"] == (name == "test")]
        tl = history_rows(m["times_us"], origin_tl, 4)  # H60: 16 frames, 4-min stride, timestamps only
        self.h60_ok = (tl >= 0).all(1)
        assert (m["hdf5_group"][np.maximum(tl, 0)][self.h60_ok] == g).all(), "STOP: window crosses splits"
        rows = np.where(tl >= 0, m["hdf5_row"][np.maximum(tl, 0)], -1)[:, ::-1]  # chronological
        self.h60_rows = np.where(self.h60_ok[:, None], rows, self.ds.hist_rows)  # placeholder rows never used
        assert (self.h60_rows[:, -1] == self.ds.hist_rows[:, -1]).all(), "STOP: window origin mismatch"
        self.data = GpuSeq(self.ds, device)
        self.table = record_table
        self.device = device
        # same numerical backend settings as training/validation inference (src/train.py main)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


_DEV_ARRAYS = None


def dev_arrays():
    """Development record times and PV (for fitting each member's fit-day anchor)."""
    global _DEV_ARRAYS
    if _DEV_ARRAYS is None:
        m = np.load(MAN)
        sel = m["hdf5_group"] == 0
        t = m["times_us"][sel][np.argsort(m["hdf5_row"][sel])].astype("datetime64[us]")
        import h5py
        with h5py.File(H5, "r") as f:
            pv = f["trainval/pv_log"][...].astype(float)
        _DEV_ARRAYS = (t, pv)
    return _DEV_ARRAYS


def member_anchor(fold: dict) -> ClearSkyPV:
    t, pv = dev_arrays()
    fit = calibration_split(fold["train_days"])[0]
    sel = np.isin(t.astype("datetime64[D]").astype(str), fit)
    return ClearSkyPV().fit(t[sel], pv[sel])


def member_predict(split: Split, model: str, seed: int, fold: dict, idx: np.ndarray) -> dict:
    """Quantiles in kW (+ context) of one frozen member on the given sample indices of the split."""
    root, window = EXPERTS[model]
    run = root / f"seed{seed}"
    cfg = json.loads((run / "config.json").read_text())
    fdir = run / f"fold_{fold['fold']}"
    pc = member_anchor(fold)(split.rec_t)
    k_rec = clear_sky_index(split.pv, pc, 1.0)
    data = split.data
    data.set_history_feature(k_rec)
    data.residual = False
    data.k_now = torch.tensor(k_rec[split.ds.hist_rows[:, -1]], dtype=torch.float32, device=split.device)
    if window == "H60":
        assert split.h60_ok[idx].all(), "STOP: H60 member asked to predict an ineligible sample"
        data.set_window(split.h60_rows)
        std = json.loads((fdir / "e8_standardisation.json").read_text())
        mask = np.array(std["mask"], np.float32)
        table = split.table if mask.any() else np.zeros((len(split.pv), 11), np.float32)
        assert table is not None, "STOP: feature table required"
        data.set_extra(((table - np.array(std["mu"])) / np.array(std["sd"]) * mask).astype(np.float32))
    else:
        data.set_window(split.ds.hist_rows)
        data.set_extra(None)
    net = build_model(cfg).to(split.device)
    net.encoder.cnn_bf16 = cfg["cnn_bf16"]
    net.load_state_dict(torch.load(fdir / "best.pt", map_location=split.device))
    raw = predict_raw(net, data, torch.as_tensor(idx, device=split.device)).cpu().numpy().astype(float)
    pc_t = pc[split.ds.target_rows[idx]]
    pc_n = pc[split.ds.hist_rows[idx, -1]]
    return {"pred_q": raw * np.maximum(pc_t, 1.0)[:, None], "pclear_target_kw": pc_t, "pclear_now_kw": pc_n,
            "pv_now_kw": split.pv[split.ds.hist_rows[idx, -1]], "quantile_levels": np.array([.05, .1, .2, .3, .5, .7, .8, .9, .95])}


def gate_params(name: str, seed: int, k: int) -> dict:
    _, root, key = GATES[name]
    p = json.loads((root / f"seed{seed}" / f"fold_{k}" / "params.json").read_text())
    return p[key] if key else p


def blend(direct: dict, resid: dict, params: dict, tau: float) -> np.ndarray:
    w = apply_gate(params, raw_features(direct, resid, tau))
    assert np.isfinite(w).all()
    return w[:, None] * direct["pred_q"] + (1 - w[:, None]) * resid["pred_q"]


def route_g3pp_f(split: Split, seed: int, fold: dict, idx: np.ndarray, tau: float, cache: dict) -> np.ndarray:
    """G3''-F member: G3'' where the 60-min history is available (timestamp rule), else G3."""
    ok = split.h60_ok[idx]
    get = lambda mdl, ii: cache.setdefault((mdl, seed, fold["fold"], ii.tobytes()), member_predict(split, mdl, seed, fold, ii))
    out = np.empty((len(idx), 9))
    if ok.any():
        ii = idx[ok]
        out[ok] = blend(get("A2", ii), get("E6-S", ii), gate_params("G3''", seed, fold["fold"]), tau)
    if (~ok).any():
        ii = idx[~ok]
        out[~ok] = blend(get("E4-B-fit", ii), get("E6-S", ii), gate_params("G3", seed, fold["fold"]), tau)
    return out


def ensemble(member_q: list[np.ndarray]) -> np.ndarray:
    """Equal-weight level-wise quantile averaging; monotone if every member is."""
    q = np.mean(np.stack(member_q), axis=0)
    assert np.isfinite(q).all() and (np.diff(q, axis=1) >= -1e-9).all(), "STOP: invalid ensemble quantiles"
    return q


def rolling_conformal_daily(origin_us, target_us, day, lo, hi, y, alpha, W, init_scores_by_day):
    """Causal rolling-window conformal, reset at every day boundary.

    For each day: the window starts with init_scores_by_day[day] (W development out-of-fold scores, chronological);
    a forecast's score enters the window only once its target time <= the current issue time (15-min maturation);
    windows never carry across days (disconnected-day policy). Within-day gaps need no special handling.
    """
    import bisect
    import heapq
    from collections import deque

    out_lo, out_hi = np.empty(len(y)), np.empty(len(y))
    base = conformity(lo, hi, y)
    for d in np.unique(day):
        sel = np.flatnonzero(day == d)
        sel = sel[np.argsort(origin_us[sel], kind="stable")]
        window = deque(np.asarray(init_scores_by_day[d], float)[-W:])
        ordered, pending = sorted(window), []
        for i in sel:
            while pending and pending[0][0] <= origin_us[i]:
                tt, j = heapq.heappop(pending)
                assert tt <= origin_us[i], "STOP: immature outcome used"
                window.append(base[j]); bisect.insort(ordered, base[j])
                if len(window) > W:
                    old = window.popleft(); del ordered[bisect.bisect_left(ordered, old)]
            a, b, _ = widen(lo[i], hi[i], conformal_q(ordered, 1 - alpha))
            assert np.isfinite(a) and np.isfinite(b) and a <= b, "STOP: invalid interval"
            out_lo[i], out_hi[i] = a, b
            heapq.heappush(pending, (target_us[i], i))
    return out_lo, out_hi


def nearest_init(dev_origin_us, dev_scores, day_start_us, W):
    """The W development out-of-fold scores closest in time to a day's start, returned chronologically."""
    nearest = np.argsort(np.abs(dev_origin_us - day_start_us), kind="stable")[:W]
    nearest = nearest[np.argsort(dev_origin_us[nearest], kind="stable")]
    return dev_scores[nearest]


def day_bootstrap(diff_by_sample, day, sunny_day, n_boot=10_000, seed=20261022):
    """Paired test-day bootstrap of a mean per-sample score difference, stratified sunny / cloudy days."""
    days, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, weights=diff_by_sample)
    c = np.bincount(inv)
    sunny = np.zeros(len(days), bool)
    sunny[inv] = sunny_day
    groups = [np.flatnonzero(sunny), np.flatnonzero(~sunny)]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        ii = np.concatenate([rng.choice(g, len(g)) for g in groups if len(g)])
        boots[b] = s[ii].sum() / c[ii].sum()
    return float(s.sum() / c.sum()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


MODELS = ("G3''-F", "E4-B-fit", "E6-S", "G4", "D4")


def assemble(split: Split, assignments: dict, folds_doc: dict, frozen_tau: dict) -> dict:
    """Ensemble forecasts per model.

    assignments: {(seed, fold): sample-index array that member predicts}. Test: every member predicts every
    sample (15-member ensemble). Development out-of-fold surrogate: member (s, k) predicts only fold k's
    validation samples (3-member ensemble per sample; never a sample the member was fitted on).
    D4 predicts only 60-min-eligible samples.
    """
    n = len(split.ds.sample_ids)
    acc = {m: np.zeros((n, 9)) for m in MODELS}
    cnt = {m: np.zeros(n) for m in MODELS}
    for (s, k), idx in assignments.items():
        fold = {**folds_doc["folds"][k]}
        tau = frozen_tau[str(k)]["training_labels_fit_days"]
        cache = {}
        get = lambda mdl, ii: cache.setdefault((mdl, ii.tobytes()), member_predict(split, mdl, s, fold, ii))
        q = {"G3''-F": route_g3pp_f(split, s, fold, idx, tau, {}),
             "E4-B-fit": get("E4-B-fit", idx)["pred_q"], "E6-S": get("E6-S", idx)["pred_q"]}
        q["G4"] = blend(get("E4-B-fit", idx), get("E6-S", idx), gate_params("G4", s, k), tau)
        for m in ("G3''-F", "E4-B-fit", "E6-S", "G4"):
            acc[m][idx] += q[m]
            cnt[m][idx] += 1
        ok = idx[split.h60_ok[idx]]
        if len(ok):
            acc["D4"][ok] += get("D4", ok)["pred_q"]
            cnt["D4"][ok] += 1
        torch.cuda.empty_cache()
    out = {}
    for m in MODELS:
        has = cnt[m] > 0
        q = np.full((n, 9), np.nan)
        q[has] = acc[m][has] / cnt[m][has][:, None]
        assert (np.diff(q[has], axis=1) >= -1e-9).all(), f"STOP: crossing ensemble quantiles ({m})"
        out[m] = {"pred_q": q, "n_members": cnt[m]}
    return out


def score(q, y, levels, masks: dict) -> dict:
    c = crps_quantile_interp(q, y, levels)
    med = q[:, list(levels).index(0.5)]
    return {g: {"n": int(m.sum()), "crps_kw": float(c[m].mean()), "rmse_median_kw": float(np.sqrt(np.mean((med[m] - y[m]) ** 2))),
                "mae_median_kw": float(np.abs(med[m] - y[m]).mean())} for g, m in masks.items() if m.any()}


def record_table_unsorted(t, gray, gray_path) -> np.ndarray:
    """P/M feature table in HDF5 row order for records not stored chronologically (official test HDF5: two blocks).

    Features are computed on the chronologically sorted timeline (record_features requires it) and mapped back.
    Deviation 2 of reports/test_execution_deviations.md.
    """
    from src.e8_features import record_features
    order = np.argsort(np.asarray(t, dtype="datetime64[us]").astype(np.int64), kind="stable")
    inv = np.argsort(order, kind="stable")
    np.save(gray_path, gray[order])
    rf = record_features(np.asarray(t)[order], gray_path)
    return np.concatenate([rf["P"], rf["M"]], 1)[inv].astype(np.float32)


def run_test(approval_token: str):
    """ONE-SHOT official test evaluation (reports/test_eval_prereg.md). Not executed during development."""
    expected = Path("reports/test_eval_approval.txt")
    assert expected.exists() and expected.read_text().strip() == approval_token, "STOP: no recorded explicit approval"
    assert not Path("reports/test_results.json").exists(), "STOP: test results already exist (one-shot)"
    assert not subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip(), "STOP: dirty tree"
    verify_frozen()
    inv = json.loads(Path("reports/checkpoint_inventory.json").read_text())
    for path, h in inv["files"].items():
        assert sha256(path) == h, f"STOP: hash mismatch {path}"
    import h5py

    from src.baselines import official_persistence
    from src.cal2 import ALPHA
    from src.skippd import SUNNY_DATES
    # 1) test-record features (inputs only) for D4
    with h5py.File(H5, "r") as f:
        ds_img = f["test/images_log"]
        gray = np.round(ds_img[...].astype(np.float32).mean(-1)).astype(np.uint8)
    m = np.load(MAN)
    sel = m["hdf5_group"] == 1
    t_test = m["times_us"][sel][np.argsort(m["hdf5_row"][sel])].astype("datetime64[us]")
    table = record_table_unsorted(t_test, gray, FEAT / "gray_u8_test.npy")
    split = Split("test", torch.device("cuda"), table)
    n = len(split.ds.sample_ids)
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    tau = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    fc = assemble(split, {(s, k): np.arange(n) for s in SEEDS for k in FOLDS}, folds_doc, tau)
    assert all((fc[mm]["n_members"] == 15).all() for mm in ("G3''-F", "E4-B-fit", "E6-S", "G4"))
    day = split.ds.day.astype(str)
    sunny = np.array([np.datetime64(d).astype(object) in SUNNY_DATES for d in day])
    e1 = np.load("experiments/E1_sunset_port/tol30_v2/preds_test.npz")
    assert np.array_equal(e1["sample_ids"], split.ds.sample_ids), "STOP: SUNSET sample order differs"
    res = evaluate(split, np.arange(n), fc, sunny, e1["ensemble"], exclude_same_day=False)
    Path("reports/test_results.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed("experiments/final/test_predictions_v1.npz", sample_ids=split.ds.sample_ids,
                        **{f"q|{mm}": fc[mm]["pred_q"] for mm in MODELS}, sunset=e1["ensemble"])
    return res


def evaluate(split: Split, idx: np.ndarray, fc: dict, sunny: np.ndarray, sunset, exclude_same_day: bool) -> dict:
    """Scoring body shared by the one-shot test run and the development rehearsal (identical code path).

    idx: evaluated samples (split indices); sunny: day label per evaluated sample; sunset: SUNSET point forecast per
    evaluated sample or None. Calibration initialises from the frozen development out-of-fold scores; for the
    development rehearsal exclude_same_day=True draws them from OTHER days only.
    """
    from src.baselines import official_persistence
    from src.cal2 import ALPHA

    n = len(idx)
    y = split.pv[split.ds.target_rows[idx]]
    day = split.ds.day.astype(str)[idx]
    rd = json.loads(Path("reports/test_ramp_definition_frozen.json").read_text())
    k = split.pv / np.maximum(ClearSkyPV(rd["anchor_coef"])(split.rec_t), 1.0)
    ramp = np.abs(k[split.ds.target_rows[idx]] - k[split.ds.hist_rows[idx, -1]]) >= rd["tau_test"]
    h60 = split.h60_ok[idx]
    masks = {"overall": np.ones(n, bool), "sunny": sunny, "cloudy": ~sunny, "ramp": ramp, "non-ramp": ~ramp,
             "sunrise ramp": ramp & ~h60}
    levels = np.array([.05, .1, .2, .3, .5, .7, .8, .9, .95])
    pers = official_persistence(split.ds.origin_time[idx].astype(object), split.pv[split.ds.hist_rows[idx, -1]])
    q = {mm: fc[mm]["pred_q"][idx] for mm in MODELS}
    res = {"n": n, "n_days": int(len(np.unique(day))), "routing_share_G3pp": float(h60.mean()), "scores": {}, "hypotheses": {}, "intervals": {}}
    for mm in ("G3''-F", "E4-B-fit", "E6-S", "G4"):
        res["scores"][mm] = score(q[mm], y, levels, masks)
    res["scores"]["D4 (60-min eligible)"] = score(q["D4"][h60], y[h60], levels, {g: v[h60] for g, v in masks.items()})
    res["scores"]["E4-B-fit (60-min eligible)"] = score(q["E4-B-fit"][h60], y[h60], levels, {g: v[h60] for g, v in masks.items()})
    res["scores"]["persistence"] = score(np.repeat(pers[:, None], 9, 1), y, levels, masks)
    if sunset is not None:
        res["scores"]["SUNSET port"] = score(np.repeat(np.asarray(sunset)[:, None], 9, 1), y, levels, masks)
    c = {mm: crps_quantile_interp(q[mm], y, levels) for mm in ("G3''-F", "E4-B-fit")}
    c["persistence"] = np.abs(pers - y)
    res["hypotheses"]["H1 overall CRPS G3''-F - E4-B-fit"] = day_bootstrap(c["G3''-F"] - c["E4-B-fit"], day, sunny)
    res["hypotheses"]["H2 overall CRPS G3''-F - persistence"] = day_bootstrap(c["G3''-F"] - c["persistence"], day, sunny)
    cl = ~sunny
    res["hypotheses"]["H2 cloudy CRPS G3''-F - persistence"] = day_bootstrap(c["G3''-F"][cl] - c["persistence"][cl], day[cl], sunny[cl])
    res["hypotheses"]["H3 ramp CRPS G3''-F - E4-B-fit (non-inferiority margin 0.02)"] = day_bootstrap(
        c["G3''-F"][ramp] - c["E4-B-fit"][ramp], day[ramp], sunny[ramp])
    rm = lambda f: float(np.sqrt(np.mean((f - y) ** 2)))
    skill = {"G3''-F": 1 - rm(q["G3''-F"][:, 4]) / rm(pers)}
    if sunset is not None:
        skill["SUNSET port"] = 1 - rm(np.asarray(sunset)) / rm(pers)
    res["hypotheses"]["H4 RMSE skill vs persistence"] = skill
    cal = np.load("experiments/final/dev_oof_calibration_v1.npz")
    meta = json.loads(Path("experiments/final/dev_oof_calibration_v1.json").read_text())
    org = split.ds.origin_time.astype("datetime64[us]").astype(np.int64)[idx]
    tgt = split.ds.target_time.astype("datetime64[us]").astype(np.int64)[idx]
    cal_day = cal["origin_us"].astype("datetime64[us]").astype("datetime64[D]").astype(str)
    for mm in ("G3''-F", "E4-B-fit", "E6-S", "G4"):
        res["intervals"][mm] = {}
        for nm, a in ALPHA.items():
            lo, hi = base_interval(q[mm], levels, nm)
            W = meta["W"][mm]
            inits = {}
            for d in np.unique(day):
                keep = cal_day != d if exclude_same_day else np.ones(len(cal_day), bool)
                inits[d] = nearest_init(cal["origin_us"][keep], cal[f"{mm}|{nm}"][keep], org[day == d].min(), W)
            clo, chi = rolling_conformal_daily(org, tgt, day, lo, hi, y, a, W, inits)
            res["intervals"][mm][nm] = {g: {"coverage": float(np.mean((y[v] >= clo[v]) & (y[v] <= chi[v]))),
                                            "width_kw": float(np.mean(chi[v] - clo[v])),
                                            "interval_score": float(interval_score(clo[v], chi[v], y[v], a).mean())}
                                        for g, v in masks.items() if v.any()}
    res["hypotheses"]["H5 G3''-F calibrated 90% coverage"] = res["intervals"]["G3''-F"]["90"]["overall"]["coverage"]
    return res
