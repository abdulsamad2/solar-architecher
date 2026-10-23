"""Phase A2 (reports/phaseA_comparability_prereg.md): ONE-SHOT evaluation of the frozen models on SkyGPT's public
external test set (5 cloudy days, Nov-Dec 2019; never seen in development, which ends 2019-10-26).

Order of operations: guards -> input checks -> ALL inference with the target PV absent -> only then the target PV
(pv_pred[:, 14]) is read for scoring. G3''-F's frozen routing gives G3 for every sample (no 60-min history exists).
Outputs: reports/phaseA_a2_results.json, experiments/phaseA/a2_predictions.npz
"""
from __future__ import annotations

import datetime as dt
import io
import json
import pickle
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch

from src.baselines import official_persistence
from src.cal2 import ALPHA, interval_score
from src.calibrate import base_interval
from src.clear_sky import ClearSkyPV
from src.final_eval import FOLDS, SEEDS, assemble, day_bootstrap, nearest_init, rolling_conformal_daily
from src.metrics import crps_quantile_interp
from src.sunset import SunsetNet
from src.train import GpuSeq

SRC = Path("data/external/skygpt")
H5X, TIMES = SRC / "test_set_2019nov_dec.hdf5", SRC / "times_curr_test_2019nov_dec.npy"
RESULTS, PREDS = Path("reports/phaseA_a2_results.json"), Path("experiments/phaseA/a2_predictions.npz")
LEVELS = np.array([.05, .1, .2, .3, .5, .7, .8, .9, .95])
PUBLISHED = {"smart persistence": {"crps": 3.67}, "SUNSET": {"crps": 3.31, "ws90": 56.95},
             "SkyGPT": {"crps": 2.81, "ws90": 26.70}}
DEV_END = np.datetime64("2019-10-26")


class _SafeUnpickler(pickle.Unpickler):
    """Only numpy array reconstruction and datetime objects; anything else in the untrusted file -> STOP."""
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype"), ("datetime", "datetime")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"STOP: disallowed class {module}.{name} in timestamp file")
        return super().find_class(module, name)


def load_times(path) -> np.ndarray:
    from numpy.lib import format as F
    with open(path, "rb") as fh:
        assert F.read_magic(fh) == (1, 0)
        shape, _, dtype = F.read_array_header_1_0(fh)
        assert dtype == np.dtype("O"), "STOP: unexpected timestamp dtype"
        arr = _SafeUnpickler(io.BytesIO(fh.read())).load()
    assert arr.shape == shape and all(type(x) is dt.datetime for x in arr), "STOP: timestamps are not datetimes"
    return arr


class ExternalDS:
    """SkippdForecastDataset-compatible view: 17 records per sample (16 chronological history frames + the target)."""

    def __init__(self, images_log: np.ndarray, pv_log: np.ndarray, t: np.ndarray, target_pv: np.ndarray,
                 rec_times: np.ndarray | None = None):
        """rec_times: optional [N, 17] record times; default = exact 1-min grid t-15..t and t+15 (external file)."""
        n = len(t)
        self.sample_ids = np.arange(n)
        rec = np.arange(n * 17).reshape(n, 17)
        self.hist_rows, self.target_rows = rec[:, :16].copy(), rec[:, 16].copy()
        imgs = np.zeros((n, 17, 64, 64, 3), np.uint8)
        imgs[:, :16] = images_log
        self.images = torch.from_numpy(imgs.reshape(n * 17, 64, 64, 3))
        pv = np.concatenate([pv_log, target_pv[:, None]], 1).reshape(-1)
        self.pv = torch.from_numpy(pv.astype(np.float32))
        origin = t.astype("datetime64[us]")
        offs = np.concatenate([np.arange(-15, 1), [15]]).astype("timedelta64[m]")
        grid = origin[:, None] + offs[None, :] if rec_times is None else np.asarray(rec_times, "datetime64[us]")
        self.rec_t = grid.reshape(-1)
        self.origin_time, self.target_time = grid[:, 15], grid[:, 16]
        self.day = origin.astype("datetime64[D]")
        self.is_sunny = np.zeros(n, bool)


class ExternalSplit:
    """Split-compatible object for src.final_eval.member_predict / assemble (no 60-min history anywhere)."""

    def __init__(self, ds: ExternalDS, device):
        self.name, self.ds, self.device, self.table = "external", ds, device, None
        self.rec_t = ds.rec_t
        self.pv = ds.pv.numpy().astype(float)
        self.h60_ok = np.zeros(len(ds.sample_ids), bool)
        self.h60_rows = ds.hist_rows
        self.data = GpuSeq(ds, device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def sunset_members(ds: ExternalDS, device) -> np.ndarray:
    """The 10 frozen SUNSET-port fold models: input = 16 frames newest first (48 channels) + raw kW PV newest first."""
    rows = torch.from_numpy(ds.hist_rows[:, ::-1].copy()).to(device)
    images, pv = ds.images.to(device), ds.pv.to(device)
    out = []
    for k in range(1, 11):
        net = SunsetNet().to(device)
        net.load_state_dict(torch.load(f"experiments/E1_sunset_port/tol30_v2/fold_{k}/best.pt", map_location=device))
        net.eval()
        preds = []
        for s in range(0, len(rows), 1000):
            r = rows[s:s + 1000]
            x = images[r]
            b, t_, h, w, c = x.shape
            preds.append(net(x.permute(0, 1, 4, 2, 3).reshape(b, t_ * c, h, w).float().div_(255.0), pv[r]))
        out.append(torch.cat(preds).squeeze(-1).cpu().numpy().astype(float))
    return np.stack(out)  # [10, N]


def ensemble_crps(members: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Empirical CRPS of an ensemble (SkyGPT Eq. 13 with the empirical predictive CDF): E|X-y| - 0.5 E|X-X'|."""
    a = np.abs(members - y[None]).mean(0)
    b = np.abs(members[:, None] - members[None]).mean((0, 1))
    return a - 0.5 * b


def main():
    # guards (one shot)
    assert not RESULTS.exists() and not PREDS.exists(), "STOP: Phase A2 already executed (one-shot)"
    assert not subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip(), "STOP: dirty tree"
    # input checks (no target PV read)
    t = load_times(TIMES)
    with h5py.File(H5X, "r") as f:
        g = f["test"]
        assert set(g.keys()) == {"images_log", "images_pred", "pv_log", "pv_pred"}, "STOP: unexpected HDF5 layout"
        images_log, pv_log = g["images_log"][...], g["pv_log"][...]
    n = len(t)
    assert images_log.shape == (n, 16, 64, 64, 3) and pv_log.shape == (n, 16) and n == 2582, "STOP: shapes"
    assert np.isfinite(pv_log).all() and pv_log.min() >= 0 and pv_log.max() < 40, "STOP: PV history range"
    days = np.unique(np.array(t, dtype="datetime64[D]"))
    skippd_test = {np.datetime64(d) for d in json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text()).get(
        "test_days", [])}
    assert days.min() > DEV_END and not (set(days) & skippd_test), "STOP: external days overlap development/test"
    tt = np.array(t, dtype="datetime64[us]")
    assert (np.diff(tt) > np.timedelta64(0, "s")).all(), "STOP: timestamps not strictly increasing"
    ref = h5py.File("data/raw/2017_2019_images_pv_processed.hdf5", "r")["trainval/images_log"][-2000:]
    mask_check = {"external_corner_mean": float(images_log[:, :, :6, :6].mean()), "skippd_corner_mean": float(ref[:, :6, :6].mean()),
                  "external_corner_zero_share": float((images_log[:, :, :6, :6] == 0).mean()),
                  "skippd_corner_zero_share": float((ref[:, :6, :6] == 0).mean())}
    # all inference with the target absent (placeholder 0 kW; never used by any model input)
    ds = ExternalDS(images_log, pv_log, t, np.zeros(n))
    split = ExternalSplit(ds, torch.device("cuda"))
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    tau = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    fc = assemble(split, {(s, k): np.arange(n) for s in SEEDS for k in FOLDS}, folds_doc, tau)
    assert all((fc[m]["n_members"] == 15).all() for m in ("G3''-F", "E4-B-fit", "E6-S", "G4"))
    sun = sunset_members(ds, split.device)
    pers = official_persistence(ds.origin_time.astype(object), pv_log[:, -1])
    PREDS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PREDS, origin_time=tt, **{f"q|{m}": fc[m]["pred_q"] for m in ("G3''-F", "E4-B-fit", "E6-S", "G4")},
                        sunset_members=sun, persistence=pers)
    # only now: the target PV
    with h5py.File(H5X, "r") as f:
        y = f["test/pv_pred"][:, 14].astype(float)
    assert np.isfinite(y).all()
    day = np.array(t, dtype="datetime64[D]").astype(str)
    sunny = np.zeros(n, bool)
    rd = json.loads(Path("reports/test_ramp_definition_frozen.json").read_text())
    anchor = ClearSkyPV(rd["anchor_coef"])
    k_now = pv_log[:, -1] / np.maximum(anchor(tt), 1.0)
    k_tgt = y / np.maximum(anchor(tt + np.timedelta64(15, "m")), 1.0)
    ramp = np.abs(k_tgt - k_now) >= rd["tau_test"]
    masks = {"overall": np.ones(n, bool), "ramp": ramp, "non-ramp": ~ramp}
    q = {m: fc[m]["pred_q"] for m in ("G3''-F", "E4-B-fit", "E6-S", "G4")}
    crps = {m: crps_quantile_interp(q[m], y, LEVELS) for m in q}
    crps["SUNSET port (10-member empirical CRPS)"] = ensemble_crps(sun, y)
    crps["SUNSET port (ensemble-mean MAE)"] = np.abs(sun.mean(0) - y)
    crps["smart persistence"] = np.abs(pers - y)
    res = {"note": "Phase A2 one-shot external test (SkyGPT's 5 cloudy Nov-Dec 2019 days). G3''-F = G3 here by its frozen routing.",
           "n": n, "days": sorted(set(day)), "published_SkyGPT_Table1_test": PUBLISHED, "mask_check": mask_check,
           "n_ramp": int(ramp.sum()), "crps": {}, "point": {}, "winkler90_raw": {}, "intervals_calibrated": {}, "per_day": {},
           "bootstrap": {}}
    pts = {**{m: q[m][:, 4] for m in q}, "SUNSET port": sun.mean(0), "smart persistence": pers}
    for m, c in crps.items():
        res["crps"][m] = {g: float(c[v].mean()) for g, v in masks.items() if v.any()}
        res["crps"][m]["fs_vs_persistence"] = float(1 - c.mean() / crps["smart persistence"].mean())
    for m, f_ in pts.items():
        res["point"][m] = {"rmse": float(np.sqrt(np.mean((f_ - y) ** 2))), "mae": float(np.mean(np.abs(f_ - y)))}
    for m in q:
        res["winkler90_raw"][m] = {g: float(interval_score(q[m][v, 0], q[m][v, 8], y[v], 0.1).mean()) for g, v in masks.items() if v.any()}
        res["winkler90_raw"][m]["coverage"] = float(np.mean((y >= q[m][:, 0]) & (y <= q[m][:, 8])))
    s_lo, s_hi = np.quantile(sun, 0.05, axis=0), np.quantile(sun, 0.95, axis=0)
    res["winkler90_raw"]["SUNSET port"] = {"overall": float(interval_score(s_lo, s_hi, y, 0.1).mean()),
                                           "coverage": float(np.mean((y >= s_lo) & (y <= s_hi)))}
    cal = np.load("experiments/final/dev_oof_calibration_v1.npz")
    meta = json.loads(Path("experiments/final/dev_oof_calibration_v1.json").read_text())
    org = tt.astype(np.int64)
    tgt = (tt + np.timedelta64(15, "m")).astype(np.int64)
    for m in q:
        res["intervals_calibrated"][m] = {}
        for nm, a in ALPHA.items():
            lo, hi = base_interval(q[m], LEVELS, nm)
            inits = {d: nearest_init(cal["origin_us"], cal[f"{m}|{nm}"], org[day == d].min(), meta["W"][m]) for d in np.unique(day)}
            clo, chi = rolling_conformal_daily(org, tgt, day, lo, hi, y, a, meta["W"][m], inits)
            res["intervals_calibrated"][m][nm] = {g: {"coverage": float(np.mean((y[v] >= clo[v]) & (y[v] <= chi[v]))),
                                                      "width_kw": float(np.mean(chi[v] - clo[v])),
                                                      "interval_score": float(interval_score(clo[v], chi[v], y[v], a).mean()),
                                                      "miss_below": float(np.mean(y[v] < clo[v])), "miss_above": float(np.mean(y[v] > chi[v]))}
                                                  for g, v in masks.items() if v.any()}
    for d in np.unique(day):
        s = day == d
        res["per_day"][d] = {"n": int(s.sum()), "n_ramp": int(ramp[s].sum()), **{m: float(c[s].mean()) for m, c in crps.items()}}
    for comp in ("E4-B-fit", "G4", "smart persistence", "SUNSET port (10-member empirical CRPS)"):
        res["bootstrap"][f"G3''-F - {comp}"] = {g: day_bootstrap(crps["G3''-F"][v] - crps[comp][v], day[v], sunny[v])
                                                for g, v in masks.items() if v.any()}
    p_ours = crps["smart persistence"].mean()
    res["population_check"] = {"our_persistence_crps": float(p_ours), "published": 3.67,
                               "within_5pct": bool(abs(p_ours / 3.67 - 1) <= 0.05)}
    RESULTS.write_text(json.dumps(res, indent=1))
    print(json.dumps({"crps": res["crps"], "winkler90_raw": res["winkler90_raw"], "population_check": res["population_check"],
                      "mask_check": mask_check}, indent=1))


if __name__ == "__main__":
    main()
