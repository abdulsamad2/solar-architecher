"""E0 baselines: official clear-sky-index persistence (as in SUNSET_forecast.ipynb cell 24) and naive persistence.

Official rule: pred = CSI(t, P_t) * P_theo(t + 15 min), using the verbatim Relative_output.
The reference time for P_theo is exactly t + 15 min (not the +-30 s matched target record).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from src.official.relative_op_func import Relative_output


def official_persistence(origin_times, pv_now, horizon_min: int = 15) -> np.ndarray:
    h = dt.timedelta(minutes=horizon_min)
    out = np.empty(len(pv_now))
    for i, (t, p) in enumerate(zip(origin_times, pv_now)):
        csi, _ = Relative_output(t, float(p))
        _, p_theo_future = Relative_output(t + h, 0.0)  # P_theo does not depend on the PV argument
        out[i] = csi * p_theo_future
    return out


def split_scores(pred, target, is_sunny) -> dict:
    pred, target, is_sunny = (np.asarray(a) for a in (pred, target, is_sunny))
    err = pred - target
    out = {}
    groups = [("overall", np.ones_like(is_sunny))]
    if is_sunny.any():  # sunny/cloudy labels exist only for the 20 official test days
        groups = [("sunny", is_sunny), ("cloudy", ~is_sunny)] + groups
    for name, mask in groups:
        e = err[mask]
        out[name] = {"n": int(mask.sum()), "rmse_kw": float(np.sqrt(np.mean(e**2))), "mae_kw": float(np.mean(np.abs(e)))}
    return out


def run(h5_path: str, manifest_path: str, out_dir: str) -> dict:
    from src.skippd import SkippdForecastDataset

    results = {"manifest": str(manifest_path)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "test"):
        ds = SkippdForecastDataset(h5_path, manifest_path, split=split, preload=False)
        pv_now = ds.pv[ds.hist_rows[:, -1]].numpy().astype(float)  # chronological: last frame is t
        target = ds.pv[ds.target_rows].numpy().astype(float)
        origin = ds.origin_time.astype(object)
        preds = {"official_csi_persistence": official_persistence(origin, pv_now), "naive_persistence": pv_now}
        results[split] = {k: split_scores(v, target, ds.is_sunny) for k, v in preds.items()}
        np.savez_compressed(
            out / f"preds_{split}.npz", sample_ids=ds.sample_ids, target=target, day=ds.day.astype(str),
            is_sunny=ds.is_sunny, **preds,
        )
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="E0 persistence baselines")
    p.add_argument("--h5", default="data/raw/2017_2019_images_pv_processed.hdf5")
    p.add_argument("--manifest", default="data/manifests/forecast_tol30_v1.npz")
    p.add_argument("--out", default="experiments/E0_persistence/tol30_v1")
    a = p.parse_args()
    print(json.dumps(run(a.h5, a.manifest, a.out), indent=2))
