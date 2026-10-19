import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.clear_sky import clear_sky_index, from_clear_sky_index, smart_persistence
from src.data import SkyPVDataset, validate_manifest, window_indices
from src.metrics import crps_ensemble, crps_from_quantiles, day_bootstrap_diff, pinball_loss
from src.models import EventSeverityHead, Forecaster, NonCrossingQuantileHead, SmallSkyEncoder
from src.ramp_labels import first_crossing_labels

QUANTILES = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]


def make_manifest(n=40, start="2018-06-04 18:00"):
    ts = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"timestamp_utc": ts, "image_path": [f"img_{i:03d}.png" for i in range(n)],
         "pv_kw": np.linspace(10, 20, n), "split": "train", "day": "2018-06-04"}
    )


def test_manifest_validator_catches_problems():
    df = make_manifest()
    assert validate_manifest(df) == []
    bad = pd.concat([df, df.iloc[:1].assign(split="val")], ignore_index=True)
    problems = " ".join(validate_manifest(bad))
    assert "duplicated" in problems and "more than one split" in problems
    naive = df.assign(timestamp_utc=df["timestamp_utc"].dt.tz_localize(None))
    assert any("timezone" in p for p in validate_manifest(naive))


def test_windows_are_causal_and_skip_gaps():
    df = make_manifest(60).drop(index=20).reset_index(drop=True)  # one missing minute
    ts = df["timestamp_utc"]
    w = window_indices(ts, history_len=15, horizons_minutes=[1, 5, 10, 15], day=df["day"])
    assert len(w.history) > 0
    origin = ts.to_numpy()[w.history[:, -1]]
    assert (ts.to_numpy()[w.history] <= origin[:, None]).all()
    assert (ts.to_numpy()[w.target] > origin[:, None]).all()
    # brute force: a window exists iff every required minute is present
    present = set(ts)
    one = pd.Timedelta(minutes=1)
    expected = [
        o for o in ts
        if all(o - k * one in present for k in range(15)) and all(o + h * one in present for h in (1, 5, 10, 15))
    ]
    assert list(origin) == expected
    deltas = np.diff(ts.to_numpy()[w.history], axis=1)
    assert (deltas == np.timedelta64(1, "m")).all()


def test_dataset_and_model_shapes(tmp_path):
    df = make_manifest(35)
    for p in df["image_path"]:
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(tmp_path / p)
    assert validate_manifest(df, check_images=True, root=tmp_path) == []
    w = window_indices(df["timestamp_utc"], horizons_minutes=[1, 5, 10, 15])
    loader = DataLoader(SkyPVDataset(df, w, root=tmp_path), batch_size=4)
    images, pv_hist, target = next(iter(loader))
    assert images.shape == (4, 15, 3, 64, 64) and pv_hist.shape == (4, 15) and target.shape == (4, 4)

    torch.manual_seed(0)
    model = Forecaster(SmallSkyEncoder(), NonCrossingQuantileHead(64, 4, QUANTILES))
    pred = model(images, pv_hist)
    assert pred.shape == (4, 4, 9)
    assert (pred[..., 1:] >= pred[..., :-1]).all()
    loss = pinball_loss(pred, target, QUANTILES)
    loss.backward()
    assert torch.isfinite(loss)


def test_event_severity_head():
    head = EventSeverityHead(hidden_size=64, n_time_bins=3)
    out = head(torch.randn(8, 64))
    assert torch.allclose(out["time_probs"].sum(-1), torch.ones(8))
    assert (out["sev_scale"] > 0).all()
    loss = head.nll(out, torch.tensor([0, 1, 2, 3, 3, 0, 1, 3]), torch.randn(8))
    loss.backward()
    assert torch.isfinite(loss)


def test_scores_match_toy_examples():
    # single-member ensemble == absolute error
    assert np.allclose(crps_ensemble([[3.0]], [1.0]), [2.0])
    # sorted-identity CRPS matches the brute-force pairwise definition
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(5, 30)), rng.normal(size=5)
    brute = np.abs(x - y[:, None]).mean(1) - 0.5 * np.abs(x[:, :, None] - x[:, None, :]).mean((1, 2))
    assert np.allclose(crps_ensemble(x, y), brute)
    # quantile CRPS approximation approaches the exact CRPS of a Normal
    q = np.linspace(0.005, 0.995, 199)
    from statistics import NormalDist
    pred_q = np.array([[NormalDist().inv_cdf(v) for v in q]])
    exact_normal_crps_at_0 = 2 * NormalDist().pdf(0) - 1 / np.sqrt(np.pi)
    assert abs(crps_from_quantiles(pred_q, [0.0], q)[0] - exact_normal_crps_at_0) < 0.01
    # interpolated-quantile CRPS: point forecast -> |error|; 9-level Gaussian -> close to exact CRPS
    from src.metrics import crps_quantile_interp, quantile_reliability
    lv = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
    assert np.allclose(crps_quantile_interp(np.full((2, 9), 3.0), [1.0, 4.0], lv), [2.0, 1.0])
    g9 = np.array([[NormalDist().inv_cdf(v) for v in lv]])
    approx = crps_quantile_interp(g9, [0.0], lv, n_grid=2000)[0]
    assert abs(approx - exact_normal_crps_at_0) / exact_normal_crps_at_0 < 0.05
    rel = quantile_reliability(np.tile(g9, (4000, 1)), rng.normal(size=4000))
    assert np.allclose(rel, lv, atol=0.03)
    # bootstrap: deterministic, and CI contains the point estimate
    loss_a, loss_b = rng.random(200) + 0.1, rng.random(200)
    day = np.repeat(np.arange(20), 10)
    strata = np.where(day < 10, "sunny", "cloudy")
    r1 = day_bootstrap_diff(loss_a, loss_b, day, strata, n_boot=500, seed=1)
    assert r1 == day_bootstrap_diff(loss_a, loss_b, day, strata, n_boot=500, seed=1)
    assert r1[1] <= r1[0] <= r1[2]


def test_persistence_and_ramp_labels():
    sp = smart_persistence([10.0, 0.5], [20.0, 0.1], [25.0, 0.3], min_clear_kw=1.0)
    assert np.allclose(sp, [12.5, 0.5])  # second sample falls back to plain persistence
    k = clear_sky_index([10.0, 1.0, 22.0], [20.0, 0.1, 20.0], min_clear_kw=1.0)
    assert k[0] == 0.5 and k[1] == 1.0 and k[2] > 1.0  # floored denominator; enhancement not clipped
    assert np.allclose(from_clear_sky_index(k, [20.0, 0.1, 20.0], 1.0), [10.0, 1.0, 22.0])

    k_future = np.full((3, 15), 0.9)
    k_future[0, 7:] = 0.4   # drop at minute 8
    k_future[1, 12:] = 1.3  # enhancement at minute 13
    lab = first_crossing_labels([0.9, 0.9, 0.9], k_future, threshold=0.3)
    assert lab["arrival_min"].tolist() == [8, 13, 0]
    assert lab["time_class"].tolist() == [1, 2, 3]
    assert np.allclose(lab["severity"], [-0.5, 0.4, 0.0])
