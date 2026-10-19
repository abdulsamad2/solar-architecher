"""Phase 2.6 preprocessing audit on the REAL frozen manifests. Opt-in: ``pytest -m realdata``.

Skips if the raw data or manifests are absent.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.clear_sky import ClearSkyPV, clear_sky_index
from src.folds import build as build_folds
from src.skippd import TEST_DATES, build_forecast_manifest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
H5 = RAW / "2017_2019_images_pv_processed.hdf5"
MANIFEST = ROOT / "data" / "manifests" / "forecast_tol30_v2.npz"
FOLDS = ROOT / "data" / "manifests" / "folds_blocked5_v1.json"
TOL_US = 30_000_000
MIN_US = 60_000_000
GROUPS = {0: "trainval", 1: "test"}

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(not (H5.exists() and MANIFEST.exists() and FOLDS.exists()), reason="real data not present"),
]


@pytest.fixture(scope="module")
def m():
    z = np.load(MANIFEST)
    return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def pv():
    with h5py.File(H5, "r") as f:
        return {g: f[f"{name}/pv_log"][...] for g, name in GROUPS.items()}


def _sample_groups(m):
    return m["hdf5_group"][m["history"][:, 0]]


def test_1_image_readability_and_quality(m):
    """Every referenced row exists; no blank/constant or fully saturated frame is used by any sample."""
    report = {}
    with h5py.File(H5, "r") as f:
        for g, name in GROUPS.items():
            ds = f[f"{name}/images_log"]
            sel = _sample_groups(m) == g
            used = np.unique(np.concatenate([m["hdf5_row"][m["history"][sel]].ravel(), m["hdf5_row"][m["target"][sel]]]))
            assert used.min() >= 0 and used.max() < ds.shape[0]
            assert ds.shape[1:] == (64, 64, 3) and ds.dtype == np.uint8
            std = np.empty(ds.shape[0], np.float32)
            mean = np.empty(ds.shape[0], np.float32)
            for s in range(0, ds.shape[0], 20000):
                block = ds[s:s + 20000].reshape(-1, 64 * 64 * 3).astype(np.float32)
                std[s:s + 20000], mean[s:s + 20000] = block.std(1), block.mean(1)
            bad = np.flatnonzero((std < 1.0) | (mean > 250.0))
            report[name] = {"n_images": int(ds.shape[0]), "bad_images": int(len(bad)),
                            "bad_used_by_samples": int(np.isin(bad, used).sum())}
    print(json.dumps(report))
    assert all(r["bad_used_by_samples"] == 0 for r in report.values()), report


def test_2_time_ordering_and_5_gap_policy(m):
    """Frames are distinct, strictly ordered, column 0 is the origin, and each frame is within 30 s of t - j min."""
    t = m["times_us"]
    h = t[m["history"]]  # newest first
    assert (m["history"][:, 0] == m["origin"]).all()
    assert (np.diff(h, axis=1) < 0).all()
    nominal = t[m["origin"]][:, None] - np.arange(h.shape[1])[None, :] * MIN_US
    assert (np.abs(h - nominal) <= TOL_US).all()
    assert len(np.unique(m["origin"])) == len(m["origin"])


def test_3_forecast_leakage(m):
    t = m["times_us"]
    t_tgt, t_org = t[m["target"]], t[m["origin"]]
    assert (t_tgt > t[m["history"]].max(axis=1)).all()
    assert (np.abs(t_tgt - (t_org + 15 * MIN_US)) <= TOL_US).all()


def test_4_day_and_split_leakage(m):
    t = m["times_us"].astype("datetime64[us]")
    day = t.astype("datetime64[D]")
    org_day = day[m["origin"]]
    assert (day[m["history"]] == org_day[:, None]).all(), "a history frame crosses days"
    assert (day[m["target"]] == org_day).all(), "a target crosses days"
    groups = m["hdf5_group"][np.concatenate([m["history"], m["target"][:, None]], axis=1)]
    assert (groups == m["is_test"][:, None]).all(), "sample mixes dev and test records"
    is_test_day = np.array([d in TEST_DATES for d in org_day.astype(object)])
    assert (is_test_day == m["is_test"]).all()

    folds = json.loads(FOLDS.read_text())
    dev_days = set(org_day[~m["is_test"]].astype(str))
    assert not {str(d) for d in TEST_DATES} & set(folds["day_labels"])
    for fd in folds["folds"]:
        tr, va, pu = set(fd["train_days"]), set(fd["val_days"]), set(fd["purged_days"])
        assert not (tr & va) and not (tr & pu) and not (va & pu)
        assert dev_days <= tr | va | pu, "a dev sample day is in no fold role"
        lo, hi = min(va), max(va)
        assert not [d for d in tr if lo <= d <= hi], "training day inside validation span"
    all_val = [d for fd in folds["folds"] for d in fd["val_days"]]
    assert len(all_val) == len(set(all_val)) and set(all_val) == set(folds["day_labels"])


def test_6_units(m, pv):
    for g in GROUPS:
        assert np.isfinite(pv[g]).all() and pv[g].min() >= 0 and pv[g].max() < 35, "PV not in plausible kW range"
    rows = m["hdf5_row"][m["target"]]
    grp = m["hdf5_group"][m["target"]]
    y = np.where(grp == 0, pv[0][np.minimum(rows, len(pv[0]) - 1)], pv[1][np.minimum(rows, len(pv[1]) - 1)])
    assert np.isfinite(y).all() and (y >= 0).all()


def test_7_clear_sky_stability_per_fold(m, pv):
    """Fit the anchor on each fold's training days; every dev target K must be finite and P_clear >= 0."""
    folds = json.loads(FOLDS.read_text())
    g0 = m["hdf5_group"] == 0
    rec_times = m["times_us"][g0][np.argsort(m["hdf5_row"][g0])].astype("datetime64[us]")
    rec_day = rec_times.astype("datetime64[D]").astype(str)
    dev = ~m["is_test"]
    tgt_rows = m["hdf5_row"][m["target"][dev]]
    report = []
    for fd in folds["folds"]:
        tr = np.isin(rec_day, fd["train_days"])
        cs = ClearSkyPV().fit(rec_times[tr], pv[0][tr])
        pc = cs(rec_times)
        assert np.isfinite(pc).all() and (pc >= 0).all()
        k = clear_sky_index(pv[0][tgt_rows], pc[tgt_rows], 1.0)
        assert np.isfinite(k).all()
        report.append({"fold": fd["fold"], "seasonal": bool(cs.seasonal), "K_p50": round(float(np.median(k)), 3),
                       "K_p99": round(float(np.quantile(k, 0.99)), 3), "K_max": round(float(k.max()), 2),
                       "frac_K_gt_2": round(float((k > 2).mean()), 5)})
    print(json.dumps(report))
    assert all(r["seasonal"] for r in report)


def test_8_determinism(tmp_path):
    info_ref = json.loads(MANIFEST.with_suffix(".json").read_text())
    info_new = build_forecast_manifest(RAW, tmp_path / "m.npz", tolerance_s=info_ref["tolerance_s"],
                                       dev_sampling_min=info_ref["dev_sampling_min"],
                                       test_sampling_min=info_ref["test_sampling_min"])
    assert info_new["content_sha256"] == info_ref["content_sha256"]
    folds_ref = json.loads(FOLDS.read_text())
    folds_new = build_folds(RAW, tmp_path / "f.json", folds_ref["n_folds"], folds_ref["purge_days"])
    assert folds_new["content_sha256"] == folds_ref["content_sha256"]
