"""Phase E: package Gatton in the SKIPP'D layout so the existing pipeline runs unchanged (prereg sec. 3, E-1..E-4).

Writes (data stays outside the repository):
  D:/solar_data/gatton/processed/gatton_skippd.hdf5   groups trainval (dev records: images_log, pv_log) and
                                                       test (test records: images_log; pv_log = NaN, SEALED)
  data/manifests/gatton_forecast_v1.npz (+.json)       SKIPP'D-format forecast manifest (dev 2-min, test 1-min, +-30 s)
  data/manifests/gatton_folds_v1.json                  folds + day labels (+ content_sha256)
  data/features_gatton/e8_windows_v1.npz, e8_record_features_v1.npz   H15/H30/H60 windows; zero solar/motion table
Rules: records with missing development PV are dropped from the timeline (E-1: samples needing them cannot form);
origins only where the development-fitted P_clear(t) >= c_min (daytime). Test PV is never read here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.e8_audit import WINDOWS, history_rows
from src.gatton_clearsky import GattonClearSky, load_params
from src.skippd import official_forecast_samples

P = Path(r"D:/solar_data/gatton/processed")
H5 = P / "gatton_skippd.hdf5"
MAN = Path("data/manifests/gatton_forecast_v1.npz")
FOLDS = Path("data/manifests/gatton_folds_v1.json")
FEAT = Path("data/features_gatton")
TEST_START = np.datetime64("2020-04-16")


def main():
    rec = np.load(P / "records.npz")
    imgs = np.load(P / "images_64.npy", mmap_mode="r")
    t = rec["times"].astype("datetime64[s]")
    is_test = rec["is_test"].astype(bool)
    assert ((t.astype("datetime64[D]") >= TEST_START) == is_test).all()
    pv = np.load(P / "pv_dev.npz")
    pv_at = dict(zip(pv["times"].astype(np.int64), pv["pv_kw"]))
    pv_rec = np.array([pv_at.get(int(s), np.nan) if not it else np.nan for s, it in zip(t.astype(np.int64), is_test)])
    keep = is_test | np.isfinite(pv_rec)  # E-1: drop dev records with missing PV
    print(f"records {len(t):,}; dev missing-PV records dropped {int((~keep).sum())}", flush=True)
    t, is_test, pv_rec, src_idx = t[keep], is_test[keep], pv_rec[keep], np.flatnonzero(keep)
    dev_rows, test_rows = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    with h5py.File(H5, "w") as f:
        for name, rows in (("trainval", dev_rows), ("test", test_rows)):
            g = f.create_group(name)
            g.create_dataset("images_log", data=np.asarray(imgs[src_idx[rows]]), compression="lzf", chunks=(256, 64, 64, 3))
            g.create_dataset("pv_log", data=pv_rec[rows].astype(np.float32))  # test: NaN (sealed)
    # timeline in SKIPP'D manifest format
    group = is_test.astype(np.int8)
    row = np.where(is_test, np.searchsorted(test_rows, np.arange(len(t))), np.searchsorted(dev_rows, np.arange(len(t))))
    times = t.astype("datetime64[us]").astype(object)
    dev_s = official_forecast_samples(times, sampling_interval_min=2, tolerance_s=30)
    test_s = official_forecast_samples(times, sampling_interval_min=1, tolerance_s=30)
    o_day = lambda s: t[s.origin].astype("datetime64[D]")
    kd, kt = o_day(dev_s) < TEST_START, o_day(test_s) >= TEST_START
    origin = np.concatenate([dev_s.origin[kd], test_s.origin[kt]])
    history = np.concatenate([dev_s.history[kd], test_s.history[kt]])
    target = np.concatenate([dev_s.target[kd], test_s.target[kt]])
    st = np.concatenate([np.zeros(kd.sum(), bool), np.ones(kt.sum(), bool)])
    # daytime origins only (development-fitted clear-sky model; an input-time quantity)
    cs = GattonClearSky.from_dev_fit()
    pc_origin = cs(t[origin].astype("datetime64[us]"))
    day_ok = pc_origin >= load_params()["c_min_kw"]
    o = np.argsort(origin[day_ok], kind="stable")
    origin, history, target, st = (a[day_ok][o] for a in (origin, history, target, st))
    groups_used = group[np.concatenate([history, target[:, None]], 1)]
    assert (groups_used == st[:, None].astype(np.int8)).all(), "STOP: split leakage across groups"
    folds = json.loads(Path("reports/phaseE_gatton_folds_v1.json").read_text())
    labels = folds["day_labels"]
    o_days = t[origin].astype("datetime64[D]").astype(str)
    is_sunny = np.array([bool(labels.get(d, False)) for d in o_days])
    arrays = {"times_us": t.astype("datetime64[us]").astype(np.int64), "hdf5_group": group, "hdf5_row": row.astype(np.int64),
              "origin": origin, "history": history, "target": target, "is_test": st, "is_sunny": is_sunny}
    MAN.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MAN, **arrays)
    digest = hashlib.sha256()
    for k in sorted(arrays):
        digest.update(k.encode()), digest.update(np.ascontiguousarray(arrays[k]).tobytes())
    info = {"n_records": int(len(t)), "n_dev": int((~st).sum()), "n_test": int(st.sum()),
            "n_dev_days": int(len(set(o_days[~st]))), "n_test_days": int(len(set(o_days[st]))),
            "dev_sampling_min": 2, "test_sampling_min": 1, "tolerance_s": 30, "content_sha256": digest.hexdigest()}
    MAN.with_suffix(".json").write_text(json.dumps(info, indent=1))
    print("manifest", info, flush=True)
    # folds file with content hash (train.py records it)
    folds["content_sha256"] = hashlib.sha256(json.dumps(folds, sort_keys=True).encode()).hexdigest()
    FOLDS.write_text(json.dumps(folds, indent=1))
    # E8-style windows for development samples; zero solar/motion table (A2 uses no features)
    FEAT.mkdir(parents=True, exist_ok=True)
    dev_idx = np.flatnonzero(~st)
    win = {"sample_ids": dev_idx}
    elig = {}
    for name, stride in WINDOWS.items():
        tl = history_rows(arrays["times_us"], origin[dev_idx], stride)
        ok = (tl >= 0).all(1)
        assert (group[np.maximum(tl, 0)][ok] == 0).all()
        win[name] = np.where(tl >= 0, row[np.maximum(tl, 0)], -1).astype(np.int64)
        elig[name] = ok
    win["eligible_common"] = elig["H15"] & elig["H30"] & elig["H60"]
    np.savez_compressed(FEAT / "e8_windows_v1.npz", **win)
    n_dev_rec = len(dev_rows)
    np.savez_compressed(FEAT / "e8_record_features_v1.npz", P=np.zeros((n_dev_rec, 5), np.float32),
                        M=np.zeros((n_dev_rec, 6), np.float32))
    (FEAT / "e8_features_v1.json").write_text(json.dumps({"note": "Gatton: no solar/motion features (zeros)"}))
    print("windows", {k: int(v.sum()) for k, v in elig.items()}, "common", int(win["eligible_common"].sum()), flush=True)


if __name__ == "__main__":
    main()
