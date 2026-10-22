"""Deviation 2 validation (development data only): record_table_unsorted on a non-chronological record order.

A slice of whole development days is stored as two out-of-order blocks (later days first), mimicking the official
test HDF5 layout. The fixed path must reproduce the frozen development features of those records bit for bit.
Output: reports/test_fix2_dev_check.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.e8_features import FEAT, GRAY, record_times
from src.final_eval import record_table_unsorted


def main():
    t = record_times()
    days = t.astype("datetime64[D]")
    ud = np.unique(days)
    a = np.flatnonzero(np.isin(days, ud[100:110]))  # 10 whole days
    b = np.flatnonzero(np.isin(days, ud[300:310]))  # 10 later whole days
    rows = np.concatenate([b, a])  # later block stored first, as in the official test file
    assert (np.diff(t[rows].astype(np.int64)) < 0).sum() == 1
    ref = np.load(FEAT / "e8_record_features_v1.npz")
    ref_table = np.concatenate([ref["P"], ref["M"]], 1).astype(np.float32)[rows]
    gray = np.load(GRAY, mmap_mode="r")[rows]
    got = record_table_unsorted(t[rows], gray, Path("data/features/gray_u8_fix2check.npy"))
    Path("data/features/gray_u8_fix2check.npy").unlink()
    res = {"n_records": int(len(rows)), "bit_identical": bool(np.array_equal(got, ref_table)),
           "max_abs_diff": float(np.abs(got - ref_table).max())}
    Path("reports/test_fix2_dev_check.json").write_text(json.dumps(res, indent=1))
    print(res)
    assert res["bit_identical"], "STOP: fix does not reproduce development features"


if __name__ == "__main__":
    main()
