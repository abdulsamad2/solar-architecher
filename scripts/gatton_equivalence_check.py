"""Equivalence of the dd39740 Gatton evaluation (old run_test) and the refactored evaluate_window (new), on the SAME
development samples (rehearsal window: last 21 development days). Development data only; no sealed row is read.

The old function source is taken verbatim from `git show dd39740:src/gatton_eval.py`; only these documented text
patches are applied (each asserted to match exactly), all of which retarget the window without changing logic:
  P1 remove the approval-token check          P2 results path -> scratch      P3 remove the clean-tree check
  P4 CSV row window -> development window     P5 HDF5 group 1 -> 0 (development records)
  P6 GattonSplit("test") -> GattonSplit("dev") (2x)
  P7 samples restricted to the window days    P8 calibration init restricted to origins before the window (causal)
  P9 predictions path -> scratch
Expected exact differences: none in predictions or statistics; the bookkeeping field n_excluded_missing_pv differs
by construction in the rehearsal only (old counts excluded samples outside the window too).
Output: reports/phaseE_gatton_equivalence.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.gatton_eval as ge  # noqa: E402

SCR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiments/E_gatton/equivalence"


def old_run_test_source() -> str:
    src = subprocess.run(["git", "show", "dd39740:src/gatton_eval.py"], capture_output=True, text=True, cwd=ROOT).stdout
    body = src[src.index("def run_test(token: str):"):src.index('\n\nif __name__ == "__main__":')]
    patches = [
        ('    assert Path("reports/gatton_test_approval.txt").read_text().strip() == token, "STOP: no recorded explicit approval"\n', ""),
        ('res_path = Path("reports/phaseE_gatton_test_results.json")', "res_path = Path(OLD_RES)"),
        ('    assert not subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip(), "STOP: dirty tree"\n', ""),
        ('(t_all >= "2020-04-16") & (t_all < "2020-05-07")', "(t_all >= WIN_START) & (t_all < WIN_END)"),
        ('sel = m["hdf5_group"] == 1', 'sel = m["hdf5_group"] == 0'),
        ('GattonSplit("test", ', 'GattonSplit("dev", '),
        ("idx = np.flatnonzero(valid)", "idx = np.flatnonzero(valid & np.isin(test.ds.day.astype(str), WIN_DAYS))"),
        ('init = cal[f"{key}|{nm}"][np.argsort(cal["origin_us"], kind="stable")][-W:]',
         'o_ = np.argsort(cal["origin_us"], kind="stable"); o_ = o_[cal["origin_us"][o_] < WIN_START_US]; '
         'init = cal[f"{key}|{nm}"][o_][-W:]'),
        ('OUT / "test_predictions_v1.npz"', "Path(OLD_NPZ)"),
    ]
    for old, new in patches:
        n = body.count(old)
        assert n == (2 if old.startswith("GattonSplit") else 1), f"patch target count {n}: {old[:60]}"
        body = body.replace(old, new)
    return body


def compare(a, b, path="", tol=0.0, out=None):
    out = [] if out is None else out
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append((path + "/" + k, "missing"))
            else:
                compare(a[k], b[k], f"{path}/{k}", tol, out)
    elif isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", tol, out)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if not (abs(a - b) <= tol or (np.isnan(a) and np.isnan(b))):
            out.append((path, a, b))
    elif a != b:
        out.append((path, a, b))
    return out


def main():
    SCR.mkdir(parents=True, exist_ok=True)
    dev_days = sorted({d for fd in ge.FOLDS["folds"] for d in fd["val_days"]})
    window = dev_days[-21:]
    start, end = np.datetime64(window[0]), np.datetime64(window[-1]) + np.timedelta64(1, "D")
    assert end <= ge.TEST_START
    ns = dict(vars(ge))
    ns.update(OLD_RES=str(SCR / "old_results.json"), OLD_NPZ=str(SCR / "old_predictions.npz"), WIN_START=str(start),
              WIN_END=str(end), WIN_DAYS=window, WIN_START_US=np.datetime64(start, "us").astype(np.int64))
    exec(old_run_test_source(), ns)
    old = ns["run_test"]("unused")
    new = ge.evaluate_window("dev", window, start, end, SCR / "new_results.json", SCR / "new_predictions.npz")
    po, pn = np.load(SCR / "old_predictions.npz"), np.load(SCR / "new_predictions.npz")
    pred = {k: float(np.max(np.abs(po[k] - pn[k]))) if po[k].dtype.kind == "f" else bool(np.array_equal(po[k], pn[k]))
            for k in po.files}
    new.pop("window", None)
    diffs_exact = compare(json.loads(json.dumps(old, default=float)), json.loads(json.dumps(new, default=float)))
    rep = {"window": [str(start), str(end)], "n_samples": new["n"], "prediction_max_abs_diff": pred,
           "statistics_differences": [list(map(str, d)) for d in diffs_exact],
           "expected_difference_fields": ["/n_excluded_missing_pv"]}
    unexpected = [d for d in diffs_exact if d[0] != "/n_excluded_missing_pv"]
    rep["predictions_identical"] = all((v == 0.0) if isinstance(v, float) else v for v in pred.values())
    rep["statistics_identical_except_expected"] = not unexpected
    Path(ROOT / "reports/phaseE_gatton_equivalence.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1)[:3000])


if __name__ == "__main__":
    main()
