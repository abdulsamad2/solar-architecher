"""Phase F runner (reports/phaseF_physics_prereg.md). Trains ONLY the A2 3D-CNN expert with physics configs.

  python scripts/run_physics_ablation.py search S1          # lambda_phys grid (seed 42, folds 1 3)
  python scripts/run_physics_ablation.py search S2a         # lambda_ramp grid, alpha=2
  python scripts/run_physics_ablation.py search S2b         # alpha {1,4} at best lambda_ramp
  python scripts/run_physics_ablation.py search S3          # lambda_bound grid on best phys+ramp
  python scripts/run_physics_ablation.py search S4          # direction (A5)
  python scripts/run_physics_ablation.py select             # apply the pre-registered selection rules -> configs/physics/A*.yaml
  python scripts/run_physics_ablation.py final A3 --seeds 42 123 456
Runs are skipped if their summary.json already exists (resumable). Never touches test data.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "physics"
OUT = ROOT / "experiments" / "F_physics"
A0_DIR = ROOT / "experiments" / "E9" / "v1" / "A2" / "csi"
SEARCH_FOLDS = (1, 3)
A2_ARGS = ["--target", "csi", "--input", "csi", "--head", "quantile", "--calib-holdout", "--e8-window", "H60",
           "--encoder", "cnn3d_gru", "--max-epochs", "15", "--patience", "5",
           "--frozen-thresholds", "reports/phase6_frozen_thresholds.json"]
GRID = {"S1": [0.01, 0.05, 0.1, 0.25], "S2a": [0.01, 0.05, 0.1, 0.25], "S2b": [1, 4], "S3": [0.001, 0.005, 0.01],
        "S4": [0.01, 0.05]}


def make_cfg(phys=None, ramp=None, alpha=2.0, bound=None, direction=None) -> dict:
    c = {"p_scale_kw": 30.0, "c_min_kw": 1.0}
    if phys:
        c["physics_loss"] = {"enabled": True, "type": "mae", "lambda_phys": phys}
    if ramp:
        c["ramp_loss"] = {"enabled": True, "lambda_ramp": ramp, "alpha_ramp": alpha}
    if bound:
        c["bound_loss"] = {"enabled": True, "lambda_bound": bound, "k_max": 1.5}
    if direction:
        c["direction_loss"] = {"enabled": True, "lambda_direction": direction, "margin": 0.02}
    return c


def run(name: str, cfg: dict, seeds, folds) -> Path:
    path = CFG / "runs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    out = OUT / name
    for s in seeds:
        if (out / "csi" / f"seed{s}" / "summary.json").exists():
            print(f"skip {name} seed {s} (done)")
            continue
        cmd = [sys.executable, "-m", "src.train", *A2_ARGS, "--seed", str(s), "--out", str(out),
               "--physics-config", str(path), "--fold-ids", *map(str, folds)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    return out


def fold_metrics(run_dir: Path, seed: int, folds) -> dict:
    """Mean over folds of the expert's validation CRPS (overall, ramp) on the common population."""
    from src.gate2_report import ramp_labels
    import numpy as np
    from src.metrics import crps_quantile_interp
    global _RL
    try:
        _RL
    except NameError:
        _RL = ramp_labels(str(ROOT / "data/raw/2017_2019_images_pv_processed.hdf5"),
                          str(ROOT / "data/manifests/forecast_tol30_v2.npz"),
                          json.loads((ROOT / "data/manifests/folds_blocked5_v1.json").read_text()))["is_ramp"]
    ov, rp = [], []
    for k in folds:
        z = np.load(run_dir / f"seed{seed}" / f"fold_{k}" / "preds_val.npz")
        c = crps_quantile_interp(z["pred_q"], z["target"], z["quantile_levels"])
        r = np.array([_RL[i] for i in z["sample_ids"]])
        ov.append(c.mean()), rp.append(c[r].mean())
    return {"overall": float(np.mean(ov)), "ramp": float(np.mean(rp))}


def best_by(cands: dict, key: str, a0: dict, max_overall: float | None = None, max_ramp: float | None = None):
    ok = {v: m for v, m in cands.items() if (max_overall is None or m["overall"] <= max_overall)
          and (max_ramp is None or m["ramp"] <= max_ramp)}
    if not ok:
        return min(cands), True  # smallest value, flagged
    return min(ok, key=lambda v: (round(ok[v][key], 6), v)), False


def stage_results(stage: str, values) -> dict:
    return {v: fold_metrics(OUT / f"{stage}_{v}" / "csi", 42, SEARCH_FOLDS) for v in values}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "select", "final"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    a = ap.parse_args()
    sel_path = CFG / "selection.json"
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    if a.cmd == "search":
        st = a.arg
        if st == "S1":
            for v in GRID["S1"]:
                run(f"S1_{v}", make_cfg(phys=v), [42], SEARCH_FOLDS)
        elif st == "S2a":
            for v in GRID["S2a"]:
                run(f"S2a_{v}", make_cfg(ramp=v, alpha=2.0), [42], SEARCH_FOLDS)
        elif st == "S2b":
            for v in GRID["S2b"]:
                run(f"S2b_{v}", make_cfg(ramp=sel["lambda_ramp"], alpha=float(v)), [42], SEARCH_FOLDS)
        elif st == "S3":
            for v in GRID["S3"]:
                run(f"S3_{v}", make_cfg(phys=sel["lambda_phys"], ramp=sel["lambda_ramp"], alpha=sel["alpha_ramp"], bound=v),
                    [42], SEARCH_FOLDS)
        elif st == "S4":
            for v in GRID["S4"]:
                run(f"S4_{v}", make_cfg(phys=sel["lambda_phys"], ramp=sel["lambda_ramp"], alpha=sel["alpha_ramp"],
                                        bound=sel["lambda_bound"], direction=v), [42], SEARCH_FOLDS)
        else:
            raise SystemExit("unknown stage")
    elif a.cmd == "select":
        a0 = fold_metrics(A0_DIR, 42, SEARCH_FOLDS)
        sel = {"A0_search": a0}
        r1 = stage_results("S1", GRID["S1"])
        sel["lambda_phys"], sel["S1_flag"] = best_by(r1, "overall", a0)
        r2 = stage_results("S2a", GRID["S2a"])
        sel["lambda_ramp"], sel["S2a_flag"] = best_by(r2, "ramp", a0, max_overall=1.01 * a0["overall"])
        sel["alpha_ramp"] = 2.0
        if all((OUT / f"S2b_{v}" / "csi" / "seed42" / "summary.json").exists() for v in GRID["S2b"]):
            r2b = {2.0: r2[sel["lambda_ramp"]], **{float(v): m for v, m in stage_results("S2b", GRID["S2b"]).items()}}
            sel["alpha_ramp"], sel["S2b_flag"] = best_by(r2b, "ramp", a0, max_overall=1.01 * a0["overall"])
        if all((OUT / f"S3_{v}" / "csi" / "seed42" / "summary.json").exists() for v in GRID["S3"]):
            r3 = stage_results("S3", GRID["S3"])
            # constraint reference: phys+ramp combination = S3 candidate ramp must not exceed the best S2 ramp CRPS
            sel["lambda_bound"], sel["S3_flag"] = best_by(r3, "overall", a0, max_ramp=min(m["ramp"] for m in r2.values()))
            sel["S3_results"] = r3
        if all((OUT / f"S4_{v}" / "csi" / "seed42" / "summary.json").exists() for v in GRID["S4"]):
            r4 = stage_results("S4", GRID["S4"])
            sel["lambda_direction"], sel["S4_flag"] = best_by(r4, "ramp", a0, max_overall=1.01 * a0["overall"])
            sel["S4_results"] = r4
        sel.update({"S1_results": r1, "S2a_results": r2})
        sel_path.write_text(json.dumps(sel, indent=1, default=str))
        arms = {"A1": make_cfg(phys=sel["lambda_phys"]), "R": make_cfg(ramp=sel["lambda_ramp"], alpha=sel["alpha_ramp"]),
                "A3": make_cfg(phys=sel["lambda_phys"], ramp=sel["lambda_ramp"], alpha=sel["alpha_ramp"])}
        if "lambda_bound" in sel:
            arms["A4"] = make_cfg(phys=sel["lambda_phys"], ramp=sel["lambda_ramp"], alpha=sel["alpha_ramp"], bound=sel["lambda_bound"])
        if "lambda_direction" in sel:
            arms["A5"] = make_cfg(phys=sel["lambda_phys"], ramp=sel["lambda_ramp"], alpha=sel["alpha_ramp"],
                                  bound=sel["lambda_bound"], direction=sel["lambda_direction"])
        for name, c in arms.items():
            (CFG / f"{name}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
        print(json.dumps({k: v for k, v in sel.items() if not k.endswith("_results")}, indent=1, default=str))
    else:
        cfg = yaml.safe_load((CFG / f"{a.arg}.yaml").read_text())
        run(f"final_{a.arg}", cfg, a.seeds, range(5))


if __name__ == "__main__":
    main()
