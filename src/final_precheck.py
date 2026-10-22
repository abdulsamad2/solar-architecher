"""Pre-unseal checks for the one-shot test evaluation; writes reports/test_execution_manifest.json.

Reads no official test images, PV or labels. Every check must pass; otherwise the run must not be unsealed.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from src.e8_features import verify_frozen
from src.final_eval import EXPERTS, FOLDS, MODELS, SEEDS, TOL_KW, sha256

PREREG = Path("reports/test_eval_prereg.md")
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def main():
    checks = {}
    # 1. prereg frozen: frozen header, only appended amendments since the freeze commit 491fac2
    txt = PREREG.read_text(encoding="utf-8")
    numstat = git("diff", "--numstat", "491fac2", "HEAD", "--", str(PREREG)).split()
    checks["1 prereg frozen"] = txt.startswith("# Sealed SKIPP'D test evaluation: pre-registration (FROZEN)") and \
        numstat[1] == "0" and not git("status", "--porcelain", "--", str(PREREG))
    # 2. clean tree
    checks["2 clean tree"] = git("status", "--porcelain") == ""
    # 3. inventory hashes
    inv = json.loads(Path("reports/checkpoint_inventory.json").read_text())["files"]
    bad = [p for p, h in inv.items() if sha256(p) != h]
    checks["3 inventory hashes (140)"] = len(inv) == 140 and not bad
    # 4. inventory covers exactly the frozen model configurations / checkpoints used by the runner
    need = []
    for model, (root, window) in EXPERTS.items():
        for s in SEEDS:
            need.append(root / f"seed{s}" / "config.json")
            for k in FOLDS:
                need.append(root / f"seed{s}" / f"fold_{k}" / "best.pt")
                if window == "H60":
                    need.append(root / f"seed{s}" / f"fold_{k}" / "e8_standardisation.json")
            cfg = json.loads((root / f"seed{s}" / "config.json").read_text())
            assert cfg.get("seed", s) == s
    for s in SEEDS:
        for k in FOLDS:
            need += [Path("experiments/E7b_gate/v1") / f"seed{s}" / f"fold_{k}" / "params.json",
                     Path("experiments/E9/v1/G3pp_params") / f"seed{s}" / f"fold_{k}" / "params.json"]
    need += [Path(p) for p in ("experiments/final/dev_oof_calibration_v1.npz", "reports/test_ramp_definition_frozen.json",
                               "reports/phase6_frozen_thresholds.json", "experiments/E1_sunset_port/tol30_v2/preds_test.npz")]
    missing = [str(p).replace("\\", "/") for p in need if str(p).replace("\\", "/") not in inv]
    checks["4 inventory covers runner models"] = not missing
    # 5. amendments documented incl. the original failed max-only criterion
    checks["5 amendments documented"] = all(s in txt for s in ("## Amendment 1", "## Amendment 2", "within **1e−4 kW** (max \\|diff\\|)",
                                                               "5.8e−3", "1.14e−3", "≥ 99.9% of samples within 1e−4 kW"))
    # 7. no previous official evaluation
    checks["7 no previous evaluation"] = not Path("reports/test_results.json").exists() and \
        not Path("experiments/final/test_predictions_v1.npz").exists() and not Path("data/features/gray_u8_test.npy").exists() and \
        "test_results" not in git("log", "--all", "--name-only", "--format=")
    verify_frozen()
    checks["geometry/eligibility frozen hashes (verify_frozen)"] = True
    cal = json.loads(Path("experiments/final/dev_oof_calibration_v1.json").read_text())
    dryrun = json.loads(Path("reports/final_dryrun_results.json").read_text())
    manifest = {
        "git_commit_before_authorization": git("rev-parse", "HEAD"),
        "checks": checks, "inventory_hash_mismatches": bad, "inventory_missing": missing,
        # 6. exact configuration being executed
        "configuration": {
            "primary": "G3''-F: G3'' (A2 3D-CNN + E6-S, frozen disagreement gate) if valid 60-min history, else G3 (E4-B-fit + E6-S, frozen E7b gate)",
            "ensemble": "15 members (seeds 42/123/456 x folds 0-4), equal-weight level-wise quantile averaging",
            "comparators": [m for m in MODELS if m != "G3''-F"] + ["SUNSET port", "official smart persistence"],
            "descriptive": "D4 and E4-B-fit on the 60-min eligible population",
            "calibration": {"method": "causal rolling-window conformal, daily reset, nearest-in-time dev out-of-fold init, 15-min maturation",
                            "W": cal["W"], "init_scores_sha256": cal["sha256"]},
            "ramp_definition": json.loads(Path("reports/test_ramp_definition_frozen.json").read_text()),
            "scoring": "crps_quantile_interp; paired test-day bootstrap 10,000 resamples, stratified sunny/cloudy, seed 20261022",
            "hypotheses": ["H1 (sole primary) overall CRPS G3''-F - E4-B-fit < 0: CI upper < 0",
                           "H2 vs persistence (overall, cloudy)", "H3 ramp non-inferiority margin 0.02 kW",
                           "H4 RMSE skill vs persistence (and SUNSET)", "H5 calibrated 90% coverage"],
            "reproduction_tolerance_kw": TOL_KW, "dry_run_routing": dryrun["routing"],
        },
        "code_sha256": {p: sha256(p) for p in ("src/final_eval.py", "src/cal2.py", "src/metrics.py", "src/gate7b.py",
                                               "src/train.py", "src/skippd.py", "src/clear_sky.py", "src/e8_features.py",
                                               "src/baselines.py", "src/calibrate.py")},
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "numpy": np.__version__},
    }
    manifest["all_checks_pass"] = all(checks.values())
    Path("reports/test_execution_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"checks": checks, "missing": missing, "bad": bad}, indent=1))
    assert manifest["all_checks_pass"], "STOP: pre-unseal check failed"


if __name__ == "__main__":
    main()
