"""Thin execution wrapper for the Gatton evaluation (audit fix A). Records commit, environment, timestamps, runtime,
artifact verification, output hashes, status and errors. Never retries.

  python -m src.gatton_execute rehearsal          development-only rehearsal (last 21 development days)
  python -m src.gatton_execute official TOKEN     the one-shot sealed test (requires reports/gatton_test_approval.txt)

Official mode refuses to start if an official execution log already exists (any status): a failed attempt must be
preserved under a new name, recorded in reports/phaseE_gatton_test_deviations.md and re-approved by the user before
any rerun. Nothing is written into the repository before the evaluation body starts (its clean-tree guard must pass).
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

OFFICIAL_LOG = Path("reports/phaseE_gatton_test_execution_log.json")
REHEARSAL_LOG = Path("reports/phaseE_gatton_rehearsal_log.json")


def _sha(p) -> str | None:
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest().upper() if p.exists() else None


def _env() -> dict:
    import numpy as np
    import torch
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__}


def main(mode: str, token: str | None = None):
    assert mode in ("rehearsal", "official")
    if mode == "official":
        assert token, "STOP: official mode needs the approval token"
        assert not OFFICIAL_LOG.exists(), "STOP: an official execution log already exists (no automatic retries)"
    log = {"mode": mode, "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "git_dirty_at_start": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()),
           "start_utc": datetime.now(timezone.utc).isoformat(), "environment": _env(), "invocations": 1}
    t0 = time.perf_counter()
    import src.gatton_eval as ge
    try:
        log["artifacts_verified"] = ge.verify_all_artifacts()  # fail closed (also re-checked inside run_test)
        res = ge.run_test(token) if mode == "official" else ge.rehearse()
        log["status"] = "completed"
        log["n_samples"], log["n_days"] = res["n"], res["n_days"]
    except BaseException:
        log["status"] = "FAILED"
        log["traceback"] = traceback.format_exc()
        raise
    finally:
        log["wall_time_s"] = time.perf_counter() - t0
        log["end_utc"] = datetime.now(timezone.utc).isoformat()
        outs = (["reports/phaseE_gatton_test_results.json", "experiments/E_gatton/test_predictions_v1.npz"] if mode == "official"
                else ["reports/phaseE_gatton_rehearsal_dev.json", "experiments/E_gatton/rehearsal_predictions_dev.npz"])
        log["output_sha256"] = {p: _sha(p) for p in outs}
        (OFFICIAL_LOG if mode == "official" else REHEARSAL_LOG).write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
