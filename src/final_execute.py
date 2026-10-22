"""Single invocation of the frozen one-shot runner src.final_eval.run_test(token), with an execution log.

Logs wall-clock timing, GPU memory and the git commit to reports/test_execution_log.json. On failure the state is
preserved and the traceback logged; nothing is retried or reset.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.final_eval import run_test

LOG = Path("reports/test_execution_log.json")


def main(token: str):
    assert not LOG.exists(), "STOP: an execution log already exists (one-shot)"
    log = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "start_utc": datetime.now(timezone.utc).isoformat(), "invocations_of_run_test": 1}
    LOG.write_text(json.dumps({**log, "status": "started"}, indent=1))
    t0 = time.perf_counter()
    try:
        res = run_test(token)
        log["status"] = "completed"
        log["n_samples"] = res["n"]
        log["n_days"] = res["n_days"]
    except BaseException:
        log["status"] = "FAILED"
        log["traceback"] = traceback.format_exc()
        raise
    finally:
        log["wall_time_s"] = time.perf_counter() - t0
        log["end_utc"] = datetime.now(timezone.utc).isoformat()
        log["gpu_max_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        if "n_samples" in log:
            log["amortised_end_to_end_s_per_sample"] = log["wall_time_s"] / log["n_samples"]
        LOG.write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
