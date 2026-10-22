# One-shot test evaluation: execution deviations

## Deviation 1: attempt 1 aborted at the runner's clean-tree guard (before unsealing)
- **When:** the first invocation after the approval commit 77fa2f3, at 2026-10-22T17:50:27Z (UTC, from the machine clock).
- **What happened:** the execution wrapper `src/final_execute.py` wrote a "started" entry to `reports/test_execution_log.json` before calling `run_test`. That untracked file made the tree dirty, so `run_test` stopped at its guard (`AssertionError: STOP: dirty tree`) after 0.2 s.
- **Test data accessed: none.**
  - The clean-tree guard runs before the inventory hash checks and before any test image, PV or label is loaded.
  - No test feature file was created (`data/features/gray_u8_test.npy` was absent), and no results or predictions were written.
  - Peak GPU memory was 0 GB.
- **Records:**
  - `reports/test_execution_attempt1_FAILED.json` (the wrapper log, with traceback);
  - `reports/test_execution_attempt1_FAILED_stdout.log` (console output).
- **Fix:** the wrapper now writes its log only after `run_test` returns or fails. The runner (`src/final_eval.py`), models, calibration, thresholds, hypotheses and scoring are unchanged, and the inventory hashes are unaffected.
- **Authorization:** the user explicitly approved the fix and a single rerun after this failure was reported. This follows the checklist rule for a crash before labels are read.
