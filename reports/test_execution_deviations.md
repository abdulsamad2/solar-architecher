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

## Deviation 2: attempt 2 crashed while computing test input features (before any PV or label was read)
- **When:** the invocation at commit b6d4b79, at 2026-10-22T17:52:31Z (machine clock).
- **What happened:**
  - All guards passed: token, one-shot check, clean tree, frozen-file hashes and the 140 inventory hashes.
  - Step 1 read the official test sky images (inputs only) and wrote `data/features/gray_u8_test.npy` (gitignored).
  - `record_features` then failed with `AssertionError: records must be sorted`.
- **Cause:** the official SKIPP'D test HDF5 is not stored chronologically. Rows 0–6985 end at 2019-10-14 18:13:40, and row 6986 is 2017-06-24 06:38:40, so the file holds two time-ordered blocks. The development HDF5 is fully chronological, so the dry run and rehearsal could not expose this. The diagnosis used timestamps only.
- **Test data accessed:** the test images, as inputs for the greyscale features. **No test PV, labels or outcomes were read**; `Split("test")` was never constructed, and no predictions or results were written.
- **Fix:** `src/final_eval.py` gains `record_table_unsorted`. It computes the features on the chronologically sorted test timeline and maps the rows back to HDF5 row order. This yields exactly the features a chronologically stored file would give. Models, calibration, thresholds, hypotheses and scoring are unchanged.
- **Validation (development data only):**
  - **Real-data check:** 20 whole dev days (13,203 records), stored as two out-of-order blocks, reproduce the frozen development features **bit for bit** (max diff 0.0; `reports/test_fix2_dev_check.json`, `src/final_fix2_check.py`).
  - **Synthetic test:** `tests/test_final_unsorted.py`.
- **Consequence:** the `src/final_eval.py` hash in `reports/test_execution_manifest.json` is the pre-fix hash. This deviation supersedes it.
- **Records:** `reports/test_execution_attempt2_FAILED.json` and `reports/test_execution_attempt2_FAILED_stdout.log`.
- **Authorization:** the user explicitly approved this fix and a single rerun after the failure was reported.
