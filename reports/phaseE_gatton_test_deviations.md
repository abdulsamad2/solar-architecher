# Gatton one-shot sealed test: execution deviations

**Scientific freeze reference:** `dd39740`. **Execution commit:** recorded in `reports/phaseE_gatton_test_execution_log.json` at run time.

## Rules (fixed before execution)
1. The official evaluation is started only through `python -m src.gatton_execute official <TOKEN>`, after the token has been recorded in `reports/gatton_test_approval.txt` and committed.
2. The wrapper never retries. If an official execution log exists in any state, a new official attempt is refused.
3. **Failure before any sealed row is read** (guard, hash, import or setup error):
   - preserve the log as `reports/phaseE_gatton_test_attempt<N>_FAILED.json`;
   - record the cause here;
   - fix only the cause, without changing models, settings or scoring;
   - obtain the user's explicit re-approval before the next attempt.
4. **Failure after sealed rows have been read:**
   - preserve all state and record it here;
   - rerun only identical code, and only with the user's explicit approval;
   - any change touching models, rules or scoring is forbidden; stop and report instead.
5. **Raw outputs are preserved unchanged:**
   - `reports/phaseE_gatton_test_results.json`, `experiments/E_gatton/test_predictions_v1.npz` and the execution log are committed immediately after the run;
   - their SHA-256 hashes are recorded in the log;
   - analysis (report, E5 lag analysis) starts only after that commit.

## Deviations
**None.** The official run (token `GATTON-FINAL-20261024`, HEAD `9f5164e`) completed on the first and only attempt in 372.6 s. No failure, no retry, no change to models, settings or scoring. Raw outputs were committed unchanged at `59ccbcd` before any analysis.
