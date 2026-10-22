# One-shot official test evaluation: execution checklist

**Status: AWAITING EXPLICIT APPROVAL. Nothing below has been executed.**

## A. Review before approval
- [ ] `reports/test_eval_prereg.md`: protocol, including **Amendment 2** (a bf16-aware reproduction criterion, changed after a check failed).
- [ ] `reports/final_dryrun_report.md`: reproduction, out-of-fold development results, calibration freeze, rehearsal.
- [ ] `reports/leakage_audit_final.md`
- [ ] `reports/checkpoint_inventory.json` (140 hashes)
- [ ] `reports/e9_report.md`: why G3″-F is the candidate.

## B. Execution (only after explicit approval)
1. Confirm a clean tree: `git status --porcelain` shows nothing.
2. Record the approval: write the approval token supplied by the user to `reports/test_eval_approval.txt`, then commit it (`Explicit approval for one-shot test evaluation`).
3. Run, once:
   ```
   .venv\Scripts\python.exe -c "from src.final_eval import run_test; run_test('<token>')"
   ```
   **Built-in guards:**
   - refuses without the token file, if the tree is dirty, or if `reports/test_results.json` exists;
   - verifies the frozen geometry and eligibility hashes and all 140 inventory hashes.

   **What it does, in order:**
   1. Computes test-record features for D4 (inputs only).
   2. Builds the test split and runs all 15 members × 5 models.
   3. **Only then** reads test PV labels for scoring and matured calibration feedback.
   4. Writes `reports/test_results.json` and `experiments/final/test_predictions_v1.npz`.

   **Expected runtime:** about 20–30 min (GPU).
4. **Immediately** commit `reports/test_results.json`, with the hash of the predictions file, unchanged.
5. Write the test report from `test_results.json`:
   - H1 (sole primary) decided by its pre-registered rule;
   - secondary hypotheses reported under their rules;
   - every stratum and per-day result reported.

## C. If something fails during execution
- **Crash before labels are read** (feature, inference or assembly stage): fix only the crash cause, document it as a deviation, and rerun from the frozen commit plus the fix.
- **Crash after labels are read:**
  - document it;
  - rerun the **identical code** only if the fix does not touch any model, rule or score;
  - otherwise stop and report.
- **Never:** change models, thresholds, calibration W, hypotheses or scoring after unsealing, or treat the test set as another validation fold.
