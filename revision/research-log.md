# Revision log

## 2026-11-12
- Review received (v5, major revision). Plan in `revision/PLAN.md`.
- Scope decisions by the author: Git LFS for binaries; post hoc test analyses allowed (labelled); E3 as a SKIPP'D chronological retrain; full E4; declarations left as placeholders.
- Baseline committed (6a1ced3) and tagged `pre-v5-revision`.
- **C07 pre-check (before any rerun):** `src/final_eval.py::nearest_init` selects the W development scores nearest in time with `abs()`, i.e. from both sides of the test day. Estimated with the median first-forecast time of day (07:10), 47% (range 5??6%) of each test day's initial G3??F scores (W = 4,620) come from later development days. This affects only calibrated intervals (H5, Table 6 coverage), not CRPS.

## 2026-11-13
- Author instruction: the work is SKIPP'D only; Gatton is out of scope for the revision and the manuscript. No Gatton analyses are run or reported.
