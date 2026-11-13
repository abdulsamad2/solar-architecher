# Protocol R-C07: past-only conformal initialisation (C07, C19)

Committed before any rerun. Label: **post hoc** (test), **development** (C19 sensitivity).

## Question
1. Did the frozen calibration (Gate 5, H5) use initial conformity scores dated after the forecast day? (C07)
2. What are the calibrated interval results when the initial scores are strictly past-only? (C07)
3. How much of each day's conformal window is still made of initial (development) scores, and does the 3- vs 15-member ensemble mismatch explain the over-coverage? (C19)

## Inputs (frozen, unchanged)
- `experiments/final/test_predictions_v1.npz` (15-member test quantiles), `experiments/final/dev_oof_calibration_v1.npz` (+ `.json`, W per model).
- Scoring and conformal code: `src/final_eval.py` (`rolling_conformal_daily`, `base_interval`), `src/cal2.py`.

## Procedure
1. **Audit:** for each test day and model, count the initial scores whose development origin time is later than the day's first forecast origin (`nearest_init` rule). Also count those whose target time is later than the first origin (not matured).
2. **Reproduce** the frozen H5 numbers exactly (coverage/width/interval score at 50/80/90, all strata) with the frozen `nearest_init`. Mismatch > 1e−9 = STOP.
3. **Past-only rerun:** identical except initial scores = the W development scores with the latest *target* time strictly before the day's first forecast origin (matured before issuance). Daily reset, same W, same update rule.
4. **Window composition (C19):** for each test day, the share of the window made of initial scores after each forecast (min, median, end of day).
5. **Ensemble-size sensitivity (C19):** member-level test predictions (`R-POSTHOC` inference dump) → five 3-member ensembles (seeds 42/123/456 of fold k). Calibrate each with the same past-only initial scores; compare raw and calibrated coverage with the 15-member ensemble.

## Outputs
`revision/results/c07_conformal_pastonly.json`; report `revision/reports/C07_C19_conformal.md`.

## Interpretation rule
The frozen H5 result stays as recorded. Manuscript reports both and states which is the strictly causal result. No claim that the ensemble mismatch causes over-coverage unless step 5 shows it.
