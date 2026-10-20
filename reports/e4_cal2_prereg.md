# E4-CAL2 pre-registration: online adaptive calibration under temporal shift

Fixed 2026-10-20, before any E4-CAL2 validation outcome was computed or examined.
- **Gate 2:** its report (`reports/gate2_report.md`) and its **NOT MET** verdict are preserved unchanged. E4-CAL2 is a follow-up experiment and does not re-open or re-score Gate 2.
- **Test set:** the official 20-day test set is not read.

## Objective
Determine whether online adaptation of interval width improves fold-validation coverage under temporal distribution shift, especially in the fold-1 winter period (Sep 2017 to May 2018). No retraining: the existing E4-B-fit predictions are reused.

## Inputs (reused, unchanged)
- `experiments/E4_calib/v1/csi/seed{42,123,456}/fold_{0..4}/preds_cal.npz` and `preds_val.npz` (E4-B-fit, trained on fit days only).
- Evaluation samples: the same 164,362 fold-validation samples as Gate 2 (asserted).

## Methods (all intervals in kW; base interval = E4-B-fit [Q_lo, Q_hi] at 50/80/90% as in Gate 2)
Conformity score: s = max(lo − y, y − hi) in kW.
1. **Uncalibrated:** [Q_lo, Q_hi].
2. **Fixed CQR-kW (Gate 2 primary):** the fold's full calibration score set, with q̂ at rank ⌈(n+1)(1−α)⌉.
3. **ACI** (adaptive conformal inference, Gibbs & Candès 2021), per interval level separately:
   - α_0 = α; the score set is the fold's full calibration set.
   - The interval at time t uses q̂_t = the empirical (1 − α_t) quantile of the score set, with the level clipped to [0, 1].
   - Update after each outcome becomes observable: α_{t+1} = α_t + γ(α − err), where err = 1 if y is outside the interval that was issued for it.
4. **Causal rolling-window conformal**, per level:
   - q̂_t = rank-⌈(n+1)(1−α)⌉ score among the most recent W *observable* scores.
   - Initialisation: the W calibration samples closest in time to the start of the validation period (training-period data only).
   - Each newly observable validation score is pushed in and the oldest is dropped.

**Width floor:** if q̂ < −(hi − lo)/2, the interval is set to the degenerate point (lo+hi)/2, and the count is reported. An interval with lo > hi after this rule, or any non-finite endpoint, is a STOP condition.

## Chronological replay (causality)
- Validation samples in each fold are processed in increasing **origin time**.
- The forecast for sample i is issued at origin_time_i and its outcome becomes observable at target_time_i, the actual matched target record time.
- Before issuing forecast i, every pending forecast j with target_time_j ≤ origin_time_i is resolved. Its miss indicator / score updates the state, in target-time order.
- Forecasts whose targets are not yet observed never influence the state.
- **STOP:** any update uses an outcome with target_time > the current issue time. This is asserted per update, and a full trace is saved.
- **State boundaries:** state is carried within a fold, across days, and is never shared between folds or seeds.

## Hyperparameter selection (calibration data only)
- **Grids:** ACI γ ∈ {0, 0.001, 0.005, 0.01, 0.02, 0.05}. Rolling W ∈ {330, 990, 2310, 4620} samples (≈ 1, 3, 7, 14 days of 2-min forecasts).
- **Procedure, per seed × fold × method:**
  - The fold's calibration samples (training-period days only) are sorted by origin time and split into a first half (initialisation set) and a second half (replay set).
  - Each candidate is replayed causally on the second half, initialised from the first half.
  - The chosen value minimises the **sum over the 50/80/90% levels of the mean interval (Winkler) score**: IS_α = (hi − lo) + (2/α)(lo − y)₊ + (2/α)(y − hi)₊.
  - One value is used for all three levels. Ties go to the smaller γ / larger W (less adaptive).
- **Selection never uses fold-validation labels.** This is enforced structurally: the selection function receives calibration arrays only, and the chosen values are written to disk before the validation replay runs.

## Metrics (fold-validation samples)
- **Coverage and mean width** at 50/80/90%:
  - overall;
  - per fold;
  - sunny / cloudy (day label);
  - ramp / non-ramp (Gate 2 definition: |ΔK₁₅| ≥ the fold's training 90th-percentile threshold).
- **Mean interval score** at each level (a proper score for central intervals; the primary comparison metric).
- **Winter fold (fold 1) 90% coverage**, reported separately.
- **Time-resolved coverage:** 90% coverage per calendar month within each fold, and per 7-day window for fold 1 (with a figure).
- **Paired comparisons vs fixed CQR-kW:** per seed, the difference in mean interval score and in 90% coverage, with 95% day-block bootstrap CIs (stratified by sunny/cloudy).
- **CRPS:** reported only for methods that define a coherent full predictive distribution. Here that is the uncalibrated E4-B-fit only. CQR, ACI and rolling conformal define intervals, not distributions, so no CRPS is computed for them.

## Decision rule (recommendation on deployment use)
- **Retain:** an adaptive method is recommended as a deployment-time calibration component iff, in at least 2 of 3 seeds:
  - it lowers the pooled mean interval score vs fixed CQR-kW at the 90% level, with a day-bootstrap CI below 0; and
  - fold-1 90% coverage improves vs fixed CQR-kW.
- **Choice between methods:** if both qualify, the one with the lower seed-mean pooled interval score is recommended.
- **No guarantee claimed:** ACI's long-run average coverage result assumes nothing about the data distribution but is asymptotic. No finite-sample or conditional guarantee is claimed.

## Stop conditions (asserted)
- An online update uses an outcome not yet observable at the update time.
- Adaptive hyperparameters are selected using fold-validation labels.
- Evaluation samples differ across compared methods.
- Interval endpoints are non-finite or inverted.

## Preserved artefacts
Per seed × fold, in `experiments/E4_cal2/v1/`:
- chosen hyperparameters;
- per-sample intervals for every method;
- the chronological update trace: issue time, observation time, α_t or q̂_t, and the miss indicator.

The code version (git commit) is stored with the results.

## Amendment 1 (2026-10-20, before any validation outcome was computed)
- **What changed:** ACI's α_t is now **clamped to [0, 1] after every update** (anti-windup).
- **Why:** a unit test with a synthetic 3× noise shift showed the problem with the original rule. Under the pre-registered level clipping, q̂ cannot exceed the largest calibration score, yet unclamped α_t kept integrating (to −29). That "integrator windup" would delay recovery after the shift ends.
- **Behaviour now:**
  - ACI remains bounded by the calibration score range and still improves on fixed CQR under shift.
  - It returns to nominal coverage after a shift reverses.
  - The rolling-window method, which absorbs new scores, recovers nominal coverage under large shifts.
  - All of this is covered by tests (`tests/test_cal2.py`).
- **Unchanged:** grids, selection, metrics and decision rule.
