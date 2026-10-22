# Sealed test-set evaluation: pre-registration DRAFT

**Status: DRAFT, 2026-10-22. Not frozen. The official SKIPP'D 20-day test set remains sealed.**
- **Seal:** no model output on test images or test PV has been computed for any model developed in this project. (The SUNSET port and the persistence baseline were scored on the test split at Gate 0, for benchmark reproduction only. They involve no model selection.)
- **Only inspection so far:** test-sample **timestamps**, used to count 60-min eligibility: 12,355 / 13,266 = 93.1%.
- **Freezing:** after approval, this document is frozen together with the exact code commit. Unsealing is a **single, explicitly approved execution** whose outputs are committed immediately. No model, threshold, rule or analysis may change after unsealing. Deviations, if forced (e.g. a crash), are documented, and the run restarts from the frozen commit.

## 1. Forecasters (all frozen; no retraining)
| Name | Definition | Role |
|---|---|---|
| **G3′-F (primary)** | G3′ gate (features g, d) blending **D2** (60-min direct-K expert) and **E6-S** (residual mixture expert). Where the 60-min window is unavailable: G3 (E4-B-fit + E6-S, same gate recipe). Then causal rolling conformal for intervals. | candidate forecaster of record |
| E4-B-fit | direct-K quantiles, 15 min | main comparator (ramp reference) |
| G4 | E7b combined-feature gate (E4-B-fit + E6-S) | best overall dev model |
| E6-S | residual mixture expert alone | calm-condition expert |
| SUNSET port (E1) | 10-fold ensemble, as at Gate 0 | literature baseline (point) |
| Smart persistence | official CSI persistence (Relative_op_func) | reference |

**Ensembling (proposed):** each learned forecaster is the **15-member ensemble** (5 development folds × 3 seeds) of its fold models, combined by **averaging quantiles level by level** (Vincentisation, which preserves monotonicity).
- **Per-member pipeline:** each member uses its own clear-sky anchor (fit days), K inputs, expert checkpoints and gate parameters, exactly as at validation.
- **Rationale:** this mirrors SUNSET's fold ensemble, needs no new training, and keeps every member's calibration days out of sample for that member.

## 2. Calibrated intervals on the test stream (proposed)
- **Method:** causal rolling-window conformal, **per member**, with W as selected on that member's calibration days.
- **Initialisation:** for each test day, the member's window starts from the W calibration-day scores **closest in time** to that day (development-period data only). It is then updated causally within the day, using only outcomes observable at issue time.
- **Reported intervals:**
  - the per-member coverage, width and interval score, as mean ± SD over the 15 members;
  - for the ensemble, the **average of member interval endpoints**, a documented heuristic with no coverage guarantee.
- **Fixed CQR** (per member, calibration days) is reported alongside.

## 3. Test-set strata and labels
- **Official labels:** sunny (10 days) and cloudy (10 days), as defined by SKIPP'D.
- **Ramp label (fixed before unsealing):** |K(t+15) − K(t)| ≥ τ_test. Here K uses the clear-sky anchor fitted on **all development days**, and τ_test is the 90th percentile of |ΔK₁₅| over **all development samples**. This is the Gate-2 definition applied to the full development set.
- **Sunrise subgroup:** test samples not eligible for the 60-min window (6.9%), where G3′-F uses its fallback.

## 4. Metrics (all in kW; same scorers as development)
- **CRPS** (`crps_quantile_interp`, 9 levels) for all probabilistic forecasters; point forecasts are scored as point masses (CRPS = MAE).
- **RMSE and MAE** of the median, plus RMSE of the predictive mean, for comparability with SUNSET.
- **Forecast skill vs smart persistence:** 1 − RMSE/RMSE_pers and 1 − CRPS/CRPS_pers.
- **Coverage, width and interval score** at 50/80/90%, after rolling conformal and after fixed CQR.
- **Reporting strata:** overall, sunny, cloudy, ramp, non-ramp, sunrise subgroup, and per test day.
- **Day-level uncertainty:** paired bootstrap over test days, stratified 10 sunny / 10 cloudy, 10,000 resamples.
  - **Caveat:** with 20 days the CIs are wide.
  - **Note on the test design:** the test set is deliberately balanced and is **not** a natural weather-frequency sample.

## 5. Confirmatory hypotheses (stated before unsealing; one primary)
**H1 (primary):** G3′-F has lower overall CRPS than E4-B-fit on the test set. It is confirmed iff the paired test-day bootstrap 95% CI of CRPS(G3′-F) − CRPS(E4-B-fit) lies entirely below 0.

**Secondary hypotheses (reported with CIs; no multiplicity claim):**
- **H2:** G3′-F CRPS < smart-persistence CRPS (point mass), overall and on cloudy days.
- **H3:** ramp CRPS(G3′-F) ≤ ramp CRPS(E4-B-fit) + 0.02 kW (the E7 ramp-protection criterion).
- **H4:** the RMSE skill of G3′-F's median vs smart persistence exceeds the SUNSET port's.
- **H5:** after rolling conformal, G3′-F's pooled 90% coverage lies within ±3 percentage points of nominal.

**Reported regardless:** all comparators, all strata, ramp-conditioned coverage, and per-day results.

## 6. Pre-unsealing checks (must pass; committed before execution)
1. **Dry run on development data:** the exact test-inference code, pointed at the development validation split, reproduces the stored fold-validation predictions of every member (D2, E6-S, E4-B-fit, G3′, G4) to within 1e−5 kW.
2. **Test-pipeline unit tests:**
   - test windows (H15/H60) are built only from test-group records at or before the origin;
   - fallback assignment follows the timestamp-only eligibility;
   - the ensemble averages quantiles correctly and never crosses them;
   - rolling conformal on the test stream is causal.
3. **Code and data hashes recorded:** code commit, frozen geometry, eligibility, feature store and all checkpoints.
4. **One command** produces every test number and writes `reports/test_results.json`. It refuses to run if the working tree is dirty or `reports/test_results.json` already exists.

## 7. Compute
- **Inference only:** 15 members × 4 learned forecasters on 13,266 samples, about 10–15 min on the GPU, with no training.
- **Analysis and bootstrap:** under 10 min of CPU.

## 8. Decisions needed before freezing
1. **Ensemble definition:** a 15-member quantile-averaged fold × seed ensemble (proposed), or alternatives such as a single seed or median ensembling.
2. **Interval reporting:** per-member rolling conformal with averaged endpoints for the ensemble (proposed), or reporting member-level intervals only.
3. **Test ramp definition:** a full-development anchor and threshold (proposed).
4. **Primary hypothesis:** H1, G3′-F vs E4-B-fit overall CRPS (proposed). An alternative would be ramp CRPS, or a co-primary with multiplicity control.
5. **Comparator set:** as listed. Should D4 (best dev ramp model) be added as a descriptive ramp comparator?
