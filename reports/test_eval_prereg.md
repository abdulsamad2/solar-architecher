# Sealed SKIPP'D test evaluation: pre-registration (FROZEN)

**Frozen 2026-10-23**, before the development-only dry run. It supersedes `test_eval_prereg_DRAFT.md`, which is kept for history.

**The official 20-day test set remains sealed.**
- **What has been touched:** only test timestamps, for 60-min eligibility (12,355 / 13,266 = 93.1%).
- **No exception for project models:** no project model has produced a test prediction, and no test PV or test image has been read by project code. (The SUNSET port and the persistence baseline were scored on the test split at Gate 0, for benchmark reproduction only.)
- **Execution requires** a recorded explicit approval token (`reports/test_eval_approval.txt`), a clean git tree, verified hashes, and the absence of `reports/test_results.json`. It runs once.

## 1. Models (frozen; no retraining or tuning on test data)
| Name | Definition | Population |
|---|---|---|
| **G3″-F (primary candidate)** | G3″: frozen disagreement gate (features g, d) blending **A2** (60-min 3D-CNN + GRU direct-K) and **E6-S** (residual event mixture); gate parameters `experiments/E9/v1/G3pp_params`. **Fallback G3** (E4-B-fit + E6-S, frozen E7b G3 gate) when a valid 60-min history is unavailable. **Routing uses only historical input availability** (timestamp rule: 16 records at t − 4j min, ±30 s, same day, same split). | full test set |
| E4-B-fit | direct-K 15-min quantiles (primary comparator) | full |
| E6-S | residual event-mixture expert | full |
| G4 | E7b combined-feature gate (E4-B-fit + E6-S) | full |
| SUNSET port | E1 10-fold ensemble (stored Gate-0 test predictions; point forecast) | full |
| Persistence | official CSI persistence (Relative_op_func) | full |
| D4 (descriptive) | 60-min + solar/motion direct-K | 60-min-eligible test samples only; never used to select or replace the primary model |

**Model inventory:** every learned model is the 3 seeds × 5 development folds = **15 members**. Checkpoints, configurations, standardisation statistics and gate parameters are listed with SHA-256 in `reports/checkpoint_inventory.json`, produced by the dry run and verified at execution.

**Selection history (preserved):** E9 screen, confirmation and gate (`reports/e9_report.md`), including the negative findings (L1 null; A1 not confirmed; A3 ineligible). G3′ is superseded by G3″.

## 2. Ensemble rule (frozen)
- **Members:** each member reconstructs its exact inputs:
  - its clear-sky anchor (fitted on its fold's fit days);
  - K history;
  - its window (H15, or H60 for A2/D4);
  - for D4, solar/motion features standardised with its stored fit-day statistics;
  - its frozen checkpoint and, for gated models, its gate parameters.
- **Member output:** 9 quantiles of K(t+15), converted to kW by multiplying by max(P_clear(target), 1 kW).
  - E6-S quantiles come from its mixture CDF (60-step bisection), so every model is scored on the same 9 levels with the same function.
  - **No physical clipping** is applied to any model, as in development.
- **Ensemble:** equal-weight, **level-wise quantile averaging** over the 15 members. A non-crossing check is asserted. The same rule applies to E4-B-fit, E6-S, G4, G3″-F and D4.
- **Validation status:** on development data, a 15-member ensemble prediction cannot be out-of-sample, because every development day is a fit day for some members.
  - **(i) Numerical reproduction:** members recompute their stored fold-validation predictions within **1e−4 kW** (max \|diff\|). This is a code-correctness test, **not** validation.
  - **(ii) Out-of-fold surrogate:** the genuinely held-out estimate is the 3-member ensemble (the members whose validation fold contains the sample).

## 3. Calibration (empirical adaptive conformal; no finite-sample coverage guarantee)
- **Scores:** conformity scores of each model's **development out-of-fold surrogate** (§2) at 50/80/90%. They are frozen in `experiments/final/dev_oof_calibration_v1.npz`, with a hash in the inventory.
- **Window length W:** chosen per model by the E4-CAL2 `select_param("rolling")` procedure on the chronological development out-of-fold stream. Frozen by the dry run.
- **Initialisation:** at each test day, the window starts with the W development out-of-fold scores **closest in time to that day's first forecast** (development data only).
- **Feedback maturation:** a forecast's score enters the window only once its target time (origin + 15 min, matched record) is ≤ the current issue time.
- **Day boundaries:** the window is **reset at every test day** (the 20 days are disconnected). No state carries across days.
- **Within-day gaps:** forecasts are processed in origin order, matured outcomes are applied before each issue, and missing minutes need no special handling.
- **Mismatch disclosure:** the scores come from 3-member surrogates, while the test ensemble has 15 members. Calibration is therefore described as **empirical adaptive calibration** with no finite-sample or conditional coverage guarantee.

## 4. Labels and strata
- **Target:** PV at the record matched to origin + 15 min (±30 s), per the frozen manifest.
- **Sunny / cloudy:** official SKIPP'D test-day lists (10 / 10).
- **Ramp:** |K(t+15) − K(t)| ≥ τ_test.
  - K = PV / max(P_clear, 1 kW), with the clear-sky anchor fitted on **all development days**: coefficients 0.026009, 0.001471, 0.001611 (annual harmonic).
  - **τ_test = 0.279483**, the 90th percentile over all development samples.
  - Frozen in `reports/test_ramp_definition_frozen.json`.
- **Sunrise ramps:** ramp samples without a valid 60-min history (routed to the fallback).
- **Also reported:** non-ramp samples and per-day results.

## 5. Hypotheses and testing rules
**H1 (sole primary):** overall raw (uncalibrated) CRPS of G3″-F is lower than E4-B-fit's.
- **Confirmed iff** the paired **test-day bootstrap** 95% CI of the mean per-sample difference lies entirely below 0.
- **Bootstrap:** stratified 10 sunny / 10 cloudy days, 10,000 resamples, seed 20261022.

**Secondary hypotheses** (each reported with its own CI; no multiplicity adjustment; secondary status stated explicitly):
- **H2:** G3″-F CRPS − persistence CRPS (as a point mass) < 0, overall and on cloudy days (CI < 0).
- **H3:** ramp non-inferiority, i.e. ramp CRPS(G3″-F) − ramp CRPS(E4-B-fit) ≤ 0.02 kW. Both the point estimate and the upper CI bound are reported; non-inferiority is declared only if the upper CI bound is ≤ 0.02.
- **H4 (descriptive):** RMSE skill of the G3″-F median vs persistence, compared with the SUNSET port's skill.
- **H5:** pooled 90% coverage after adaptive calibration within ±3 pp of nominal (point estimate; descriptive).

**Reported regardless of significance:** every model, every stratum, raw and calibrated intervals, D4 on its population, and per-day results.

## 6. Development-only verification (must pass before any approval request)
1. Numerical reproduction of all stored fold predictions (4 experts × 15 members) and of the frozen G3/G4 gates, to ≤ 1e−4 kW.
2. The G3″ gate refit reproduces E9 Stage-G CRPS exactly (done: `src/freeze_g3pp.py`).
3. An out-of-fold development run of the complete G3″-F routing, including the fallback.
4. Causality and leakage:
   - windows and routing use only past timestamps;
   - gate features use only forecasts, no labels;
   - anchors and standardisation come from development fit days;
   - calibration initialisation comes from development only;
   - updates are mature only.
5. Synthetic fixtures for the ensemble, daily-reset calibration, bootstrap and execution guards (`tests/test_final_eval.py`).
6. Feature, eligibility and checkpoint hashes are verified.
