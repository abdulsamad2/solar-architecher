# E7b pre-registration: enhanced gate features for expert routing

Fixed 2026-10-21, before any E7b result was computed. No joint expert training is launched; the experts are frozen.
- **Preserved unchanged:** E7 (`e7_blend_report.md`, B1 **NOT RETAINED**), Gate 2, E4-CAL2 and Gate 3.
- **Test set:** the official 20-day test set stays sealed.

## Question
Do expert uncertainty and expert disagreement carry ramp-routing information **beyond** the current event probability?

## Frozen references
- **Direct expert:** E4-B-fit (`experiments/E4_calib/v1/csi`).
- **Residual expert:** E6-S (`experiments/E6/v1/E6-S/csi`).
- **E7 B1** (gated blend, parameters in `experiments/E7_blend/v1`).

All are seeds 42/123/456 × 5 blocked folds, with identical calibration-day and fold-validation samples (asserted).

## Gate features (forecast-time information only; each expert in its own fit-day K space)
| Name | Definition | Transform |
|---|---|---|
| g | residual expert's quantile-implied P(\|ΔK₁₅\| ≥ τ) (the E7 gate; τ = frozen fold threshold) | logit(clip(g, 1e−4, 1 − 1e−4)) |
| s_D | direct-expert 80% spread in K: (Q.90 − Q.10) / max(P_clear(target), 1 kW) | log(s + 1e−3) |
| s_R | residual-expert 80% spread in K, same definition | log(s + 1e−3) |
| d | expert disagreement: \|mean_D − mean_R\| in K. Each predictive mean is the mean of the piecewise-linear quantile function on 200 midpoint levels. | log(d + 1e−3) |

**Standardisation:** z-scores with mean and SD computed on the **fitting set only**, then applied unchanged to the selection set or validation.

## Gates (logistic link, low capacity)
Mixture: Q(τ) = w · Q_D(τ) + (1 − w) · Q_R(τ), with w = sigmoid(β₀ + βᵀz).
- **G0:** z = [g] (original gate, refitted with this procedure)
- **G1:** z = [g, s_D] (direct spread)
- **G2:** z = [g, s_D, s_R] (dual spread)
- **G3:** z = [g, d] (expert disagreement)
- **G4:** z = [g, s_D, s_R, d] (combined)

## Fitting (per seed × fold × gate; calibration days only)
- **Calibration-day predictions are out-of-sample for both experts:** the experts were trained on fit days only, and these days are disjoint from fit days and from fold-validation days.
- **Objective:** mean CRPS (kW) of the blended quantiles + λ · ‖β‖². The intercept is not penalised.
  - CRPS is computed exactly as in `crps_quantile_interp`: piecewise-linear quantile function, 200 midpoint levels, constant tails.
  - The blended quantile function is linear in w, so the objective is evaluated exactly.
- **Optimiser:** L-BFGS (float64, strong-Wolfe line search, at most 200 iterations, tolerance 1e−9), initialised at β = 0, β₀ = 0. Deterministic.
- **λ selection:** λ ∈ {1e−4, 1e−3, 1e−2, 1e−1}.
  - Calibration days are split chronologically (by origin time) into halves.
  - Each λ is fitted on the first half, with standardisation from the first half, and scored by mean CRPS on the second half.
  - The best λ is then refitted on all calibration days, with standardisation from all calibration days.
  - Ties go to the larger λ.
- **Validation labels** are never used for fitting, standardisation, λ selection or gate choice.

## Evaluation (fold-validation samples, identical across all methods)
- **Methods:** E4-B-fit, E6-S, E7-B1 (frozen), G0–G4.
- **Raw-distribution CRPS:** overall, sunny, cloudy, ramp and non-ramp (Gate 2 ramp stratum), per fold, and Sep–Oct 2017. Same scorer as E7.
- **Paired day-block bootstrap 95% CIs** (stratified sunny/cloudy) per seed, plus seed mean ± SD.
- **Gate diagnostics:**
  - w distribution (mean and 10/50/90 percentiles) on ramp vs non-ramp samples;
  - fitted λ and β per fold;
  - **ramp AUPRC** (average precision) and **ramp Brier score** of w treated as a ramp score, plus the same for g alone. The ramp base rate is reported.
- **Calibrated intervals (separate from raw CRPS):** E4-CAL2 causal rolling-window conformal (W on calibration days). Reported: 50/80/90% coverage, width and interval score, overall and ramp/non-ramp.
- **Cost:** feature computation and gate evaluation time per sample; gate fitting time.

## Decision rules (pre-registered)
**Information test, per gate Gk (k = 1…4) vs G0.** Gk "adds routing information" iff, in **≥ 2 of 3 seeds**:
1. ramp CRPS(Gk) − ramp CRPS(G0) < 0, with the day-bootstrap 95% CI entirely below 0 (**ramp protection is the primary criterion**); and
2. overall CRPS(Gk) − overall CRPS(G0) ≤ +0.005 kW (point estimate).

**Recommended gate (selection with ramp protection, not overall CRPS alone):** among gates passing the information test, the one with the **lowest seed-mean ramp CRPS**, with ties broken by lower overall CRPS. If none passes, G0 / B1 remains the reference.

**Sufficiency check against the E7 retention criteria** (same thresholds as E7). The recommended gate is said to *close the routing gap* iff, in ≥ 2/3 seeds:
- ramp CRPS ≤ ramp CRPS(E4-B-fit) + 0.02 kW; and
- overall CRPS ≤ overall CRPS(E6-S) + 0.02 kW.

**Next-step recommendation:**
- **If the routing gap closes:** improved gate information is sufficient.
- **Otherwise:** the next experiment should target ramp detection itself (longer histories, solar-relative motion features).

## Stop conditions (asserted)
- Any feature, standardisation statistic, fitted parameter or λ derived from fold-validation labels.
- Sample IDs differ between experts or across methods within a split.
- Non-finite or crossing blended quantiles; non-finite gate weights.
- An inverted or non-finite calibrated interval, or a non-causal conformal update.
- The torch CRPS used in fitting disagrees with `crps_quantile_interp` by more than 1e−8 on the fitting data (checked per fit).
