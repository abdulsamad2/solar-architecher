# E7-BLEND pre-registration: post-hoc gated mixture of experts (no retraining)

Fixed 2026-10-21, before any blend was evaluated on fold-validation data.
- **Preserved unchanged:** Gate 2 (NOT MET), E4-CAL2 and Gate 3 (NOT RETAINED).
- **Test set:** the official 20-day test set stays sealed.

## Question
Gate 3 showed a replicated trade-off:
- **calm conditions:** the residual mixture model (E6-S) is best;
- **ramps:** direct clear-sky-index quantiles (E4-B-fit) are best.

Can a gate driven by a forecast-time event probability combine them, keeping close to the best of both, and does *gating* add value beyond plain ensembling?

## Inputs (reused unchanged; seeds paired 42/123/456; 5 blocked folds)
- **Ramp expert:** E4-B-fit, `experiments/E4_calib/v1/csi/seed{s}`.
- **Calm expert:** E6-S, `experiments/E6/v1/E6-S/csi/seed{s}`.
- **Splits:** both experts were trained on the same fit days and predict on the same calibration days and fold-validation days. Identical sample IDs are asserted per split.
- **Gate feature** g ∈ (0, 1): the calm expert's quantile-implied probability P(|K(t+15) − K(t)| ≥ τ).
  - Computed with `src.events.implied_event_probability` in E6-S's own (fit-day) K space, with τ = the frozen fold training-label threshold.
  - It is available at forecast time and contains no outcome information.

## Blend methods (all fitted per seed × fold on CALIBRATION days only; applied to validation)
The blend is quantile averaging at the 9 levels: Q(τ) = w · Q_ramp(τ) + (1 − w) · Q_calm(τ). It is non-crossing by construction.
- **B0, constant blend (control):** w = constant ∈ {0.00, 0.05, …, 1.00}, chosen to minimise mean calibration CRPS.
- **B1, gated blend (primary):** w = sigmoid(a + b · logit(clip(g, 1e−4, 1 − 1e−4))), with b ≥ 0 so that higher event probability means more weight on the ramp expert.
  - Grid: a ∈ {−8.0, −7.75, …, 4.0}, b ∈ {0.0, 0.1, …, 3.0}.
  - Chosen to minimise mean calibration CRPS. b = 0 reproduces B0.
- **B2, hard switch:** use the ramp expert iff g ≥ θ. θ is chosen from the calibration-day g percentiles {50, 55, …, 99}, minimising calibration CRPS.
- **Ties:** go to the simpler or less ramp-weighted option (smaller b, then smaller a; larger θ).

CRPS everywhere uses `crps_quantile_interp`, the same scorer as E4–E6.

## Evaluation (fold-validation samples, identical across all methods)
- **Raw CRPS:** overall, sunny, cloudy, ramp, non-ramp (Gate 2 ramp stratum), per fold, and fold 1 Sep–Oct 2017.
- **Methods compared:** both experts, B0, B1 and B2.
- **Paired day-block bootstrap 95% CIs** (stratified sunny/cloudy) per seed, plus seed mean ± SD across seeds.
- **Calibrated intervals (secondary):** E4-CAL2 rolling conformal (W chosen on calibration days) applied to each method's quantiles. Reported: 50/80/90% coverage, width and interval score, overall and ramp/non-ramp.
  - **Disclosure:** the blend's 2 parameters and the conformal initialisation both use calibration days.
- **Gate diagnostics:** fitted (a, b) or θ per fold, the mean ramp-expert weight on ramp vs non-ramp validation samples, and the Brier score of g.
- **Cost:** blend fitting time and per-sample application time.

## Decision rule (pre-registered)
The gated blend B1 is **retained as the forecaster of record** iff, in **at least 2 of 3 seeds**, all of the following hold:
1. **Ramp non-inferiority to the ramp expert:** ramp CRPS(B1) − ramp CRPS(E4-B-fit) ≤ 0.02 kW.
2. **Overall non-inferiority to the calm expert:** overall CRPS(B1) − overall CRPS(E6-S) ≤ 0.02 kW.
3. **Gating adds value over ensembling:** ramp CRPS(B1) − ramp CRPS(B0) < 0, with the day-bootstrap 95% CI entirely below 0.

The margin of 0.02 kW is the same as Gate 3 (same justification). Conditions 1–2 use point estimates; condition 3 uses the CI.

- **If 1 and 2 hold but 3 fails:** the result is reported as "ensembling, not gating, explains the combination" and B0 is reported instead.
- **If 1 or 2 fails:** no blend is retained.

## Stop conditions (asserted)
- The gate or blend parameters use any fold-validation label.
- Sample IDs differ between experts or across methods within a split.
- Crossing or non-finite blended quantiles.
- An inverted or non-finite calibrated interval, or a non-causal conformal update.
