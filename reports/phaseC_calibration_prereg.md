# Phase C: calibration under temporal shift and in ramp regimes (pre-registration, FROZEN before execution)

2026-10-23. **Development data only.** The SKIPP'D test set and SkyGPT's external test have each been used once and are **not** used here. Any method found better here can only be confirmed on a future sealed set (Phase E). Phase C cannot change the Gate 5 or Phase A results.

## Motivation (observed so far)
- Causal rolling conformal restored nominal *marginal* coverage on development folds (E4-CAL2).
- Ramp-conditional 90% coverage stays about 0.65 on dev (0.75 on the SKIPP'D test, 0.46 on the SkyGPT external days).
- On the external cloudy winter days, overall 90% coverage was only 0.77.
- The development data show a season-adjusted decline of about 2.2%/yr in clear-day PV relative to a fixed clear-sky model (Figure F4).

**Question:** do calibration schemes that condition on a *forecast-time* regime signal, or that adapt faster, improve ramp-conditional coverage without hurting overall interval quality?

## Data
- **Out-of-fold development streams.** Each sample is predicted only by the 3 members (seeds × its own fold) that never trained on it.
- **Models:** G3″-F (with its frozen routing) and E4-B-fit, over the full development validation set (164,362 samples, 497 days, 5 blocked folds).
- **Stored per sample:** the 9 quantiles, the outcome, the origin and target times, the day and fold, the ramp label (frozen τ and anchor rule), and two forecast-time regime signals:
  - **spread**: the forecast's own 10–90% width in kW;
  - **g**: the event probability implied by the E6-S expert, averaged over members, from the frozen gate feature code.
- Both signals are known at issue time.

## Methods
All methods are causal: an outcome enters only once its target time ≤ the current issue time (15-min maturation). All share the frozen raw quantile bounds.

| Code | Method |
|---|---|
| M0 | Uncalibrated (raw quantile bounds) |
| M1 | Rolling conformal, **daily reset** (the frozen Gate 5 method). Each day is initialised with the W scores nearest in time from **other folds** only |
| M2 | Rolling conformal, **continuous** within a fold (no daily reset). Initialised once at the fold start from other folds |
| M3 | ACI (Gibbs & Candès 2021), continuous, with a fixed score set of the W other-fold scores nearest the fold start |
| M4 | DtACI-style aggregation: ACI experts for γ ∈ {0.001, 0.005, 0.01, 0.02, 0.05}, combined by an exponentially weighted average of their levels using the pinball loss of each expert's level. η and σ are set by the rule in the implementation note below, before any result |
| M5 | **Mondrian rolling by spread**: separate rolling windows for 3 regime bins, cut at terciles of spread computed on the other folds. Continuous, each bin initialised from other folds |
| M6 | **Mondrian rolling by event probability g**: as M5 with terciles of g |

**Hyperparameters.** W comes from {330, 990, 2310, 4620}, the per-bin W for M5/M6 from the same grid, and γ for M3 from {0.001, 0.005, 0.01, 0.02, 0.05}.
- They are chosen **leave-one-fold-out**: for evaluation fold k, select on the other 4 folds' streams.
- **Criterion:** the sum over the 50/80/90% levels of the mean interval score. Ties go to the less adaptive setting.
- Selection never uses fold k.

**Implementation note for M4 (fixed now):**
- Expert levels start at α.
- Each matured outcome updates every expert: α_k ← clip(α_k + γ_k(α − err_k), 0, 1).
- Weights are w_k ∝ exp(−η · cumulative pinball loss of the issued score quantile at level 1 − α_k), with η = 0.1 / α and a σ = 1/(2K) uniform mixing floor.
- The issued level is Σ w_k α_k.

## Outcomes
- **Primary outcomes:**
  - (a) **ramp-conditional 90% coverage**, the main target;
  - (b) **overall 90% interval score**, the guard.
- **Secondary outcomes:**
  - 50/80/90% coverage and width by stratum (overall, sunny, cloudy, ramp, non-ramp, fold);
  - interval scores;
  - lower/upper miss rates on up- and down-ramps;
  - monthly coverage over time.

**Inference.** Paired day-block bootstrap (10,000 resamples, stratified sunny/cloudy, seed 20261022) of each method minus M1.

## Pre-registered hypothesis (G3″-F stream; E4-B-fit stream is a replication)
- **C1:** the better of M5 and M6 (selected by the *same* leave-one-fold-out criterion, never by ramp coverage) raises ramp-conditional 90% coverage over M1 by **≥ 0.05**, with a bootstrap CI lower bound > 0.
  - **Guard:** its overall 90% interval score must not be worse than M1's by more than 2% (point estimate).
  - **Verdict:** confirmed only if both hold.
- **Descriptive:** M2/M3/M4 vs M1 (continuous adaptation vs daily reset). This is the question raised by the external-day under-coverage. No confirmatory claim.

## Explicit limits
- Ramp labels use outcomes, so they serve only for *evaluation strata*, never as a conditioning variable.
- Leave-one-fold-out selection uses data from later folds for earlier folds. This is acceptable for hyperparameter choice on dev, and is stated in the report.
- No test or external data are touched. A method adopted from this study needs a new sealed evaluation.
