# Gate 2: replication and calibration pre-registration

Fixed 2026-10-20, before any replication or calibration run. The official 20-day SKIPP'D test set is never read.

## Part A: E4 replication
- **Runs:** E4-A (kW quantiles) and E4-B (K quantiles, K inputs) with seeds 123 and 456. The configuration is identical to seed 42 (commit 51c46ee, plus logging-only changes).
- **Folds, samples and scoring:** same 5 blocked folds and validation samples; same scoring (`crps_quantile_interp`, 9 levels).
- **Saved per sample:** quantiles, target, origin and target timestamps, P_clear at origin and target, PV now, persistence, fold ID, clear/cloudy label.
- **Seed 42:** the existing seed-42 files are backfilled with the same context fields, recomputed deterministically. The predictions are unchanged.
- **Reported:**
  - per seed and pooled CRPS, median RMSE, reliability, and 50/80/90% coverage and width;
  - paired E4-B − E4-A day-block bootstrap CI per seed;
  - seed variation (mean ± SD and range across seeds), separately from the bootstrap CIs.
  - No seed is selected or dropped.

## Part B: calibration
### Data split, within each fold
- **Calibration days:** `src.folds.calibration_split` takes the fold's training days in date order, cuts them into consecutive blocks of 10 training days, and assigns block k to calibration iff k mod 5 = 2. That is about 20% of training days, spread across seasons.
- **Purge:** fit days within 1 calendar day of a calibration day are dropped.
- **Validation days:** the fold's original validation days, untouched.

### Model used for calibration
- **E4-B-fit:** E4-B retrained (`--calib-holdout`) on fit days only.
  - The clear-sky anchor and head initialisation also use fit days only.
  - Seeds 42, 123 and 456.
  - Checkpoint selection uses fold-validation CRPS, the same convention as every dev experiment so far. This is disclosed: it affects the calibrated and uncalibrated versions of the checkpoint identically.
  - Calibration-day labels are never used for fitting, checkpoint selection or tuning.
- **Stop conditions, asserted in code:**
  - fit, calibration, validation and purged days, and their samples, are mutually disjoint;
  - calibration predictions come from the checkpoint trained on fit days only.

### Methods (fitted on calibration-day predictions only, per fold and seed)
1. **CQR, kW (primary):**
   - Base intervals: 50% = [Q(.25), Q(.75)] (interpolated), 80% = [Q.10, Q.90], 90% = [Q.05, Q.95].
   - Score E = max(lo − y, y − hi) in kW.
   - q̂ = the ⌈(n+1)(1−α)⌉-th smallest calibration score. The calibrated interval is [lo − q̂, hi + q̂].
   - No clipping. Inverted intervals (lo > hi) are counted and reported.
2. **CQR, K-scaled (secondary):** as above with E / max(P_clear(target), 1 kW). The interval is widened by q̂ · max(P_clear, 1 kW).
3. **Monotone full-quantile recalibration (for CRPS):**
   - Each forecast's quantile function is linear between the 9 levels and linearly extrapolated beyond 0.05/0.95 with the outermost segment slope, on τ ∈ [0.001, 0.999]. It is floored at 0 kW.
   - The calibration curve is F̂(τ) = mean over calibration samples of 1[y ≤ Q(τ)] on a 0.001 grid, made non-decreasing by cumulative maximum.
   - Each nominal level τ* maps to τ' = min{τ : F̂(τ) ≥ τ*}. The recalibrated quantile is Q(τ'), at the original 9 levels.
   - **Stop condition:** any crossing quantile.
   - Scored with the same CRPS function as everything else.

### Evaluation (fold-validation samples only; identical across all compared models)
- **Models compared:**
  - E4-B (full training days, uncalibrated)
  - E4-B-fit uncalibrated
  - E4-B-fit + CQR-kW
  - E4-B-fit + CQR-K
  - E4-B-fit + recalibration
- **Strata:**
  - overall;
  - sunny and cloudy days;
  - ramp vs non-ramp: ramp ⇔ |K(t+15) − K(t)| ≥ τ_ramp, where τ_ramp is the 90th percentile of |ΔK| over the fold's training samples, with K from the fold's full-training-day anchor;
  - per fold, with fold 1 (Sep 2017 to May 2018, winter) reported explicitly.
- **Metrics:** coverage and mean width of the 50/80/90% intervals. CRPS and reliability for recalibration (CQR changes intervals only, so it has no CRPS).
- **Not claimed:** split conformal gives at most a marginal, approximate guarantee under exchangeability. Calibration and validation days come from different periods and minutes within a day are dependent, so no guarantee, and no conditional (per-stratum) coverage guarantee, is claimed.

### Gate 2 closing rule
Gate 2 closes for the selected representation (E4-B) iff, with the primary method (CQR-kW), in at least 2 of 3 seeds:
- pooled validation coverage is within ±3 percentage points of nominal at 80% and 90%, and within ±5 pp at 50%; and
- no fold's 90% coverage falls below 80%.

Otherwise Gate 2 remains open, and the report states which condition failed.

## Computational cost (reported)
- Training wall time and epochs per run, from the logs.
- Inference time per sample, batched and at batch size 1, measured on the same GPU.
- Calibration fitting and application time.
