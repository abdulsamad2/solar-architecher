# Phase F: physics-informed training objective (pre-registration, FROZEN before any training)

2026-10-23.
- **Code:** `src/physics_losses.py`, `src/physics_metrics.py`; `src/train.py --physics-config`.
- **Tests:** `tests/test_physics_losses.py` (86 tests pass).

## Scope and invariants
- **Unchanged:**
  - architectures (3D-CNN image expert A2, E6-S event-mixture expert, E4-B-fit, gates);
  - 9 quantile outputs and the K target with P̂ = K̂ · max(P_clear(t+15), 1 kW);
  - folds, seeds (42/123/456), optimiser, schedule, early stopping, ensemble rule;
  - the causal conformal procedure.
- **Changed:** only the training objective of the **A2 3D-CNN quantile expert**, the G3″ ramp expert that serves about 92% of forecasts.
  - E6-S is trained with an event-mixture likelihood, not a pinball loss, so the K-quantile physics terms do not apply to it. It stays frozen.
  - The G3 fallback (E4-B-fit + E6-S) stays frozen.
  - The G3″ gate is **refitted** for every arm with the frozen procedure (`src/freeze_g3pp.py` recipe: L-BFGS on calibration days, features g and d), because its inputs change.
- **Ramp indicator used inside the loss:** the existing rule, \|K(t+15) − K(t)\| ≥ τ_fold, where τ_fold is the 90th percentile over fit-day training samples. This equals the frozen Phase 6 threshold, and the code asserts that.
- **Normaliser:** P_scale = 30 kW, the existing PV_SCALE constant (nameplate). K_max = 1.5 (conservative). The training-set 99.9th percentile of K is logged for reference.

## Data use: the official test set is NOT used
- The official 20-day SKIPP'D test set and SkyGPT's external days have each been used once. Using them to compare loss variants would be test-set reuse.
- **All comparisons in Phase F are made on held-out development data:** 3-member out-of-fold forecasts, each sample predicted only by members that never trained on it. The procedure is the same as the ablation ladder and Phase C.
- A one-shot test of the selected variant is possible only on the **sealed Gatton test** (Phase E) or another unused set. It requires a separate pre-registration.

## Staged hyperparameter search (seed 42; search folds 1 and 3 only; development validation days)
Arm A0 is the existing A2 run (seed 42, folds 1 and 3), so it costs no extra training. Metrics are the A2 expert's validation CRPS in kW from `metrics.json`, on the common population.

| Stage | Varied | Fixed | Selection rule |
|---|---|---|---|
| S1: physics | λ_phys ∈ {0.01, 0.05, 0.1, 0.25} (MAE) | – | Lowest **overall** CRPS (mean of folds 1 and 3) |
| S2: ramp | λ_ramp ∈ {0.01, 0.05, 0.1, 0.25} with α = 2; then α ∈ {1, 4} at the best λ_ramp | – | Lowest **ramp** CRPS, subject to overall CRPS ≤ 1.01 × A0's |
| S3: bounds | λ_bound ∈ {0.001, 0.005, 0.01} | best λ_phys and λ_ramp/α | Lowest overall CRPS, subject to ramp CRPS ≤ that of the S1+S2 combination |
| S4: direction (A5, optional) | λ_direction ∈ {0.01, 0.05}, margin 0.02 | best S3 | Lowest ramp CRPS, subject to overall CRPS ≤ 1.01 × A0's |

Ties go to the smaller λ. If no candidate meets a constraint, the component is set to the smallest grid value and flagged.

## Final ablation (all 3 seeds × 5 folds)

| Arm | Objective |
|---|---|
| A0 | L_quantile (existing runs, no retraining) |
| A1 | + λ_phys L_phys |
| A2-arm (named "R") | + λ_ramp L_ramp |
| A3 | + physics + ramp |
| A4 | + physics + ramp + bounds |
| A5 (optional) | A4 + direction |

## Evaluation (development out-of-fold, G3″-F routing)
- **Forecasts:** G3″-F uses G3″ (A2-variant + E6-S, gate refitted) where a 60-min history exists, and the frozen G3 otherwise.
- **Metrics, per arm:**
  - CRPS overall, sunny, cloudy, ramp, sunrise ramp, rising and falling ramps;
  - RMSE and MAE of the median, and RMSE skill vs smart persistence;
  - 50/80/90% coverage overall, ramp, sunrise ramp, rising and falling ramps;
  - interval width and score, from the frozen daily-reset rolling conformal with initial scores from other days (as in `src/final_rehearsal.py`);
  - physical diagnostics (`src/physics_metrics.py`).
- **Inference:** paired day bootstrap (10,000 resamples, stratified sunny/cloudy, seed 20261022) of best-arm − A0, for CRPS overall, cloudy and ramp, and for ramp 90% coverage.
- **Primary comparison:** the best arm, chosen by development criteria fixed here: lowest overall CRPS among A1–A5, subject to its ramp CRPS being no worse than A0's. That arm is compared against A0.
- **Reporting:** no claim of improvement unless the 95% CI excludes 0. Every metric is reported for every arm, including the ones that get worse.
