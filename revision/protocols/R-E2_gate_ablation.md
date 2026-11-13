# Protocol R-E2: gate feature and constant-weight ablation

Committed before any run. Labels: **development** (primary), **post hoc** (test).

## Question
Does the fitted disagreement gate improve on constant blending of the same frozen experts, and which gate feature carries the gain?

## Frozen inputs
A2 (`experiments/E9/v1/A2/csi`) and E6-S (`experiments/E6/v1/E6-S/csi`) per seed × fold `preds_cal.npz` / `preds_val.npz`, common 60-min population (as `src/freeze_g3pp.py`); G4 from E4-B-fit + E6-S (`experiments/E7b_gate/v1`). Frozen event thresholds `reports/phase6_frozen_thresholds.json`.

## Methods (each fitted per seed × fold on calibration days only, including λ selection)
| ID | Blend of A2 and E6-S quantiles |
|---|---|
| B50 | w = 0.5 |
| BC | single constant w minimising mean calibration CRPS (bounded scalar search on [0, 1], tolerance 1e−6) |
| Gg | logistic gate, z = [g] (event-probability only) |
| Gd | logistic gate, z = [d] (disagreement only) |
| G3″ | proposed, z = [g, d] (must reproduce E9 Stage G to 1e−9; otherwise STOP) |
| G4 (reference) | frozen E7b gate on E4-B-fit + E6-S, same samples |

Gate fitting uses `src/gate7b.select_and_fit` unchanged (λ grid, chronological halves, L-BFGS).

## Metrics
Raw CRPS overall, sunny, cloudy, ramp, non-ramp (Gate 2 ramp label); seed mean ± SD; per-seed paired day-block bootstrap (stratified sunny/cloudy) of G3″ − each method; fitted w / coefficients summary; mean w by decile bins of forecast-time g and of d (validation samples).
**Post hoc test:** 15-member ensembles of each method on the 60-min-eligible test samples (inference dump, `R-POSTHOC`); paired test-day bootstrap.

## Decision rule (development)
The gate "adds value over constant blending" if G3″ − BC ramp CRPS has CI < 0 in ≥ 2/3 seeds and overall difference ≤ +0.005 kW. A feature "carries the gain" if dropping it (Gg or Gd vs G3″) worsens ramp CRPS with CI > 0 in ≥ 2/3 seeds.
