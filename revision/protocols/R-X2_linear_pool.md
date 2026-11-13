# Protocol R-X2: gated linear pool (probability mixture) control

Committed before any run. Labels: **development** (primary), **post hoc** (test).
Origin: review C04 ("a weighted average of corresponding quantiles is not the same distribution as a probability mixture") and the E2 extension listed as not done in the v7 response.

## Question
Does combining the two experts as a probability mixture (gated linear pool) instead of a quantile blend change the result, with the same gate inputs?

## Frozen inputs
As `R-E2_gate_ablation.md`: A2 and E6-S per seed x fold `preds_cal.npz` / `preds_val.npz`, common 60-min population; frozen G3'' gate parameters; frozen event thresholds. Post hoc: per-member test dump (`experiments/R_posthoc`), 60-min-eligible test samples.

## Distributions
Each expert's predictive distribution is its piecewise-linear quantile function on the 200-point CRPS midpoint grid (`gate7b.grid_quantiles`; the same representation in which the quantile blend is fitted), i.e. 200 equally weighted atoms.
- Quantile blend (QB): Q = w Q_A2 + (1 - w) Q_E6S (the proposed G3'').
- Linear pool (LP): F = w F_A2 + (1 - w) F_E6S (mixture of the two atom sets with weights w and 1 - w).

## Methods (fitted per seed x fold on calibration days only)
| ID | Combination | Gate |
|---|---|---|
| G3'' | QB | frozen G3'' parameters (reproduction check vs R-E2 to 1e-9; otherwise STOP) |
| LP-frozen | LP | frozen G3'' weights w unchanged |
| LP-refit | LP | logistic gate on z = [g, d], fitted by minimising the exact mixture CRPS (closed form, quadratic in w per sample: E\|X - y\| - 0.5 E\|X - X'\| over the atoms) + L2 on beta, lambda chosen on the chronological half split exactly as `gate7b.select_and_fit` (same lambda grid, standardisation, L-BFGS from zero) |
| LP-const | LP | single constant w minimising the calibration mixture CRPS (bounded scalar search on [0, 1]) |

## Metrics
Primary: CRPS of the 9 nominal quantiles (LP quantiles by numerical inversion of the mixture CDF, linear interpolation between atoms), scored with `crps_quantile_interp` like every other model. Secondary: exact atom-grid CRPS of QB and LP (same 200-point representation for both). Strata overall, sunny, cloudy, ramp, non-ramp; seed mean; per-seed paired day-block bootstrap of G3'' - each LP method. Raw 90% interval coverage and width (5%-95%) for QB and LP. Post hoc test: 15-member ensembles (member quantiles averaged as everywhere else), paired test-day bootstrap.

## Interpretation rule
LP is "no better" than QB if G3'' - LP-refit has CI <= 0 or covering 0 overall in >= 2/3 seeds; it is "better" if CI > 0 overall in >= 2/3 seeds. Reported whichever way it falls; the paper's combination is not changed post hoc.
