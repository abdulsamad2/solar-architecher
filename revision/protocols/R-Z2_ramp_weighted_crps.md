# Protocol R-Z2: proper ramp-weighted CRPS for the ramp comparisons

Committed before any run. Label: **post hoc** (official test, category ii). Frozen predictions only; no retraining.
Origin: the manuscript compares models by the CRPS averaged over samples whose observed outcome was a ramp. Scores
restricted to outcome-selected cases are not proper (forecaster's dilemma, Lerch et al. 2017): they can reward
forecasts that overstate the selected event. The ramp conclusions are therefore re-examined with a proper score that
weights the ramp region of the outcome space and is averaged over all forecasts.

## Score
For each forecast the ramp region is known at issue time: outcomes z (kW) with |z / P_clear(target) - k_now| >= tau_test,
i.e. z <= L or z >= U with L = (k_now - tau_test) P_clear, U = (k_now + tau_test) P_clear (frozen test ramp definition).
Threshold-weighted CRPS (Gneiting and Ranjan 2011) with weight w(z) = 1{z <= L or z >= U}:
twCRPS_ramp(F, y) = integral w(z) (F(z) - 1{y <= z})^2 dz = CRPS(v(F), v(y)), with the chaining function
v(z) = z for z <= L, L for L < z < U, z - (U - L) for z >= U. The complement (weight 1 - w) is twCRPS_calm, and
CRPS = twCRPS_ramp + twCRPS_calm. All three use the manuscript's CRPS computation (200-level midpoint rule on the
piecewise-linear quantile function, constant tails), which is the exact CRPS of 200 equally weighted atoms, so the
decomposition is exact. **STOP checks:** (i) the unweighted CRPS reproduces `paper/data/t4_main_test.csv` (frozen
models) and `revision/results/x3_images_only.json` (I15, P15) to 1e-6 kW; (ii) |CRPS - twCRPS_ramp - twCRPS_calm| <= 1e-9.

## Models and comparisons (all 13,266 test forecasts; 15-member ensemble quantiles)
G3''-F, E4-B-fit, E6-S, G4 (frozen), I15 and P15 (R-X3 / R-E1 member dumps). Pre-listed paired comparisons:
G3''-F - E4-B-fit, G3''-F - E6-S, G3''-F - G4, I15 - E4-B-fit, I15 - P15, E4-B-fit - P15.
For each: mean difference of CRPS, twCRPS_ramp and twCRPS_calm over all forecasts with the paired day-block bootstrap
(10,000 resamples, stratified, seed 20261022), next to the outcome-stratified ramp CRPS difference recomputed in the
same script (reference).

## Decision rule (fixed now)
For each comparison, the ramp conclusion **holds under the proper score** if the 95% CI of the twCRPS_ramp difference
excludes 0 with the same sign as the outcome-stratified ramp difference; it is **not confirmed** if the CI includes 0,
and **reversed** if it excludes 0 with the opposite sign. Reported whichever way it falls; where a conclusion is not
confirmed or reversed, the manuscript's wording on that comparison is changed. The gated probability mixture (R-X2) is
not included (its score uses a different, atom-exact computation).
