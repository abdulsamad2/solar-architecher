# Protocol R-Z1: calibrated-forecast null for ramp coverage (selection versus miscalibration)

Committed before any run. Label: **post hoc** (official test, category ii). Frozen predictions only; no retraining.
Origin: the calibrated 90% intervals cover about 0.75 of ramp outcomes (C07 past-only rule). A ramp is defined by the
observed outcome (|k_target - k_now| >= tau_test). Coverage on outcome-selected events is below nominal even for a
perfectly calibrated forecast (forecaster's dilemma, Lerch et al. 2017), so the shortfall mixes a selection effect with
genuine miscalibration. R-Y2 conditioned on issue-time features only and does not separate the two.

## Question
How much of the gap between nominal (0.90) and observed ramp coverage would also occur if the issued predictive
distributions were exactly correct, and how much remains?

## Inputs
- Frozen 15-member test quantiles of G3''-F, E4-B-fit, E6-S, G4 (`experiments/final/test_predictions_v1.npz`).
- Calibrated 50/80/90% intervals from the C07 past-only rolling rule, recomputed. **STOP check:** overall coverage must
  reproduce `revision/results/c07_conformal_pastonly.json` to 1e-9 for every model and level.
- Frozen test ramp definition (`reports/test_ramp_definition_frozen.json`: tau_test, clear-sky anchors).

## Null model (the forecast is the truth)
For each forecast, the calibrated predictive distribution has the quantile nodes
(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95) = (lo90, lo80, lo50, raw median, hi50, hi80, hi90), rearranged to be
non-decreasing, linear in between. Tails beyond 0.05 / 0.95:
- **T-exp (primary):** exponential excess with mean equal to the adjacent node spacing (v10 - v05, v95 - v90), which
  continues the density at the node.
- **T-lin (sensitivity):** linear extrapolation of the outer segment (uniform excess of that spacing).
Draws are clipped to [0, 1.5 x clear-sky power at the target time]. A simulated outcome y* is a ramp if
|y* / P_clear(target) - k_now| >= tau_test, i.e. the frozen rule applied to y*; it is covered if it lies inside the
issued calibrated interval. M = 2000 independent draws per forecast (seed 20261115) give, per forecast, the
probabilities of ramp, rising / falling ramp, and ramp-and-covered; each draw column is one replicate test set.

## Quantities
- c_obs: observed coverage on ramps (and rising / falling ramps, non-ramps, overall) at 50/80/90%.
- c_null: expected coverage on simulated ramps under the null, sum p(ramp and covered) / sum p(ramp), with the 95%
  range over the 2000 replicate test sets; also the null ramp rate against the observed ramp rate.
- Gap c_obs - c_null with a paired day-block bootstrap (10,000 resamples of days, stratified by sunny / cloudy days,
  seed 20261022; ratio statistics recomputed per resample).
- Selection share s = (nominal - c_null) / (nominal - c_obs) at 90%.
- Issue-time view (no outcome selection): observed 90% coverage by bins of the forecast's own null ramp probability
  p(ramp) in [0, 0.05), [0.05, 0.2), [0.2, 0.5), [0.5, 1], against the nominal 0.90.

## Decision rule (fixed now; primary: G3''-F, 90%, T-exp)
- **Genuine shortfall** if the bootstrap 95% CI of c_obs - c_null lies below 0; otherwise **no shortfall beyond
  selection shown**.
- The 0.90 -> c_obs gap is **mostly selection** if s >= 2/3, **mostly miscalibration** if s <= 1/3, **both** otherwise.
Reported whichever way it falls. If the verdict contradicts the manuscript's wording on ramp coverage, the manuscript
is reworded; the frozen method and earlier verdicts are not changed. T-lin and the other models are reported as
sensitivity and do not change the verdict.
