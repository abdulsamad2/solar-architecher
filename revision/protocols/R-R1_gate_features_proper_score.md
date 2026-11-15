# Protocol R-R1: which gate feature carries the benefit, under the overall and the proper ramp-weighted CRPS

Committed before the analysis is run. Labels: **development out-of-fold** (category iii) and **post hoc test**
(category ii). No model is trained; the frozen evaluation is unchanged.

## Origin
A simulated review of manuscript v8 (2026-11-16) objected that the statement "the gate's benefit is carried mainly by
the expert disagreement d" rests on outcome-stratified ramp CRPS only (R-E2: removing d worsens ramp CRPS in 3/3
seeds, removing g in 0/3). The stored R-E2 results already show that on the overall CRPS the two-feature gate is not
detectably better than the g-only gate (development -0.0017, +0.0003, -0.0012 kW; test -0.0017 [-0.0054, +0.0019]) and
only slightly better than the d-only gate (development -0.0023, -0.0049, -0.0046; test -0.0044 [-0.0109, +0.0027]).
Outcome-stratified ramp scores are biased (R-Z2). These numbers were known when this protocol was written; the
ramp-region and calm-region scores of the gate variants were not computed before.

## Analysis
Same frozen expert predictions, same fits and same samples as R-E2 (development: common 60-min-eligible validation
samples, three seeds; test: 60-min-eligible official test samples, 15 members, quantile-averaged). Methods: 60-min
expert alone, calm expert alone, equal-weight blend, CRPS-optimal constant weight, g-only gate, d-only gate,
two-feature gate (G3''), G4. For each method the CRPS is split exactly into its ramp-region and calm-region parts
(threshold-weighted CRPS of R-Z2; ramp region known at issue time: outcomes at least tau away from the current
clear-sky index, tau = the fold's frozen threshold on development data and the frozen test threshold on the test).
STOP checks: the overall CRPS values reproduce the stored R-E2 values; the two parts add up to the CRPS.

Reported: differences two-feature gate minus each alternative for the overall CRPS, the ramp-region part and the
calm-region part, with paired day-block bootstrap intervals (10,000, stratified sunny / cloudy; per seed on
development data); the correlation between the two standardised gate features; the share of the gate's gain over the
constant-weight blend that each single-feature gate recovers.

## Reading rule (fixed now)
- "State-dependent gating helps": two-feature gate minus constant-weight blend, overall CRPS interval below 0 on the
  test and in 3/3 development seeds.
- "Disagreement carries the benefit": supported only if, for the ramp-region part **and** the overall CRPS, removing d
  (g-only gate) is worse than the two-feature gate with the interval below 0 on the test and in 3/3 seeds, while
  removing g (d-only gate) is not.
- Otherwise the manuscript states what the proper scores show (for example "either feature recovers most of the gain
  overall; on observed ramps the effect comes through d") and the highlight, abstract and contribution (2) are
  reworded accordingly. The result is reported whichever way it falls.
