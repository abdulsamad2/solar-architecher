# R-Z1 to R-Z3: ramp verification without the outcome-selection bias

All three use the frozen predictions only, with no retraining. The protocols were committed and pushed (1ff53a8) before any run. Results: `revision/results/z1_ramp_coverage_null.json`, `z2_ramp_weighted_crps.json`, `z3_outcome_class_conformal.json`. Label: post hoc on the official test (category ii); R-Z3 takes its initial scores from development.

Why: a ramp is defined by the observed outcome (|k_target − k_now| ≥ 0.279). Coverage and scores computed on outcome-selected cases are biased (forecaster's dilemma, Lerch et al. 2017): a perfectly calibrated forecast covers selected extreme outcomes less often than nominal, and a forecast that overstates ramps scores better on realized ramps.

## R-Z1: how much of the low ramp coverage is selection? Verdict: mostly selection; no shortfall beyond selection shown at 90%

Null: outcomes are drawn from each forecast's own calibrated distribution (2,000 draws per forecast), labelled as ramps by the frozen rule, and checked against the issued interval.

| Proposed forecaster, ramp outcomes | Observed coverage | Coverage if the forecast were exactly right | Gap [95% CI] | Share of the gap to nominal explained by selection |
|---|---|---|---|---|
| 90% interval | 0.746 | 0.779 | −0.032 [−0.098, +0.040] | 0.79 |
| 90%, rising ramps | 0.696 | 0.773 | −0.077 [−0.146, +0.001] | 0.62 |
| 90%, falling ramps | 0.795 | 0.784 | +0.011 [−0.082, +0.098] | 1.10 |
| 80% interval | 0.496 | 0.595 | −0.099 [−0.166, −0.017] | 0.67 |
| 50% interval | 0.156 | 0.270 | −0.113 [−0.168, −0.049] | 0.67 |

- The forecast's own ramp rate is 0.156 against 0.144 observed.
- With linear instead of exponential tails the 90% null is 0.789 (gap −0.043 [−0.109, +0.029]).
- Other models at 90%: direct expert 0.747 observed vs 0.725 null; G4 0.718 vs 0.765 (gap −0.048 [−0.107, +0.021]); calm-condition expert 0.686 vs 0.774 (gap −0.088 [−0.155, −0.012], a genuine shortfall).
- Issue-time view (no outcome selection), 90% coverage by the forecast's own ramp probability: 0.938 (p < 0.05), 0.918 (0.05–0.2), 0.927 (0.2–0.5), 0.867 (p ≥ 0.5; 1,204 forecasts).

Reading: about four fifths of the distance between 0.90 and 0.746 is what an exactly correct forecast would also show. A genuine deficit remains in the centre of the distribution on ramps (50% and 80% intervals) and when the forecast itself expects a ramp.

## R-Z2: do the ramp comparisons survive a proper score?

CRPS = ramp-region part + calm-region part (threshold-weighted CRPS; the ramp region is known at issue time; exact to 1e-9; averaged over all 13,266 forecasts).

| Model | CRPS | Ramp region | Calm region | of the ramp-region score: on ramp outcomes / on non-ramp outcomes |
|---|---|---|---|---|
| Proposed (G3″-F) | 1.006 | 0.381 | 0.625 | 0.329 / 0.051 |
| Direct image expert | 1.117 | 0.384 | 0.733 | 0.340 / 0.044 |
| Calm-condition expert | 1.090 | 0.411 | 0.679 | 0.372 / 0.039 |
| G4 | 1.024 | 0.386 | 0.638 | 0.347 / 0.039 |
| Images only (I15) | 1.191 | 0.387 | 0.805 | 0.303 / 0.083 |
| PV history only (P15) | 1.239 | 0.453 | 0.787 | 0.428 / 0.025 |

| Comparison | Ramp-region difference [95% CI] | Outcome-stratified ramp CRPS difference | Verdict |
|---|---|---|---|
| Proposed − direct expert | −0.004 [−0.015, +0.007] | −0.117 [−0.244, +0.032] | not confirmed |
| Proposed − calm-condition expert | −0.030 [−0.042, −0.018] | −0.771 [−0.877, −0.652] | holds |
| Proposed − G4 | −0.005 [−0.014, +0.003] | −0.271 [−0.364, −0.173] | not confirmed |
| Images only − direct expert | +0.002 [−0.013, +0.017] | −0.426 [−0.546, −0.304] | not confirmed |
| Images only − PV only | −0.066 [−0.110, −0.029] | −1.650 [−2.153, −1.180] | holds |
| Direct expert − PV only | −0.068 [−0.103, −0.039] | −1.224 [−1.649, −0.821] | holds |

Reading: images add ramp skill beyond the PV history under a proper score (both image models beat the PV-only model by about 0.07 kW in the ramp region), and the PV history adds the calm-region skill (images only is 0.072 kW worse than the direct expert there). The apparent ramp advantage of the images-only model over the direct expert, and of the proposed forecaster over G4, is an effect of outcome selection: the images-only model gains on realized ramps what it loses through ramp mass on non-ramp outcomes.

## R-Z3: what does nominal coverage on ramp outcomes cost? Verdict: ramp protection attained

Class-conditional conformal sets (separate rolling score windows for ramp and non-ramp outcomes; each candidate outcome is tested against its own class).

| 90%, test | Coverage overall | Ramp | Rising | Falling | Non-ramp | Mean set size (kW) |
|---|---|---|---|---|---|---|
| Unconditional (C07 past-only) | 0.926 | 0.746 | 0.696 | 0.795 | 0.956 | 6.23 |
| Outcome-class conformal | 0.935 | 0.902 | 0.881 | 0.921 | 0.940 | 7.18 |

The sets are 15% larger (+0.95 [+0.63, +1.26] kW) and 26% of them are not a single interval.
