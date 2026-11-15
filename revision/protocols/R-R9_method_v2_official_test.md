# Protocol R-R9: the selected combination rule (method v2) on the official test set

Committed before the analysis is run. Label: **post hoc test** (category ii). The official test set was used once
for the frozen evaluation and since then for post hoc analyses; this run adds one more. The rule itself was selected
on development data only (R-M1) and is applied here without any change. R-X2 already showed, on this test set, that a
probability mixture of the 60-min ramp expert and the calm expert with the frozen gate weights or a refitted gate was
slightly better than the frozen quantile blend on the 60-min-eligible forecasts; the routed forecaster with both
mixtures and its comparison with G4, the calm expert and the references were not computed.

## Method v2 (as fixed in R-M1)
For each of the 15 members (5 folds x 3 seeds), gates fitted on that member's calibration days: gated probability
mixture of the 60-min ramp expert and the calm expert (features g and d) on the common 60-min-eligible calibration
samples; gated probability mixture of the direct expert and the calm expert on all calibration samples. Test
forecasts with a 60-min image history use the first, the others the second; level-wise quantile averaging over the 15
members.

## Comparisons (method v2 minus each; CRPS, ramp-region and calm-region threshold-weighted CRPS)
Frozen rule G3''-F, G4, calm expert, direct expert, PV-only model P15; skill against the naive probabilistic
references of R-R3 from their stored mean CRPS. Inference: paired day-block bootstrap and the exact day-level
sign-flip test (R-R2). STOP check: the frozen models reproduce their stored CRPS.

## Reading rule (fixed now)
"Method v2 is better than X on the official test" only if the bootstrap interval of the overall CRPS difference is
below zero and the sign-flip p is below 0.05. Reported whichever way it falls. The frozen evaluation (H1-H5) is not
replaced: it remains the only confirmatory result, and it concerns the frozen rule.
