# Protocol R-R2: day-level exact tests for the key contrasts on the official test set

Committed before the analysis is run. Label: **post hoc test** (category ii); robustness check of the inference, the
frozen evaluation and its bootstrap intervals are unchanged. No model is trained.

## Origin
A simulated review of manuscript v8 (2026-11-16) objected that the paired day-block bootstrap rests on about ten
effective clusters per stratum (20 test days, 10 sunny and 10 cloudy), that non-studentised cluster-bootstrap
intervals are too narrow with so few clusters (Cameron, Gelbach and Miller 2008), and that the interval type is not
stated. The intervals of the manuscript are percentile intervals of the resampled day-block mean.

## Contrasts (fixed now; frozen test predictions, proposed forecaster G3''-F minus comparator, kW)
1. Overall CRPS vs the direct expert (H1).
2. Ramp CRPS vs the direct expert (H3; outcome-stratified, as pre-registered; margin 0.02 kW).
3. Overall CRPS vs the calm-condition expert.
4. Overall CRPS vs the 15-min gated reference G4.
5. Non-ramp CRPS vs G4.
6. Ramp CRPS vs G4 (outcome-stratified).
7-9. Ramp-region threshold-weighted CRPS (R-Z2) vs the direct expert, the calm expert and G4.

## Methods
For each contrast: per-day sums S_d of the per-forecast differences and counts n_d (days without a sample in the
stratum are dropped).
- **Exact sign-flip test** (randomisation test under symmetry of the day-level differences about zero; Canay, Romano
  and Shaikh 2017): statistic = sum(S_d) / sum(n_d), the pooled mean of the manuscript; all 2^D sign patterns of the
  days are enumerated; two-sided p = share of patterns with |statistic| at least the observed value. For the
  non-inferiority contrast 2 the differences are shifted by the margin (S_d - 0.02 n_d) and the one-sided p for
  "difference < margin" is reported.
- **t interval on day means** (Ibragimov and Mueller 2010): equal-day-weight mean of the daily mean differences,
  stratified by sunny / cloudy days, standard error from the within-stratum variances, t quantile with D - (number
  of strata present) degrees of freedom.
- Number of days on which the proposed forecaster is better, and the largest single-day share of sum(S_d).
- The stored percentile bootstrap interval is reproduced next to these for comparison.
Holm adjustment is applied to the sign-flip p-values of contrasts 1 and 2 (the pre-registered hypotheses that involve a
model comparison); contrasts 3-9 are unadjusted and descriptive.

## Reading rule (fixed now)
A conclusion of the manuscript is called **robust** if the sign-flip test (two-sided p < 0.05) and the t interval agree
with the reading of the bootstrap interval (excludes zero / includes zero), **not robust** if they disagree. Every
contrast is reported whichever way it falls; a contrast that is not robust is reworded in the manuscript as
inconclusive. The forward period (83 consecutive days) is treated in a separate step with multi-day blocks, because it
needs the forward ensemble predictions to be regenerated.
