# Protocol R-R6: is the calm-condition gain due to the change target or to the event-mixture head?

Committed before this comparison is computed. Label: **development out-of-fold** (category iii). No model is trained:
the comparison uses three expert types that already exist with identical folds, seeds, inputs, encoder and training
budget (3 seeds x 5 folds each):
- **direct expert** (E4-B-fit): nine quantiles of the clear-sky index K(t+15), linear quantile head;
- **change-target quantile expert** (E6-QR, a development variant of phase 6): nine quantiles of the change
  K(t+15) - K(t), quantile head with one hidden layer;
- **calm-condition expert** (E6-S): the same change target with the event-mixture head.
The overall development scores of these models were seen during development (phase 6); their split into ramp-region
and calm-region parts and the paired comparison below were not computed.

## Origin
A simulated review (2026-11-16) noted that the calm expert differs from the direct expert in the target (change of K,
i.e. anchoring on persistence) and in the head at once, so its threefold lower sunny-day CRPS cannot be assigned to
either, and asked for a direct quantile expert on the change target.

## Analysis
Out-of-fold validation forecasts of all development samples, per seed: overall CRPS, CRPS on sunny and cloudy days,
ramp-region and calm-region threshold-weighted CRPS (R-Z2, fold thresholds); paired day-block bootstrap of
(change-target quantile expert - direct expert) and (calm expert - change-target quantile expert).

## Reading rule (fixed now)
Share of the calm expert's sunny-day gain over the direct expert that the change-target quantile expert recovers
(seed mean). If at least two thirds, the manuscript attributes the calm-condition accuracy mainly to the change
target (persistence anchoring); if at most one third, mainly to the event-mixture head; otherwise to both. The
head of the change-target quantile expert has one hidden layer more than the direct expert's, which is stated as a
remaining difference. Reported whichever way it falls.
