# Protocol R-R7: the selected combination rule (method v2) and the review-driven checks on the forward period

Committed before the analysis is run. Label: **forward-in-time, post hoc** (category iv data, analysis specified after
the forward results of the frozen rule were known: R-E3 and R-X4). No expert is retrained: the forward experts of
R-E3 (five seeds, fitted on days up to 2019-06-30, selection days 2019-07-02 to 07-31) are used; the 83 evaluation
days (2019-08-02 to 2019-10-26) are scored once with the rule below. These days were part of the development period
when the combination rule was selected in R-M1 (as out-of-fold validation days of other experts), so this is a
chronological refit, not an untouched test. No untouched day with PV exists (R-H0 note).

## Method v2 on the forward period (rule selected in R-M1, fixed before this run)
Per seed, gates fitted on the selection days only: gated probability mixture of the 60-min ramp expert and the calm
expert (features g and d) for forecasts with a 60-min image history; gated probability mixture of the direct expert
and the calm expert otherwise; level-wise quantile averaging over the five seeds; continuous rolling conformal
prediction (W as in R-E3, initial scores from the selection days).

## Comparisons (method v2 minus each; CRPS, ramp-region and calm-region threshold-weighted CRPS)
Frozen rule G3''-F, G4, calm expert, direct expert, PV-only model P15, official smart persistence (point), and the
naive probabilistic references of R-R3 built from data before the evaluation period (persistence ensemble 120 min,
smart persistence dressed with the last 120 min of errors, complete-history ensemble from the fit period).
Inference: paired day-block bootstrap (as R-E3) and, because the days are consecutive, an exact sign-flip test and a
t interval on the means of **blocks of seven consecutive evaluation days** (12 blocks, fixed now).
STOP check: the CRPS of the frozen rule, G4 and the experts reproduce R-E3.

## Ramp verification on the forward period
90/80/50% coverage of ramp outcomes of method v2 with continuous conformal intervals, and its selection reference
(R-Z1 construction: quantile nodes from the calibrated intervals and the median, exponential tails, 2,000 draws);
gap and selection share.

## Reading rule (fixed now)
"Method v2 is better than X on the forward period" only if the day-bootstrap interval of the overall CRPS difference
is below zero and the seven-day-block sign-flip test gives p < 0.05. Otherwise "not different" or "worse" as the
intervals say. All comparisons are reported whichever way they fall.
