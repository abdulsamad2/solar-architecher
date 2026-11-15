# Protocol R-M1: development of the combination rule of the method (v2), on development data only

Committed before the analysis is run. Label: **development out-of-fold** (category iii, exploratory). No expert is
retrained; the official test set, the forward evaluation days and the raw days of November-December 2019 are not
used. The purpose is to fix one combination rule, by a rule written here, **before** a frozen confirmatory comparison
on untouched data (separate protocol R-M2, to be committed before those data are opened).

## Origin
The frozen method (gated quantile blend with a 60-min ramp expert, G3''-F) beats the conventional single network but
not its simpler variants: against the 15-min gated reference G4 the overall difference includes zero on the official
test and is slightly positive in the forward period; a gated probability mixture of the same experts was about 1%
better than the quantile blend (R-X2); either gate feature recovers most of the gate's gain (R-R1). A simulated
review (2026-11-16) therefore asked which design should be called the method.

## Candidates (all fitted per seed and fold on the fold's calibration days only; frozen expert predictions)
Experts: D = direct image expert (E4-B-fit), R = 60-min ramp expert (A2), C = calm-condition expert (E6-S).
- **Q-RC** (reference): the frozen two-feature quantile blend of R and C (G3'').
- **M-DC:** gated probability mixture of D and C, features g and d(D, C). 15-min history only, two experts.
- **M-RC:** gated probability mixture of R and C, features g and d(R, C) (the LP-refit of R-X2).
- **M-DRC:** gated probability mixture of D, R and C with a softmax gate (C is the reference class), features g,
  d(D, C) and d(R, C).
- Constant-weight versions of M-DC, M-RC and M-DRC (CRPS-optimal weights), as controls for the gate.
Features are transformed and standardised as in the frozen gates (logit g, log(d + 1e-3)); the mixture CRPS is exact
(energy form on the 200-atom grids); the ridge penalty is chosen on the chronological half split of the calibration
days from the frozen grid (1e-4 ... 1e-1); L-BFGS from zero. Scores are computed, as for every model, on the nine
nominal quantiles of the mixture.

## Population and scores
Common 60-min-eligible development validation samples (as R-E2 and R-X2), seeds 42, 123, 456. Overall CRPS (primary),
ramp-region and calm-region threshold-weighted CRPS (R-Z2), CRPS on sunny and cloudy days, raw 90% coverage and
width. Paired day-block bootstrap per seed.

## Selection rule (fixed now)
Complexity order: M-DC (two experts, 15-min history, no routing) < M-RC (60-min expert, routing) < M-DRC (three
experts, routing).
1. The method is **M-DC** unless a more complex candidate beats it.
2. A more complex candidate replaces a simpler one only if, against that simpler candidate, its overall CRPS is lower
   with the bootstrap interval below zero in **3 of 3 seeds**, the seed-mean improvement is at least **1%** of the
   simpler candidate's overall CRPS, and its calm-region part is not worse with an interval above zero in any seed.
3. If M-RC and M-DRC both qualify against M-DC, M-DRC is chosen only if it also qualifies against M-RC by rule 2.
The gated version of the chosen design is kept only if it beats its constant-weight control overall with the interval
below zero in 3 of 3 seeds; otherwise the constant-weight mixture is the method.

## Fixed without further selection (evidence already reported)
- Calibration: continuous rolling conformal prediction with the frozen window W (R-E3 forward evaluation and the
  external-set result showed the daily reset to fail); no daily reset.
- Ensemble: level-wise quantile averaging over members, as in the frozen method.
- Routing for samples without a 60-min history: M-DC.

## Reporting
All candidates are reported whichever way the selection falls, including the case that nothing beats the simplest
design. The chosen rule is then written into protocol R-M2 together with the hashes of the expert files, before any
untouched data are opened.
