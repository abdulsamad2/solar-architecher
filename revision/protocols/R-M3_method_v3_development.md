# Protocol R-M3: one further development round of the method (v3), then one evaluation

Committed before any run. Development stages are **development out-of-fold** (category iii). The single evaluation of
the selected method on the official test set, the forward period and SkyGPT's external days is **post hoc**
(category ii / iv): all three were already seen with method v2, and no untouched SKIPP'D data exist (protocol note in
R-H0). This is therefore the **last** development round; whatever it selects is evaluated once and reported, and
method v2 is reported next to it.

## Origin
Method v2 (R-M1: gated probability mixture M-RC of the 60-min ramp expert R and the calm expert C) is the lowest of
all models on the test, the forward period and the external days, but is not shown better than the 15-min gated
reference G4 on the test (sign-flip p 0.071) or in the forward period. R still predicts the clear-sky index directly
with a plain quantile head, whereas R-R6 showed that the change target gives 86% of the calm expert's sunny-day gain
and the event-mixture head a further 4-7%. R-R7 and R-R10 showed poor conditional coverage (ramp coverage 0.535 in
the forward period; 90% interval coverage 0.80 on the external days).

## Stage A: 60-min expert with the calm expert's target and head
**New expert R'.** Encoder, window and inputs of R (3D CNN + GRU, 16 frames at 4-min stride, K history, no auxiliary
per-frame features, as A2), target and head of C (change in clear-sky index, event-mixture head with 2 components, supervised
event classes). Training settings identical to A2 and E6-S (lr 1e-4, batch 128, 15 epochs, patience 5, cosine, EMA
0.999, calibration-day holdout), seeds 42, 123, 456, five folds, common 60-min-eligible population. The event
threshold of each fold is the 0.9 quantile of |dK| on that fold's fit-day samples of the common population (training
data only). Output: `D:/skippd_hires/R_M3/Rdk`. No hyperparameter is tuned.

**Candidates** (fitted per seed and fold on calibration days, exactly as R-M1: features logit g of C and
log(d + 1e-3) of every non-calm expert against C, exact mixture CRPS, ridge grid on the chronological half split,
L-BFGS from zero):
- **M-RC** (method v2, incumbent).
- **R'** alone (one network, no gate).
- **M-R'C**: gated mixture of R' and C.
- **M-R'RC**: gated mixture of R', R and C.
- Constant-weight controls of the two mixtures.

**Selection rule (fixed now).** Complexity order: R' alone < M-RC = M-R'C < M-R'RC.
1. A candidate replaces the incumbent M-RC only if its overall CRPS is lower with the paired day-block bootstrap
   interval below zero in **3 of 3 seeds**, the seed-mean improvement is at least **1%**, and its calm-region part is
   not worse with an interval above zero in any seed. (R' alone, being simpler, also replaces M-RC if it is not worse:
   interval not above zero in any seed.)
2. Among candidates that qualify, the simplest is chosen; a more complex one replaces it only if it qualifies against
   it by rule 1.
3. A gate is kept only if it beats its constant-weight control with the interval below zero in 3 of 3 seeds.
If nothing qualifies, the method stays v2 and Stages B and C are applied to v2.

## Stage B: joint fine-tuning of the experts and the gate on the mixture CRPS
Applied to the design selected in Stage A. Because the loss, the parameterisation and the learning rate depend on
which experts are in that design, the exact specification is written as **Amendment 1 to this protocol and pushed
before Stage B runs**. Fixed now: fine-tuning uses fit days only, early stopping on calibration days, validation
folds are never used for fitting; the fine-tuned method replaces the Stage A method only by rule 1 above.

## Stage C: calibration conditional on the forecast-time ramp probability
Applied to the method after Stage B. No retraining. Continuous rolling conformal prediction is run separately in
classes of the calm expert's forecast-time ramp probability g (class edges and window fixed on development data
only, written as **Amendment 2 and pushed before Stage C runs**); R-Y2 is the starting point. Accepted only if, on
development validation folds, it brings 90% coverage on ramp samples closer to nominal without widening the overall
90% interval by more than 5% and without a worse overall interval score in any seed. CRPS is not affected.

## Evaluation (once, after Stages A-C)
The selected method, with every parameter fitted on development data, is run once on the official test set, the 83
forward days and SkyGPT's five external days with the scripts of R-R9, R-R7 and R-R10 (same scores, same day-level
tests, same reading rules). Reported next to method v2, G4, the calm expert, the direct expert, the PV-only network
and the naive probabilistic references, whichever way the results fall. No further variant is evaluated on these
data afterwards.

## Reporting
All candidates and both outcomes of every stage are reported, including the case that method v2 stays.
