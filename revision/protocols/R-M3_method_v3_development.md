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

---

## Amendment 1 (2026-11-16, after Stage A, before Stage C and before any evaluation): Stage B is not run
**Stage A outcome** (results/m3_stage_a.json): the three-way mixture M-R'RC was selected by the rule (seed-mean CRPS
0.5737 against 0.5796 for method v2, a gain of 1.02% with the interval below zero in 3 of 3 seeds and the calm region
not worse). R' alone (0.636) and M-R'C (0.615) were worse than method v2; the gain of M-R'RC is in the calm region
only.

**Decision.** Stage B (joint fine-tuning) is dropped. Reasons, all known before any evaluation of method v3 on the
test set, the forward period or the external days: (i) the selected design has three networks, two with 60-min
windows, so joint fine-tuning is the costliest step of the protocol; (ii) its expected gain is small against a
Stage A gain that only just clears the 1% threshold; (iii) Stage A showed that the mixture depends on the experts
being different, which joint training on one loss does not obviously preserve. No fine-tuned variant will be
evaluated on the test set, the forward period or the external days. "Method v3" below means M-R'RC of Stage A with
the fallback M-DC, before calibration.

## Amendment 2 (2026-11-16, before Stage C): specification of Stage C
**Data.** Development validation forecasts of method v3 (common population; one member per seed and fold, as in
Stage A), in chronological order per seed. Intervals: 50%, 80% and 90% (base_interval on the nine quantiles).
**Reference (U).** Continuous rolling conformal prediction with the frozen window W = 4620: for every forecast, Q is
the ceil((n + 1)(1 - alpha))-th smallest of the n <= W most recent scores whose target time is not later than the
issue time; with n < 30 the raw interval is issued.
**Candidate (G).** The same procedure run separately in four classes of the calm expert's forecast-time ramp
probability g (the gate feature): g < 0.1, 0.1-0.3, 0.3-0.5 and >= 0.5, each class with its own window of the W most
recent matured scores of that class; while a class has fewer than 200 matured scores, the pooled window U is used.
The class edges are fixed here and are not tuned.
**Acceptance (all three, at the 90% level, on the development validation folds).** (a) Coverage on ramp outcomes is
closer to 0.90 under G than under U in 3 of 3 seeds; (b) the mean width over all forecasts is at most 5% larger
(seed mean); (c) the interval (Winkler) score over all forecasts is not worse in any seed. Coverage, width and
score are also reported per class and for the 50% and 80% intervals. If accepted, G replaces continuous pooled
calibration in method v3 and is used in the evaluation; otherwise U stays. CRPS is unaffected.

## Amendment 3 (2026-11-16, before the evaluation): details of the single evaluation
- **Official test set.** Members of R' are run on the test set with the inputs of the other 60-min expert
  (src/revision_posthoc_dump.py); per member, the three-way gate is fitted on that member's calibration-day
  predictions (common population) and the fallback gate as in R-R9; forecasts without a 60-min history use M-DC.
  Scores, contrasts, day-level tests and the reading rule are those of R-R9, with method v3 in place of method v2
  and method v2 added to the comparators.
- **Forward period.** R' is trained on the forward fit days with the settings of Stage A and the forward manifest
  (five seeds, as the other forward experts; early stopping on the selection days); gates are fitted on the
  selection days; everything else as R-R7.
- **External days.** As R-R10, with R' members added to the 60-min branch.
- Calibration in all three: the outcome of Stage C.
- Method v2 keeps its already published numbers; both methods are reported side by side whichever is lower, and the
  article states which evaluations each was run on and in which order.
