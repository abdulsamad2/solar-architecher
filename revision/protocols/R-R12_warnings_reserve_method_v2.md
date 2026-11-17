# Protocol R-R12: ramp warnings and reserve of method v2 on the official test set

Committed before the analysis is run. Label: **post hoc test** (category ii), stylised decision analyses. No model is
trained and nothing is fitted on test data; method v2's test quantiles are those of R-R9
(`D:/skippd_hires/R_M3/eval/test_quantiles.npz`, key `method_v2`). Development is closed; this run does not select or
change the method.

## Origin
R-R4 (ramp warnings) and R-R5 (reserve at equal reliability) were computed for the frozen forecaster G3''-F. The
manuscript now presents method v2 (R-M1, R-R9). Its operational section must describe the method it proposes.

## Known before the run
R-R4 and R-R5 for the frozen forecaster (falling ramps: relative value at the base rate 0.72 [0.63, 0.77] for its
probability against 0.34 for its median warning and 0.40 for the PV-only model; 94% of falling ramps warned, 49% with
at least 7 min to the crossing; reserve 9.4% of capacity at 95% reliability). R-R9 and R-R11 for method v2 (CRPS
0.9996 kW; median errors practically equal to the frozen forecaster's). Nothing about method v2's warnings or reserve
is known.

## Analysis
R-R4 and R-R5 are repeated without any change of event definitions, thresholds, cost-loss ratios, bootstrap, timing
rule or equalisation, with **method v2 in the place of the frozen forecaster**. The scripts of R-R4 and R-R5 take the
forecaster as an argument; with their defaults they must reproduce their stored result files (checked before the run;
STOP otherwise). Method v2's test CRPS must reproduce R-R9 (STOP otherwise).

Added, descriptive:
- Warnings: the frozen forecaster's probability as a further source, and the value difference method v2 minus the
  frozen forecaster at every cost-loss ratio (day-block bootstrap band).
- Reserve: the frozen forecaster as a further method, and the paired reserve difference method v2 minus the frozen
  forecaster (day-block bootstrap, day-level sign-flip p).

## Reading rules (those of R-R4 and R-R5, applied to method v2; fixed now)
- "Probability warnings are worth more than median warnings" is stated for an event only if, for method v2's own
  probability against its own median warning, the value difference has a bootstrap interval above zero at the base
  rate and at no cost-loss ratio is the median warning better with an interval above zero; otherwise the ranges are
  reported.
- "Images add warning value" is stated only if the value of method v2's probability exceeds that of the PV-only
  model's probability at the base rate with the interval above zero, for the falling-ramp event.
- Ramp-rate control is named as a possible use only if at least half of the falling ramps are warned at least 7 min
  before the crossing.
- "Less reserve at equal reliability than X" is stated only if the difference has a bootstrap interval below zero and
  sign-flip p < 0.05. The reliability on falling ramps is reported next to every reserve figure.
- No statement that method v2 is better or worse than the frozen forecaster in warnings or reserve is made unless the
  corresponding difference has an interval excluding zero (and, for reserve, sign-flip p < 0.05); otherwise "not
  shown to differ".

Everything is reported whichever way it falls. Results: `revision/results/r12_ramp_warnings_method_v2.json`,
`revision/results/r12_reserve_method_v2.json`. Script: `src/revision_r12_warnings_reserve_v2.py`.
