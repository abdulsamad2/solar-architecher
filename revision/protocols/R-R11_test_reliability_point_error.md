# Protocol R-R11: interval reliability and point error of method v2 on the official test set

Committed before the analysis is run. Label: **post hoc test** (category ii), **descriptive**. No model is trained,
no design is selected and no comparison decides anything: development closed with R-M3, and the method of the
article is method v2 (gated probability mixture M-RC with the fallback M-DC, R-M1). R-R9 scored its CRPS on the
official test set; the coverage of its intervals, the reliability of its quantiles and the error of its median on
that set were never computed. This run adds them, because a results section needs them.

## Inputs (all existing; nothing is refitted on test data)
- Test quantiles of method v2 as stored by the R-M3 evaluation (`D:/skippd_hires/R_M3/eval/test_quantiles.npz`).
  STOP check: their mean CRPS equals the R-R9 value.
- Frozen test predictions of G3''-F, G4, the calm expert (E6-S) and the direct expert (E4-B-fit).
- Development validation predictions of the three experts (direct, calm, 60-min ramp), three seeds x five folds.

## Development out-of-fold scores of method v2 (needed to start the conformal window)
For every seed and fold the two gates are fitted on the calibration days exactly as in R-M1 / R-R9 and applied to the
validation days of that fold: M-RC on the common 60-min-eligible samples, M-DC on the others. The three seeds are
averaged level by level (as the frozen pipeline did for its development scores). The conformity scores
max(L - y, y - U) of the 50%, 80% and 90% base intervals are the development scores of method v2. The development
CRPS of this routed three-seed forecast is reported.

## Calibrated intervals on the test set
The procedure of R-C07 (past-only): at the start of each test day the window holds the W latest development scores
of the same model whose target time is before the day's first issue time; scores of the day enter when their target
time has passed. W = 4620 for method v2 (the setting of the method; the value selected for G3''-F and the calm expert
on development data). For the frozen models the stored development scores and their stored W are used, and the
stored R-C07 coverage must be reproduced (STOP check).

## Reported, for method v2, G3''-F, G4, calm expert, direct expert
1. Raw and calibrated 50/80/90% intervals: coverage, mean width, interval (Winkler) score; overall, sunny, cloudy,
   ramp, non-ramp; share of outcomes below and above the interval.
2. For method v2: ramp coverage against the selection reference of R-Z1 (coverage that exactly correct forecasts
   reach on equally selected outcomes), at 50/80/90%.
3. Reliability of the nine quantiles: share of outcomes below each quantile, overall and on cloudy days.
4. Median: RMSE, MAE (kW and per cent of the 30.1 kW capacity), overall, sunny, cloudy, ramp, non-ramp; RMSE and MAE
   skill against the official smart persistence (stored frozen value).
5. Paired day-bootstrap differences method v2 minus G3''-F and minus G4 for the 90% interval score, the absolute
   error of the median and the squared error of the median (descriptive; no claim of "better" is derived).

## Reading
Reported whichever way it falls. H5 (calibrated 90% coverage within 0.90 +/- 0.03) was a hypothesis about the frozen
rule and stays with it; the corresponding number of method v2 is reported as a post hoc description and named as
such. The calibrated bounds of method v2 are stored next to the quantiles for figures.
