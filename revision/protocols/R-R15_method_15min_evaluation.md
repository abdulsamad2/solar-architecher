# Protocol R-R15: evaluation of the 15-min form as the proposed method

Committed and pushed before the analysis is run. Label: **post hoc** (category ii on the official test set, category
iii on development folds, category iv data in the forward period, descriptive on the five external days). No expert
is trained. Development is closed: nothing here tunes, selects or changes a model, a gate, a threshold or the
calibration window.

## Origin and what is decided here
R-M1 selected, by its committed rule, the gated probability mixture of the 60-min ramp expert and the calm expert,
with a 15-min mixture of the direct and calm experts as fallback ("method v2"). R-R13 then compared method v2 with its
own 15-min form (the fallback mixture applied to every forecast, "M-DC all") and did not show a benefit of the 60-min
expert: official test set −2.2% (interval below zero, sign-flip p 0.061), forward period +0.3% (p 0.77); only the
five external days show a difference (−6.2%, descriptive).

**The author now proposes the 15-min form as the method.** This is a choice made after the test results were seen.
It is a choice of the simpler design with the *higher* (worse) CRPS in two of three evaluations (official test set
1.022 against 1.000 kW; external days 2.42 against 2.27 kW; forward period 0.364 against 0.366 kW), made because the
benefit of the larger design is not established, and because the 15-min form needs no hour of image history, no
routing and no rebuilt history on the external days. The article must state this choice and report the 60-min design
as a tested extension with its results.

## Known before the run
For the 15-min form: its CRPS and its ramp-region and calm-region parts on the official test set and in the forward
period; its contrasts with method v2, the frozen predecessor G3''-F and the 15-min gated reference G4 (R-R13, part A);
the lag of its median (R-R13, part D); its CRPS on the external days with the exact and the sample-based estimators
(R-R10, "method v2, fallback only"); its development CRPS and that of a constant weight (R-M1: 0.593 and 0.608 kW).
For method v2: everything in R-R7, R-R9 to R-R13. **Not known:** the contrasts of the 15-min form with the single
networks, the PV-only network and the naive references; its calibrated intervals; its ramp coverage against the
selection reference; its warnings and reserve; its gate ablation.

## Analyses (the 15-min form in the place of method v2; definitions unchanged from the cited protocols)
Forecasts: the saved quantiles of R-R13 (`test_quantiles_r13.npz`, key `mdc_all`; `forward_quantiles_r13.npz`, key
`mdc_all`); on the external days the forecasts are recomputed as in R-R10. Each must reproduce its stored CRPS (STOP
otherwise). The gates are those of R-M1 (fitted on calibration days; two features; ridge penalty chosen there).

Official test set (CPU):
1. CRPS with its ramp-region and calm-region parts, sunny and cloudy days, and paired contrasts with: the direct image
   expert, the calm expert, the PV-only network, G4, G3''-F, method v2, the quantile blend of the same two experts
   with the frozen 15-min gate, the six naive probabilistic references of R-R3 and the smart-persistence point
   forecast. Day bootstrap (10,000 resamples, stratified) and exact day-level sign-flip test, as in R-R13.
2. Gate: the same mixture with a constant weight, with a gate on g only and on d only (test set, and development
   out-of-fold forecasts per seed; the development values of the full gate and the constant weight must reproduce R-M1).
3. Intervals at 50%, 80% and 90%: raw; calibrated with one continuous rolling window (W = 4620, started from the
   latest matured development out-of-fold scores of the 15-min form before the first test day), which is the
   calibration of the method; and with the window refilled at the start of each test day (as R-R11), for comparison.
   Coverage, width and interval score overall, on sunny and cloudy days and on ramp and non-ramp outcomes; quantile
   reliability; RMSE and MAE of the median.
4. Ramp coverage against the selection reference with a joint day bootstrap of the gap, and coverage by direction
   (as R-R13, part E), for the continuous window.
5. Ramp warnings (as R-R4/R-R12) and reserve at equal reliability (as R-R5/R-R12), the operator quantities of the
   warnings and the reserve equalised within cloudy days (as R-R13, parts G and H).

Forward period (GPU inference of the comparators, as R-R7/R-R13): scores and paired contrasts with the same models and
with the three naive references of R-R7 (day bootstrap; sign-flip test on 7-day blocks); continuous conformal
intervals started from the selection-period scores of the 15-min form; ramp coverage against the selection reference.

External days (GPU inference, as R-R10): paired day-bootstrap contrasts of the 15-min form with the direct expert,
the calm expert, G4, the quantile blend of the same experts and method v2 (five days; descriptive); 50%, 80% and 90%
coverage raw, with a continuous window and with ACI, started from the development scores of the 15-min form. The
sample-based CRPS values are those of R-R10.

## Reading rules (fixed now)
- A difference is called **shown** only if the day-bootstrap interval excludes zero and the sign-flip test gives
  p < 0.05 (days on the test set, 7-day blocks in the forward period). External-day contrasts are descriptive.
- "The method is better than X" is written for an evaluation only if shown there. Where the 15-min form is not shown
  better than a comparator (in particular the calm expert, G4 or the quantile blend), the article says so.
- "The gate is needed" is written only if the gated mixture is better than the constant weight with the interval
  below zero in three of three development seeds and shown on the test set.
- Calibration is described by the continuous window. Coverage of ramp outcomes is reported with the selection
  reference; a deficit beyond selection is stated at a level if the interval of the gap lies below zero.
- The reading rules of R-R12 for warnings and reserve apply unchanged to the 15-min form.
- All numbers are reported whichever way they fall. The results of method v2 stay in the public record and are
  summarised in the article as those of the tested 60-min extension.

## Outputs
`revision/results/r15_method_15min.json` (keys `development`, `test`, `forward`, `external`),
`r15_ramp_warnings_15min.json`, `r15_reserve_15min.json`; logs `r15_log_*.txt`. Script: `src/revision_r15_method_15min.py`.
