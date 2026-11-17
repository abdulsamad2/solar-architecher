# Protocol R-R13: explanatory and verification analyses of method v2 (after an internal review)

Committed before any analysis of this protocol is run. Date: 2026-11-17. No model is trained. Development is closed
(R-M3 was the last development round): **nothing in this protocol selects or changes the method.** Method v2 stays the
gated probability mixture M-RC with the fallback M-DC, 15 members, continuous rolling conformal calibration, whatever
the results below are. The analyses measure parts of the method that the earlier protocols did not measure, and
report quantities that the manuscript states without support.

Labels: **post hoc test** (category ii) for the official test set; **forward-period data, post hoc analysis** for
the forward period (the forward period was already scored for method v2 in R-R7 and for method v3 in R-M3);
**development out-of-fold** (category iii) for the development folds; **external days, post hoc, five days,
descriptive** for the external days.

## Known before the run
Everything in `revision/results/` up to R-R12, in particular: method v2 on the test set 0.9996 kW, G4 1.024 kW
(v2 - G4: -0.024 [-0.046, -0.003], sign-flip p 0.071, not shown); forward period v2 0.3655, G4 0.368 (p 0.60);
external days v2 routed 2.268, v2 fallback only 2.418 (routed - fallback -0.150 [-0.294, -0.035]); development
(R-M1) M-RC 0.580, M-DC 0.593 on the common population; R-E4 (single experts; history and encoder effects on ramps);
R-R1 (gate features of the quantile blend); the 15-min best lag of the frozen forecaster's median
(`reports/lag_analysis.json`); R-R11 (test coverage of v2 with daily initialisation); R-R7 (forward coverage).
Not known: any number defined below.

## A. The method's own 15-min form (like-for-like comparator of the 60-min expert)
G4, the reference used so far, differs from the method in three ways (quantile blend, four gate features, no 60-min
expert). The like-for-like comparator is **M-DC applied to every forecast**: the gated probability mixture of the
direct and calm experts with the two features of the method, that is, the method's fallback branch used everywhere.
It differs from the method only in the ramp-capable expert of the 60-min branch.

- Official test set: 15 members, gates fitted on the calibration days as in R-R9, level-wise average. Scores as
  R-R9 (CRPS, ramp-region and calm-region parts, sunny, cloudy, ramp outcomes). Contrasts: v2 - M-DC-all,
  M-DC-all - G4, M-DC-all - frozen forecaster; on all forecasts and on the forecasts with a 60-min history. Day
  bootstrap (10,000 resamples, stratified), exact day-level sign-flip test, t interval.
- Forward period: five seeds as R-R7 (experts and gates of R-E3/R-R7), same contrasts, day bootstrap and sign-flip
  test on 7-day blocks. Method v2 must reproduce its stored R-R7 score (STOP otherwise).
- External days: the stored R-R10 values are quoted (v2 routed against v2 fallback only); nothing is recomputed.

Reading rule (fixed now). "The 60-min expert improves the method" is stated for the test set or the forward period
only if v2 - M-DC-all has a bootstrap interval below zero **and** a sign-flip p < 0.05 there. Otherwise "not shown".
If it is shown in neither, the manuscript states that the contribution of the 60-min expert is supported by the
development folds and by five external cloudy days only, and the title, highlights and first contribution must not
rest on the 60-min expert. If M-DC-all is better than v2 with interval above zero and p < 0.05 in an evaluation, that
is reported as such. The method is not changed in either case.

## B. History length, encoder and capacity of the ramp-capable expert (as partner of the calm expert)
R-E4 trained the 2 x 2 design as single experts on the common 60-min population: D0 (2D, 15 min), D2 (2D, 60 min),
A2-H15 (3D, 15 min), A2 (3D, 60 min). Here each of the four is the partner of the calm expert in the gated
probability mixture of the method (features g and d, gate rule of R-M1, gates fitted on calibration days).

- Development out-of-fold, three seeds, common population. The mixture with A2 must reproduce the R-M1 score of
  M-RC per seed (STOP otherwise). Effects per seed with paired day-bootstrap intervals, for the CRPS and its
  ramp-region and calm-region parts: history within 3D (A2 - A2-H15), history within 2D (D2 - D0), encoder at
  60 min (A2 - D2), encoder at 15 min (A2-H15 - D0). Parameter counts are reported (2D 92,873; 3D 213,545).
- Official test set, forecasts with a 60-min history, 15 members from the stored member dumps: the same four
  mixtures and the same four effects, with day bootstrap and sign-flip p.

Reading rule. In the mixture, a gain is credited to the 60-min history only if the history effect has an interval
below zero in at least two of three seeds within both encoders; to the 3D encoder only if the encoder effect does so
at both history lengths. Otherwise the manuscript says "the 60-min 3D expert" and states that history, encoder and
capacity are not separated. The test-set effects are reported next to the development effects and are called shown
only with interval below zero and p < 0.05.

## C. Gate features of the probability mixture
R-R1 and the manuscript's gate table describe the gate of the quantile blend. The same question for the method:
M-RC with (i) a constant weight (CRPS-optimal on the calibration days), (ii) a gate on g only, (iii) a gate on d only,
(iv) the gate on g and d (the method). Development out-of-fold (three seeds, common population; (i) and (iv) must
reproduce R-M1) and official test set (forecasts with a 60-min history, 15 members). Scores: CRPS and its two parts;
differences (iv) - (i), (ii), (iii) with day-bootstrap intervals (and sign-flip p on the test set); share of the gain
of (iv) over (i) recovered by (ii) and by (iii).

Reading rule. "The gate beats a constant weight" is stated for the method if (iv) - (i) has an interval below zero
in three of three seeds on development data; the test-set contrast is reported as post hoc. "Either feature is
sufficient" is stated only if neither (iv) - (ii) nor (iv) - (iii) has an interval below zero on the test set and in
at least two of three development seeds; otherwise the features that matter are named.

## D. Lag of the median
`src/lag_analysis.best_lag` (first differences of the forecast median and of the observed power, lags -5 to 20 min,
per day) on the ten cloudy days of the official test set, for the median of method v2, with the frozen forecaster
(must reproduce `reports/lag_analysis.json`), the direct expert, the calm expert and smart persistence as
references. Reported: median over days of the best lag, mean correlation at lag 0 and at lag 15 min, the number of
days with a best lag of at least 14 min.

Reading rule. "The median lags ramps by 15 min" is stated for the method only if the median best lag is at least
14 min; otherwise the measured value is stated.

## E. Interval coverage
1. Day-bootstrap interval (10,000 resamples of whole days, stratified by sunny and cloudy days) of the gap between
   the observed coverage of ramp outcomes and the selection reference, at 50%, 80% and 90%, on the official test set
   (intervals of R-R11) and in the forward period (intervals of R-R7; its stored coverage must be reproduced). The
   selection reference is computed per forecast (2,000 outcomes drawn from the forecast itself, as R-Z1), so that
   observed coverage and reference are resampled together. "Deficit beyond selection" is stated for a level only if
   the interval of the gap lies below zero.
2. Coverage of falling-ramp and rising-ramp outcomes separately (test set and forward period), with the mean
   distance by which missed outcomes fall outside the interval. Descriptive.
3. Official test set with one window that runs across the 20 test days without a daily reset (initial content: the
   4,620 latest matured development scores of method v2 before the first test day): coverage (all, sunny, cloudy,
   ramps) and width at the three levels, next to the R-R11 values (window refilled from development scores at the
   start of each test day). Descriptive; it shows how much Table 10 depends on the initialisation.

## F. Secondary comparisons
1. Official test set: paired difference v2 - reference for each of the six naive probabilistic references of R-R3
   and for the smart-persistence point forecast, with day-bootstrap interval and sign-flip p. The references are
   recomputed by the R-R3 code and must reproduce their stored CRPS (STOP otherwise).
2. Official test set: sign-flip p for contrasts that have only a bootstrap interval so far (absolute error of the
   median, v2 - frozen forecaster and v2 - G4; 90% interval score, v2 - frozen forecaster and v2 - G4).
3. External days: the fair ensemble estimator of the CRPS (pair term divided by m(m - 1)) for the method's stored
   quantiles at 10, 30 and 100 samples (mean of 20 draws), next to the standard estimator used by the published
   papers. Descriptive; the published values use the standard estimator.

## G. Operator quantities of the falling-ramp warnings (official test set)
For warnings from the event probability of method v2 at thresholds equal to the base rate, 0.2, 0.3 and 0.5, and for
warnings from its median: share of forecasts under warning (all, cloudy days), precision (share of warnings followed
by a falling ramp), hit rate, number of warning episodes per cloudy day and their median length, false-alarm
episodes per cloudy day. Definitions of events, probabilities and episodes are those of R-R4. Descriptive.

## H. Reserve (official test set)
1. Out-of-sample constant: the additive constant of R-R5 is fitted on the odd-numbered test days (in date order) and
   applied to the even-numbered days, and the reverse; achieved reliability and mean reserve on the held-out days,
   for method v2, the frozen forecaster, G4, the calm expert, dressed smart persistence and the constant reserve.
2. Equal 95% reliability within cloudy days only: mean reserve on cloudy days for the same forecasters and the
   paired difference v2 - each (day bootstrap over the ten cloudy days, sign-flip p).
Descriptive; the reading rule of R-R5 ("less reserve than X" needs interval below zero and p < 0.05) applies to H2.

## Multiplicity and reporting
All contrasts are exploratory and unadjusted. Every quantity above is reported whichever way it falls, in the
manuscript or its supplement. Result file: `revision/results/r13_explanatory_analyses_method_v2.json` (parts A-H as
keys; the forward part may be written by a separate invocation into the same file). Script:
`src/revision_r13_explanatory.py`.
