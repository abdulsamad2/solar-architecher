# Protocol R-Z3: outcome-class (ramp-protected) conformal sets and their cost

Committed before any run. Labels: **development** (initial scores only, no selection), **post hoc** (test, category ii).
Frozen predictions only; no retraining. Run after R-Z1 and reported whatever R-Z1 finds.
Origin: R-Y2 showed that conditioning on issue-time features does not raise ramp coverage. Whether an outcome is a
ramp is unknown at issue time, but class-conditional conformal prediction can still protect an outcome-defined class
by testing each candidate outcome against the scores of its own class (label-conditional / candidate-response
inversion, as used for response-defined wind-ramp groups).

## Question
What does nominal coverage on ramp outcomes cost in set size for the proposed forecaster?

## Method (G3''-F, 90%)
- Conformity score s = max(lo - y, y - hi) of the raw 90% interval (as in the frozen rule).
- Two classes by the realized outcome: ramp (|y / P_clear - k_now| >= tau) and non-ramp. Each class keeps its own
  rolling window of the most recent matured scores of that class: W_ramp = 660, W_non = 3960 (the frozen W = 4620
  split by the ramp share of about 1/7; fixed here, not tuned). Daily reset; each test day starts from the latest
  matured development out-of-fold scores of the class dated strictly before the day's first issue time (development
  ramp labels from the frozen fold thresholds; the 140 development samples excluded in R-Y2 are excluded here too).
- At issue time, with class quantiles q_ramp and q_non at level 0.90, the prediction set is
  ([lo - q_non, hi + q_non] within the non-ramp region) united with ([lo - q_ramp, hi + q_ramp] within the ramp
  region), the regions being those of R-Z2 (known at issue time). The frozen width floor applies per class.
- Reference U: the unconditional C07 past-only interval (reproduced to 1e-9, else STOP).

## Metrics
Coverage overall, on ramps, rising / falling ramps and non-ramps; mean set size (total length, clipped at 0 kW), mean
length of the convex hull; share of forecasts whose set is not an interval. Paired day-block bootstrap of the set-size
difference to U (overall).

## Decision rule (fixed now)
**Ramp protection attained** if coverage is >= 0.87 on ramps and >= 0.87 on non-ramps; otherwise **not attained**. The
cost is reported as the ratio of the mean set size to the mean width of U. Reported whichever way it falls; the
frozen method is not changed.
