# Protocol R-R4: ramp warnings - value curves, direction, reliability and timing

Committed before the analysis is run. Label: **post hoc test** (category ii), stylised decision analysis. No model is
trained; frozen test predictions. R-Y3 results are known (maximum value 0.61 for the calm expert's event probability
with development-optimal thresholds against at most 0.29 for median warnings, both at the event base rate).

## Origin
A simulated review (2026-11-16) objected that the warning value (R-Y3) (a) comes from the calm expert's probability,
not from the forecaster, (b) concerns an event without direction, (c) compares two maxima instead of the curves, with
no uncertainty, (d) gives no timing of the warning, and that no source is cited for the cost-loss model (Murphy 1977;
Richardson 2000).

## Events (frozen test threshold tau = 0.279 in clear-sky index, 15-min endpoint change)
- **Any ramp:** |K(t+15) - K(t)| >= tau (as R-Y3).
- **Falling ramp:** K(t+15) - K(t) <= -tau.
- **Power drop (sensitivity):** PV(t+15) <= PV(t) - 20% of the 30.1 kW capacity.

## Warning sources
Event probability implied by the nine quantiles (piecewise-linear CDF, constant tails) of: the calm expert, the
frozen forecaster G3''-F, the direct expert and the PV-only model P15 (15 members, quantile-averaged). Median warnings
of G3''-F and of the direct expert (median change beyond the threshold). A user with cost-loss ratio a acts when the
probability is at least a (the optimal rule for reliable probabilities; no threshold is tuned). For the calm expert,
additionally: probabilities recalibrated by isotonic regression fitted on development out-of-fold forecasts only.

## Reported
- Relative economic value V(a) (Richardson 2000) on a = 0.02 ... 0.90 for every source and event, with day-block
  bootstrap 95% bands (2,000 resamples, stratified sunny / cloudy), and the differences between sources at each a.
- Base rate, Brier skill against the base rate, reliability table (ten probability bins), hit rate and probability of
  false detection at a = base rate.
- Timing for falling ramps: for each observed falling ramp, the minutes from issue time to the first 1-min record in
  the 15-min window at which K is at least tau below K(t); distribution over all falling ramps and over warned ones
  (probability at least the base rate); share warned with at least 7 min to the crossing. Falling-ramp episodes
  (first forecast of each run of consecutive falling-ramp targets in a day): number, share warned; false-alarm
  episodes per cloudy day.

## Reading rule (fixed now)
- "Probability warnings are worth more than median warnings" is stated for an event only if, for the forecaster's
  own probability against its own median warning, the value difference has a bootstrap interval above zero at the
  base rate and at no cost-loss ratio is the median warning better with an interval above zero; otherwise the range of
  cost-loss ratios in which each is better is reported.
- "Images add warning value" is stated only if the value of the forecaster's probability exceeds that of the PV-only
  model's probability at the base rate with the interval above zero, for the falling-ramp event.
- Ramp-rate control is named as a possible use only if at least half of the falling ramps are warned at least 7 min
  before the crossing; otherwise the manuscript speaks of a warning of a 15-min change only.
Everything is reported whichever way it falls.
