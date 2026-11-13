# Protocol R-Y3: operational value (reserve margin and ramp warnings)

Committed before any run. Labels: **post hoc** (test), development for thresholds only.

## Questions
1. **Reserve.** If PV is scheduled at the forecast median, how much upward reserve is needed to cover shortfalls with 95% reliability, per model?
2. **Ramp warnings.** Do warnings derived from event probabilities have more economic value than warnings derived from the forecast median?

## Models
G3''-F (proposed), E4-B-fit, E6-S, G4, and smart persistence. The development quantities are reconstructed as in R-Y2, with the same STOP check, extended to E4-B-fit, E6-S and G4 against their saved conformity scores.

## Reserve (one-sided)
- Upward reserve r_t = median_t - L_t, where L_t is the lower bound at nominal one-sided level 0.95 (raw quantiles shifted by a past-only rolling one-sided conformal correction, daily reset, W as frozen per model).
- Persistence uses a constant reserve: the 95th percentile of its development shortfalls (persistence - y)+, applied on test.
- An oracle-free static reference: a constant reserve per model from development, shortfall quantile of (median - y).
- Report on test, overall and on cloudy days: achieved reliability P(y >= median - r), mean reserve (kW), the 95th percentile of reserve, and unserved energy E[(median - r - y)+].
- Primary comparison: mean reserve of G3''-F vs each model at achieved reliability >= 0.94 (paired day-block bootstrap of the reserve difference).

## Ramp warnings (relative economic value, cost-loss model)
- Event: endpoint ramp, |K(t+15) - K(t)| >= tau_r (frozen thresholds).
- Probability warnings use the endpoint probability g from the ensemble quantiles, as in C18.
- Median warnings fire when |median K(t+15) - K(t)| >= tau_r.
- Value V(alpha) (Richardson 2000) for cost-loss ratios alpha in 0.01–0.99. For probabilistic warnings, use the alpha-optimal probability threshold chosen on development. Report the value envelope, max V and the range of alpha with V > 0, on test.

## Interpretation (fixed now)
Descriptive. A model "needs less reserve" if the bootstrap CI of the reserve difference lies below 0 at matched reliability. Probability warnings "add value" if their V exceeds the median warnings' V over the alpha range where either is positive.
