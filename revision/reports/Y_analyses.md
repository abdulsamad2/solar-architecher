# R-Y1 to R-Y3: disagreement mechanism, conditional calibration, operational value

All three use the frozen predictions only, with no retraining. Each protocol was committed and pushed (b656b51) before any run. Results: `revision/results/y1_disagreement_mechanism.json`, `y2_conditional_conformal.json`, `y3_operational_value.json`. Labels: development is category iii; the official test is post hoc (category ii).

## R-Y1: does expert disagreement explain the probability-mixture advantage? Verdict: supported

For the CRPS, a two-component linear pool satisfies CRPS(LP) = w CRPS(A) + (1 − w) CRPS(R) − w(1 − w) D. Here D is the Cramér divergence between the two experts; the quantile blend has no such term. The identity holds to 4e-15 kW.

| | Dev seed 42 | Dev seed 123 | Dev seed 456 | Test |
|---|---|---|---|---|
| Spearman correlation (decile of w(1−w)D vs mean advantage) | 1.00 | 1.00 | 1.00 | 0.92 |
| Top − bottom decile advantage (kW) | +0.054 [+0.047, +0.062] | +0.057 [+0.047, +0.068] | +0.052 [+0.044, +0.059] | +0.049 [+0.028, +0.070] |
| Share of the total advantage in the top 20% of D | 0.82 | 0.84 | 0.79 | 0.75 |
| Mean D, ramp / non-ramp | 0.66 / 0.26 | 0.71 / 0.30 | 0.65 / 0.28 | 0.71 / 0.29 |

On the test, the mixture's advantage is +0.062 kW on ramps against +0.0005 kW on non-ramp samples.

## R-Y2: conditional (Mondrian) conformal calibration. Verdict: does not improve ramp coverage

| 90% intervals, test | Coverage overall | Coverage ramp | Width (kW) | Interval score |
|---|---|---|---|---|
| U: unconditional, past-only (C07) | 0.926 | 0.746 | 6.23 | 7.60 |
| M-d: terciles of disagreement d (W_b = 990) | 0.928 | 0.750 | 6.31 | 7.63 |
| M-w: terciles of the raw 90% width (W_b = 990) | 0.930 | 0.747 | 6.29 | 7.62 |

- Coverage within the disagreement terciles is 0.930 / 0.918 / 0.933 under M-d, against 0.926 / 0.947 / 0.913 unconditional.
- Ramp misses are directional: on rising ramps, 30% of outcomes fall above the interval.
- Interpretation: calibration conditioned on forecast-time state achieves conditional coverage on that state. It cannot remove errors of location (lag) on outcome-defined ramps.

**Deviations, all fixed before any test quantity was computed:**
1. The reconstruction tolerance is the project's frozen criterion (99.9% within 1e-4 kW, max 5e-3) instead of 1e-6.
2. 140 development samples that qualify for the 60-min branch but have no stored A2 prediction are left out of the development features and pools.
3. The W_b rehearsal uses past-only initial scores.

## R-Y3: operational value (post hoc test)

**Reserve.** PV is scheduled at the median; the reserve covers a one-sided 95% lower bound calibrated past-only.

| Model | Reliability | Mean reserve (kW) | Cloudy: reliability / reserve |
|---|---|---|---|
| G3''-F | 0.959 | 2.88 | 0.957 / 5.46 |
| E4-B-fit | 0.965 | 4.45 | 0.955 / 6.09 |
| E6-S | 0.952 | 2.95 | 0.942 / 5.59 |
| G4 | 0.953 | 2.93 | 0.953 / 5.55 |
| Smart persistence, constant reserve | 0.889 | 2.01 | 0.784 / 2.01 |

Reserve difference, G3''-F minus each model: E4-B-fit −1.57 [−1.75, −1.35]; E6-S −0.07 [−0.18, +0.05]; G4 −0.04 [−0.08, +0.00] kW.

**Ramp warnings (relative economic value, cost–loss model).**

| Warning source | Max value | Range with positive value |
|---|---|---|
| Endpoint probability g (thresholds chosen on development data) | 0.61 | 0.01–0.90 |
| Best median-based warning (direct expert) | 0.29 | 0.11–0.63 |

Probability warnings are better on 68% of the cost–loss ratios; median warnings are better between 0.38 and 0.66. Under the protocol's rule ("better over the whole range"), the verdict is "partly".

## Not run
- **R-Y5 (other horizons).** Needs new manifests, thresholds and calm-expert event classes, about 40 GPU hours. Not started; a reduced 30-min version is proposed separately.
- **Second-site replication.** Not run.
