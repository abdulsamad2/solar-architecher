# Paired comparison on blocked dev folds (validation only)

164,362 validation samples over 497 fold-days; identical samples across models.

## overall RMSE (kW) per fold

| Fold | E2 | E3 | E3b | persistence |
|---|---|---|---|---|
| 0 | 1.917 (ep 5) | 2.127 (ep 1) | **1.881 (ep 2)** | 2.009 |
| 1 | **2.991 (ep 4)** | 3.534 (ep 2) | 3.114 (ep 3) | 3.434 |
| 2 | **1.799 (ep 4)** | 2.008 (ep 2) | 1.851 (ep 1) | 1.913 |
| 3 | 2.131 (ep 5) | 2.343 (ep 4) | **2.131 (ep 3)** | 2.396 |
| 4 | 1.501 (ep 4) | 1.523 (ep 3) | **1.376 (ep 3)** | 1.618 |
| pooled | 2.074 | 2.329 | 2.087 | 2.289 |

## sunny RMSE (kW) per fold

| Fold | E2 | E3 | E3b | persistence |
|---|---|---|---|---|
| 0 | 0.822 (ep 5) | 1.120 (ep 1) | 0.967 (ep 2) | **0.460** |
| 1 | 0.675 (ep 4) | 1.241 (ep 2) | 0.873 (ep 3) | **0.382** |
| 2 | 0.685 (ep 4) | 0.850 (ep 2) | 0.697 (ep 1) | **0.380** |
| 3 | 0.620 (ep 5) | 0.838 (ep 4) | 0.685 (ep 3) | **0.383** |
| 4 | 0.761 (ep 4) | 0.719 (ep 3) | 0.501 (ep 3) | **0.354** |
| pooled | 0.726 | 0.919 | 0.727 | 0.390 |

## cloudy RMSE (kW) per fold

| Fold | E2 | E3 | E3b | persistence |
|---|---|---|---|---|
| 0 | 2.339 (ep 5) | 2.546 (ep 1) | **2.258 (ep 2)** | 2.515 |
| 1 | **3.466 (ep 4)** | 4.054 (ep 2) | 3.594 (ep 3) | 3.999 |
| 2 | **2.397 (ep 4)** | 2.650 (ep 2) | 2.468 (ep 1) | 2.615 |
| 3 | 2.560 (ep 5) | 2.793 (ep 4) | **2.552 (ep 3)** | 2.906 |
| 4 | 2.185 (ep 4) | 2.245 (ep 3) | **2.090 (ep 3)** | 2.525 |
| pooled | 2.643 | 2.947 | 2.661 | 2.977 |

## Paired pooled RMSE differences, 95% day-block bootstrap CI (stratified clear/cloudy)

| Comparison | Overall | Clear days | Cloudy days |
|---|---|---|---|
| E3 - E2 | +0.255 [+0.189, +0.328] | +0.193 [+0.114, +0.271] | +0.303 [+0.219, +0.401] |
| E3b - E2 | +0.013 [-0.022, +0.053] | +0.001 [-0.064, +0.066] | +0.018 [-0.029, +0.068] |
| E3b - E3 | -0.242 [-0.287, -0.199] | -0.192 [-0.228, -0.156] | -0.286 [-0.351, -0.230] |
| E2 - persistence | -0.215 [-0.274, -0.159] | +0.336 [+0.292, +0.379] | -0.334 [-0.407, -0.266] |
| E3 - persistence | +0.040 [-0.049, +0.133] | +0.529 [+0.440, +0.615] | -0.030 [-0.145, +0.101] |
| E3b - persistence | -0.202 [-0.262, -0.138] | +0.337 [+0.263, +0.412] | -0.316 [-0.393, -0.235] |

Negative = first model better. Fold wins (first model lower RMSE):

- E3 < E2: overall 0/5, clear 1/5, cloudy 0/5
- E3b < E2: overall 3/5, clear 1/5, cloudy 3/5
- E3b < E3: overall 5/5, clear 5/5, cloudy 5/5
- E2 < persistence: overall 5/5, clear 0/5, cloudy 5/5
- E3 < persistence: overall 2/5, clear 0/5, cloudy 2/5
- E3b < persistence: overall 5/5, clear 0/5, cloudy 5/5

## Gate 1 assessment and recommendation

**Status of the evidence.**
- E2 and E3 were pre-specified.
- E3b is a **post-hoc diagnostic extension**, designed after E3 folds 0 and 1 were seen.
- Single seed (42).
- Validation folds only; the official test set was not used.

**Findings.**
1. **Target-only normalisation (E3) hurts.**
   - It is worse than E2 on 5/5 folds, by +0.255 kW overall [+0.19, +0.33]. Every stratum's interval excludes zero.
   - It is no better than smart persistence overall.
2. **The E3 failure comes from the input/target mismatch, not from the clear-sky target itself.**
   - Feeding K history (E3b) recovers -0.242 kW against E3 [-0.29, -0.20], on 5/5 folds and in every stratum.
3. **Properly normalised (E3b) vs raw kW (E2): no detectable difference.**
   - Overall +0.013 kW [-0.022, +0.053]; cloudy +0.018 [-0.029, +0.068]; clear +0.001 [-0.064, +0.066].
   - Folds split 3/5 vs 2/5.
   - E3b peaks earlier (epochs 1 to 3 vs 4 to 5), i.e. it overfits sooner under the same budget.
4. **Both E2 and E3b beat smart persistence** overall (about -0.21 kW) and on cloudy days (about -0.32 kW), on 5/5 folds.
   **Both lose on clear days** (about +0.34 kW, 0/5 folds). This sunny regression is the dominant weakness, shared with SUNSET.

**Recommendation.**
- **Do not claim** that clear-sky normalisation improves point accuracy. The Gate 1 criterion "improves cloudy error without unacceptable sunny regression" is **not met**: E3b is statistically indistinguishable from E2.
- **Do not use target-only normalisation (E3)** in any later experiment.
- **Retain physical normalisation in the pipeline, in its consistent form (E3b: K inputs + K target), as a representation choice rather than a claimed contribution.**
  - The Phase 6 ramp/event labels are defined on ΔK, so the anchor is needed regardless.
  - K-space may matter for the distributional models: forecast spread scales with P_clear.
  - That is a testable hypothesis for E4, not a result.
- **For E4 (direct quantiles), carry both representations** (kW as in E2, and K as in E3b) with identical heads. Decide on cloudy CRPS and calibration at Gate 2.
- **Before final claims:** repeat E2 and E3b with seeds 123 and 456, as the roadmap requires. A ~0.01 kW difference is well within plausible seed-to-seed variation.
- **Suggested follow-up for the shared sunny weakness** (not yet run): predict the residual from smart persistence instead of the absolute value. Under such a formulation, a model that cannot improve on a clear-sky trajectory defaults to it.
