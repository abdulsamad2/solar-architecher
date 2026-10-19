# Paired comparison on blocked dev folds (validation only)

164,362 validation samples over 497 fold-days; identical samples across models.

## overall RMSE (kW) per fold

| Fold | E4A | E4B | persistence |
|---|---|---|---|
| 0 | 1.906 (ep 7) | **1.781 (ep 3)** | 2.009 |
| 1 | 3.164 (ep 3) | **2.974 (ep 3)** | 3.434 |
| 2 | **1.699 (ep 5)** | 1.779 (ep 4) | 1.913 |
| 3 | 2.209 (ep 6) | **2.088 (ep 5)** | 2.396 |
| 4 | 1.760 (ep 6) | **1.376 (ep 6)** | 1.618 |
| pooled | 2.152 | 2.012 | 2.289 |

## sunny RMSE (kW) per fold

| Fold | E4A | E4B | persistence |
|---|---|---|---|
| 0 | 0.897 (ep 7) | 0.757 (ep 3) | **0.460** |
| 1 | 0.991 (ep 3) | 0.751 (ep 3) | **0.382** |
| 2 | 0.683 (ep 5) | 0.501 (ep 4) | **0.380** |
| 3 | 0.918 (ep 6) | 0.404 (ep 5) | **0.383** |
| 4 | 1.203 (ep 6) | 0.460 (ep 6) | **0.354** |
| pooled | 0.969 | 0.565 | 0.390 |

## cloudy RMSE (kW) per fold

| Fold | E4A | E4B | persistence |
|---|---|---|---|
| 0 | 2.308 (ep 7) | **2.175 (ep 3)** | 2.515 |
| 1 | 3.642 (ep 3) | **3.440 (ep 3)** | 3.999 |
| 2 | **2.254 (ep 5)** | 2.409 (ep 4) | 2.615 |
| 3 | 2.613 (ep 6) | **2.528 (ep 5)** | 2.906 |
| 4 | 2.364 (ep 6) | **2.105 (ep 6)** | 2.525 |
| pooled | 2.694 | 2.590 | 2.977 |

## Paired pooled RMSE differences, 95% day-block bootstrap CI (stratified clear/cloudy)

| Comparison | Overall | Clear days | Cloudy days |
|---|---|---|---|
| E4B - E4A | -0.139 [-0.171, -0.107] | -0.403 [-0.468, -0.341] | -0.104 [-0.144, -0.065] |
| E4A - persistence | -0.138 [-0.194, -0.083] | +0.578 [+0.522, +0.635] | -0.283 [-0.356, -0.219] |
| E4B - persistence | -0.277 [-0.329, -0.225] | +0.175 [+0.128, +0.225] | -0.387 [-0.456, -0.321] |

Negative = first model better. Fold wins (first model lower RMSE):

- E4B < E4A: overall 4/5, clear 5/5, cloudy 4/5
- E4A < persistence: overall 4/5, clear 0/5, cloudy 5/5
- E4B < persistence: overall 5/5, clear 0/5, cloudy 5/5

## CRPS (kW, interpolated-quantile; persistence = point mass) per fold

| Fold | E4A overall / clear / cloudy | E4B overall / clear / cloudy | persistence overall / clear / cloudy |
|---|---|---|---|
| 0 | 0.892 / 0.526 / 1.111 | 0.718 / 0.402 / 0.908 | 0.764 / 0.238 / 1.079 |
| 1 | 1.461 / 0.659 / 1.750 | 1.284 / 0.541 / 1.552 | 1.524 / 0.263 / 1.979 |
| 2 | 0.692 / 0.382 / 0.972 | 0.620 / 0.272 / 0.935 | 0.731 / 0.219 / 1.194 |
| 3 | 0.972 / 0.576 / 1.164 | 0.736 / 0.216 / 0.987 | 0.971 / 0.230 / 1.330 |
| 4 | 0.908 / 0.758 / 1.134 | 0.469 / 0.251 / 0.798 | 0.556 / 0.198 / 1.097 |
| pooled | 0.959 / 0.582 / 1.228 | 0.739 / 0.309 / 1.045 | 0.877 / 0.223 / 1.343 |

## Paired pooled CRPS differences, 95% day-block bootstrap CI

| Comparison | Overall | Clear days | Cloudy days |
|---|---|---|---|
| E4B - E4A | -0.220 [-0.244, -0.196] | -0.272 [-0.314, -0.232] | -0.183 [-0.213, -0.153] |
| E4A - persistence | +0.082 [+0.046, +0.120] | +0.359 [+0.321, +0.397] | -0.115 [-0.175, -0.060] |
| E4B - persistence | -0.138 [-0.171, -0.107] | +0.086 [+0.064, +0.110] | -0.298 [-0.352, -0.246] |

## Calibration and Gate 2 assessment (seed 42; validation folds only; 2026-10-20)

### Pooled probabilistic scores (kW), 164,362 samples
| | CRPS overall / sunny / cloudy | CRPS skill vs persistence (overall / sunny / cloudy) | Coverage 50 / 80 / 90% | Width 50 / 80 / 90% (kW) |
|---|---|---|---|---|
| E4-A (kW) | 0.959 / 0.582 / 1.228 | −9.4% / −161% / +8.6% | 0.290 / 0.572 / 0.749 | 1.35 / 2.69 / 3.96 |
| **E4-B (K)** | **0.739 / 0.309 / 1.045** | **+15.7%** / −38.8% / **+22.2%** | 0.320 / 0.740 / 0.853 | 1.15 / 3.00 / 5.30 |
| Smart persistence (point mass) | 0.877 / 0.223 / 1.343 | – | – | – |
| E2 point mass, 3 seeds (= MAE) | 1.132–1.215 / 0.539–0.629 / 1.554–1.632 | – | – | – |
| E3b point mass, 3 seeds (= MAE) | 1.043–1.110 / 0.415–0.505 / 1.491–1.541 | – | – | – |

Reliability, observed P(y ≤ Q_τ) at τ = .05/.10/.20/.30/.50/.70/.80/.90/.95:
- **E4-A:** .088/.215/.369/.452/.573/.675/.724/.787/.837. The median is too high and the upper tail too narrow.
- **E4-B:** .022/.070/.334/.404/.522/.651/.722/.810/.874. The centre is too narrow (the 50% interval covers 32%); the tails are closer to nominal but still too narrow at the top.

Per-fold 90% coverage:
- E4-A: .745/.616/.928/.666/.752.
- E4-B: .942/.668/.872/.859/.875. Fold 1 (winter) is the worst for both.

### Findings
1. **E4-B dominates E4-A on CRPS.**
   - Paired differences: overall −0.220 [−0.244, −0.196], sunny −0.272 [−0.314, −0.232], cloudy −0.183 [−0.213, −0.153].
   - Lower CRPS on 5/5 folds.
   - Contrast with Gate 1: for point forecasts, K vs kW made no detectable difference. For distributional forecasts, the K representation is clearly better. This is consistent with the hypothesis recorded at Gate 1, that forecast spread scales with P_clear.
2. **Primary Gate 2 metric (cloudy CRPS):** E4-B 1.045, against 1.228 (E4-A), 1.343 (persistence) and ≥ 1.49 (best deterministic model as a point mass).
3. **Both are competitive with the deterministic baselines.** E4-B also beats persistence overall (+15.7% CRPS skill). It still loses to persistence on clear days (0.309 vs 0.223), though by much less than any earlier model.
4. **Intervals are not yet reliable.** Both models are over-confident. E4-B's nominal 90/80/50% intervals cover 85/74/32%, and only 67% at 90% on fold 1. E4-A is worse (75/57/29%).

### Gate 2 status: NOT CLOSED
- **Met:** "competitive with deterministic baselines".
- **Not met:** "reliable interval estimates" (roadmap Gate 2).
- **Representation decision:** carry **E4-B (K representation)** forward. E4-A is dropped as a candidate and kept only as a reference.
- **Required before Phase 6:**
  - (a) Replicate E4-A/E4-B with seeds 123 and 456, as for Gate 1.
  - (b) A pre-registered calibration step, fitted on training days only, then re-check coverage. For example, conformalised quantile regression (CQR) using a held-out block of each fold's *training* days.
