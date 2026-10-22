# Gate 5: official SKIPP'D sealed test evaluation of G3″-F

**This is the one-shot evaluation on the 20 official held-out test days** (13,266 forecasts, 10 sunny / 10 cloudy days). Everything in sections 1–5 is from this test run. Development (dev) numbers appear only in section 6, labelled as such.

**Provenance**
- **Protocol:** `reports/test_eval_prereg.md` (frozen 491fac2, Amendments 1–2).
- **Approval:** commit 77fa2f3. **Executed code:** e4b1048. **Raw results:** cd7c30a (committed unchanged before any interpretation).
- **Raw outputs:**
  - `reports/test_results.json` (SHA-256 in `reports/test_raw_output_hashes.json`);
  - `experiments/final/test_predictions_v1.npz` (raw quantiles for all models, plus SUNSET);
  - `reports/test_execution_log.json` (git commit, timing, GPU memory);
  - `reports/test_execution_manifest.json` (pre-unseal checks, configuration, environment).
- **Deviations:** two aborted attempts before any PV or label was read (`reports/test_execution_deviations.md`). `run_test` completed exactly once.
- **Post-hoc descriptive statistics** (miss rates, per-day results, stratum CIs) come from `src/final_posthoc.py` → `reports/test_posthoc_descriptive.json`.
  - They are computed from the committed predictions with the same frozen functions, and first reproduce the committed CRPS and coverage exactly.
  - They are **not** pre-registered hypothesis tests.

## 1. Pre-registered hypotheses

Paired test-day bootstrap: 10,000 resamples, stratified sunny/cloudy, seed 20261022. Values are differences, G3″-F minus comparator, in kW.

| | Estimate | 95% CI | Rule | Verdict |
|---|---|---|---|---|
| **H1 (sole primary): overall CRPS vs E4-B-fit** | **−0.111** | **[−0.149, −0.075]** | CI entirely < 0 | **CONFIRMED** |
| H2: overall CRPS vs persistence | −0.701 | [−0.823, −0.582] | CI < 0 | Confirmed |
| H2: cloudy-day CRPS vs persistence | −1.208 | [−1.454, −0.971] | CI < 0 | Confirmed |
| H3: ramp CRPS vs E4-B-fit, non-inferiority margin 0.02 | −0.117 | [−0.244, **+0.031**] | upper bound ≤ 0.02 | **NOT established** (upper bound 0.031 > 0.02) |
| H4: RMSE skill vs persistence (descriptive) | G3″-F **18.6%**, SUNSET port 16.2% | – | – | descriptive |
| H5: calibrated 90% coverage | **0.935** | – | within 0.90 ± 0.03 | **NOT met** (over-covers by 3.5 pp) |

**Primary answer:** yes. G3″-F has lower overall CRPS than E4-B-fit on the 20 official test days: 1.006 vs 1.117 kW, a 9.9% reduction. The CI excludes 0.

**Secondary hypotheses**
- **H3:** the ramp point estimate favours G3″-F by 0.117 kW. However, with only 1,916 ramp samples spread over 20 days the interval is wide, so non-inferiority at the 0.02 margin is not shown. It is reported as failed.
- **H5:** the calibrated intervals are conservative overall (too wide), while still under-covering ramps (section 3).

## 2. CRPS, RMSE and MAE by stratum (test, kW)

RMSE and MAE are of the median. SUNSET and persistence are point forecasts, so their CRPS equals their MAE.

| Model | Overall | Sunny | Cloudy | Ramp | Non-ramp | Sunrise ramp | RMSE | MAE |
|---|---|---|---|---|---|---|---|---|
| **G3″-F** | **1.006** | 0.074 | **1.939** | **3.510** | 0.583 | 0.808 | **2.959** | **1.411** |
| E4-B-fit | 1.117 | 0.218 | 2.017 | 3.627 | 0.693 | **0.778** | 3.037 | 1.534 |
| G4 | 1.024 | 0.073 | 1.976 | 3.782 | 0.558 | 0.802 | 3.033 | 1.429 |
| E6-S | 1.090 | **0.073** | 2.108 | 4.281 | **0.551** | 0.900 | 3.274 | 1.499 |
| SUNSET port | 1.719 | 0.475 | 2.965 | 4.815 | 1.197 | 1.070 | 3.046 | 1.719 |
| Smart persistence | 1.707 | 0.269 | 3.147 | 7.131 | 0.791 | 1.610 | 3.636 | 1.707 |

**Sample counts:** overall 13,266; sunny 6,638; cloudy 6,628; ramp 1,916; non-ramp 11,350; sunrise ramp 268. 93.1% of forecasts were routed to G3″; the other 6.9% used the G3 fallback.

**D4 (descriptive, 60-min eligible population, n = 12,355)**

| | Overall | Ramp | Non-ramp |
|---|---|---|---|
| D4 CRPS | 1.180 | 3.737 | 0.786 |
| E4-B-fit CRPS, same samples | 1.172 | 4.091 | 0.722 |
| D4 − E4-B-fit [95% CI] | +0.008 [−0.015, +0.031] | −0.354 [−0.415, −0.273] | +0.064 [+0.040, +0.091] |

This is the same pattern as on dev: solar/motion features help on ramps but cost elsewhere.

**Post-hoc stratum CIs (descriptive), G3″-F − comparator**

| Comparator | Overall | Ramp | Non-ramp | Sunrise ramp |
|---|---|---|---|---|
| E4-B-fit | −0.111 [−0.149, −0.075] | −0.117 [−0.244, +0.031] | −0.110 | **+0.031 [−0.014, +0.090]** |
| G4 | −0.018 [−0.039, +0.002] | **−0.271 [−0.364, −0.173]** | +0.025 [+0.007, +0.045] | – |
| E6-S | −0.084 [−0.108, −0.060] | −0.771 [−0.877, −0.652] | – | – |
| SUNSET port | −0.713 [−0.858, −0.578] | – | – | – |

- **Against E4-B-fit, other strata:** sunny −0.144 [−0.204, −0.097]; cloudy −0.078 [−0.129, −0.026]; up-ramps −0.069 [−0.230, +0.126]; down-ramps −0.163 [−0.299, +0.004].
- **Against G4:** the A2 history branch buys a large ramp gain and pays a small non-ramp cost. Overall the difference is not significant.
- **Sunrise ramps:** G3″-F is slightly *worse* than E4-B-fit (fallback samples, n = 268), though not significantly.

## 3. Calibrated intervals (G3″-F; causal rolling conformal, daily reset, dev-initialised)

Each cell shows coverage, then width (kW) or the share of outcomes falling below the lower / above the upper bound.

| Nominal | Overall | Sunny | Cloudy | Ramp | Sunrise ramp | Width overall (kW) | Width ramp (kW) | Interval score overall | Interval score ramp |
|---|---|---|---|---|---|---|---|---|---|
| 50% | 0.542 | 0.658 | 0.426 | **0.161** | 0.172 | 2.03 | 4.46 | 4.53 | 15.93 |
| 80% | 0.846 | 0.917 | 0.775 | **0.506** | 0.448 | 4.42 | 9.45 | 6.30 | 19.96 |
| 90% | 0.935 | 0.968 | 0.902 | **0.746** | 0.653 | 6.26 | 12.99 | 7.62 | 21.07 |

**Miss rates at 90% (below lower / above upper)**

| Stratum | Below lower | Above upper |
|---|---|---|
| Overall | 0.022 | 0.043 |
| Ramp | 0.106 | 0.148 |
| Up-ramps | 0.000 | 0.302 |
| Down-ramps | 0.208 | 0.000 |
| Sunrise ramp | 0.146 | 0.201 |

- **Ramp misses are directional:** the forecast lags the ramp. This is the persistence-like behaviour already identified on dev.
- **Raw coverage before calibration:** 90% → 0.930, 50% → 0.491. Calibration changed little on test.
- **E4-B-fit:** 90% coverage 0.927 with width 8.20 kW. G3″-F intervals are 24% narrower overall at higher coverage, and narrower still on sunny days (0.82 vs 4.10 kW).
- **E6-S and G4:** 90% coverage 0.927 / 0.934, ramp 0.681 / 0.715.

## 4. Per-day CRPS (test, kW)

G3″-F beats E4-B-fit on **19 of 20 days**; the exception is 2017-09-20 (cloudy), 2.567 vs 2.483. It beats persistence and SUNSET on all 20 days.

| Day | Sunny | Ramps | G3″-F | E4-B-fit | G4 | Persistence | SUNSET |
|---|---|---|---|---|---|---|---|
| 2017-06-24 | – | 225 | 1.681 | 1.722 | 1.759 | 2.812 | 2.602 |
| 2017-09-15 | ✓ | 28 | 0.102 | 0.344 | 0.101 | 0.286 | 0.654 |
| 2017-09-20 | – | 194 | 2.567 | 2.483 | 2.502 | 3.896 | 3.403 |
| 2017-10-06 | ✓ | 20 | 0.073 | 0.152 | 0.073 | 0.307 | 0.462 |
| 2017-10-11 | – | 60 | 1.263 | 1.377 | 1.294 | 1.887 | 1.700 |
| 2017-10-22 | ✓ | 8 | 0.090 | 0.433 | 0.083 | 0.351 | 0.312 |
| 2018-01-25 | – | 197 | 3.367 | 3.392 | 3.323 | 5.146 | 5.285 |
| 2018-02-16 | ✓ | 1 | 0.088 | 0.169 | 0.078 | 0.333 | 0.741 |
| 2018-03-09 | – | 108 | 1.551 | 1.573 | 1.602 | 2.602 | 2.524 |
| 2018-06-12 | ✓ | 6 | 0.063 | 0.169 | 0.062 | 0.185 | 0.200 |
| 2018-06-23 | ✓ | 7 | 0.059 | 0.205 | 0.058 | 0.187 | 0.255 |
| 2018-10-04 | – | 248 | 2.465 | 2.530 | 2.487 | 4.150 | 3.810 |
| 2019-01-25 | ✓ | 8 | 0.105 | 0.365 | 0.107 | 0.419 | 1.100 |
| 2019-05-27 | – | 183 | 2.064 | 2.177 | 2.044 | 2.991 | 3.026 |
| 2019-06-23 | ✓ | 24 | 0.051 | 0.130 | 0.051 | 0.180 | 0.314 |
| 2019-06-28 | – | 94 | 0.774 | 0.838 | 0.786 | 1.433 | 1.259 |
| 2019-07-14 | ✓ | 32 | 0.069 | 0.150 | 0.069 | 0.210 | 0.395 |
| 2019-08-10 | – | 263 | 2.020 | 2.208 | 2.174 | 3.705 | 3.491 |
| 2019-10-14 | ✓ | 12 | 0.060 | 0.129 | 0.061 | 0.319 | 0.543 |
| 2019-10-19 | – | 198 | 2.119 | 2.344 | 2.249 | 3.541 | 3.257 |

## 5. Inference cost

These figures come from `reports/test_execution_log.json`.
- **Complete one-shot run:** **475 s** wall time.
  - This covers test image features, 15 members × 5 models (60 checkpoint passes plus 30 gate blends), ensembling, scoring, three calibration levels × 4 models, and 10,000-resample bootstraps.
  - Amortised: 35.8 ms per forecast for all models together. Peak GPU memory: 5.8 GB (NVIDIA RTX 5000 Ada; full environment in the manifest).
- **Single-forecast latency** for G3″-F alone was **not** measured. The run is batched, and the protocol did not specify a per-request latency benchmark.

## 6. Development vs official test (both from the frozen pipeline)

Dev uses held-out 3-member ensembles over 497 dev days (164,362 samples). Test uses the 15-member ensemble over 20 test days (13,266 samples).

| Quantity | Dev | Test |
|---|---|---|
| G3″-F overall CRPS | 0.573 | 1.006 |
| E4-B-fit overall CRPS | 0.730 | 1.117 |
| H1 Δ [95% CI] | −0.157 [−0.170, −0.145] | −0.111 [−0.149, −0.075] |
| H3 ramp Δ [95% CI] | −0.025 [−0.054, +0.005] (NI met) | −0.117 [−0.244, +0.031] (NI not met) |
| H4 RMSE skill vs persistence | 17.0% | 18.6% |
| 90% coverage, overall / ramp | 0.901 / 0.651 | 0.935 / 0.746 |
| Routing to G3″ | 91.7% | 93.1% |

- **Absolute CRPS is higher on test** because the test days are harder: 14.4% of samples are ramps vs 10.0% on dev, and cloudy-day CRPS is 1.94 vs 0.89. Dev is 41.6% clear-day samples by its own label, and its "cloudy" class covers every non-clear day.
- **The relative improvement holds:** 9.9% on test vs 21.5% on dev. The sunny-day advantage replicated strongly.
- **Ramp:** the point estimate is more favourable on test, but with only 20 days the CI is about 5× wider, so H3 non-inferiority fails on precision rather than on the estimate.
- **Coverage moved from nominal on dev to conservative on test:** the dev-initialised windows carry over the dev error distribution, which differs from the cloudier test days.

## 7. Failures and deviations
- **Execution:**
  - Attempt 1 failed because my wrapper dirtied the tree, before any test data was read.
  - Attempt 2 failed because the test HDF5 is not in chronological order. It had read the test images (inputs only); no PV or labels were read.
  - Both fixes were approved by the user and validated on dev (bit-identical features). No model, threshold, calibration setting or scoring rule changed. See `reports/test_execution_deviations.md`.
- **Pre-registration:** Amendments 1 (TF32 backend) and 2 (bf16-aware reproduction criterion, adopted after a failed check) stay in force as written.
- **Outcome failures:**
  - H3 ramp non-inferiority was not established.
  - H5 calibration over-covers overall, and ramps remain under-covered (50% interval: 0.16).
  - Sunrise-ramp forecasts in the fallback population are no better than E4-B-fit.

## 8. Claims and limits
- **Supported claim:** on the official SKIPP'D 20-day test set, under this repository's protocol, G3″-F improves overall probabilistic accuracy over the strongest single expert (E4-B-fit) and over smart persistence. Its median beats our SUNSET port on RMSE (2.959 vs 3.046 kW).
- **Not claimed: superiority over published SKIPP'D/SUNSET results.**
  - Our sample matching (±30 s, forecast manifest v2) differs from the original benchmark.
  - The published SUNSET RMSE (3.029) was reproduced only approximately (3.046).
  - No winner was re-selected after viewing these results.
- **Not claimed:** improved ramp forecasting relative to E4-B-fit, or reliable ramp-conditioned uncertainty. Both remain open problems; ramp detection (AUPRC ≈ 0.5 on dev) is the bottleneck.
