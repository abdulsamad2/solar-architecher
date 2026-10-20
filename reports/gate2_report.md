# Gate 2 report: E4 replication, calibration, computational cost

2026-10-20. Protocol: `reports/gate2_calibration_prereg.md` (commit 7169375, amendment 1 at 308ecee, both before results). Raw numbers: `reports/gate2_results.json`, `reports/gate2_inference_cost.json`.

**Scope:**
- Development folds only (164,362 validation samples, 5 blocked folds). The official 20-day test set was not read.
- All compared models are scored on identical validation samples (asserted).
- Seeds 42, 123 and 456; none selected or dropped.
- **Uncertainty is reported two ways:** training-seed variation as mean ± SD across seeds, and sampling uncertainty as a 95% day-block bootstrap CI within each seed.

---

## 1. Representation gain: E4-B (K quantiles, K inputs) vs E4-A (kW quantiles)

Pooled validation CRPS, kW (mean ± SD over 3 seeds):

| | Overall | Sunny | Cloudy | Ramp (10% of samples) |
|---|---|---|---|---|
| E4-A | 0.946 ± 0.014 | 0.548 ± 0.029 | 1.229 ± 0.014 | 2.583 ± 0.028 |
| **E4-B** | **0.733 ± 0.006** | **0.295 ± 0.012** | **1.045 ± 0.008** | **2.417 ± 0.014** |
| Smart persistence (point mass) | 0.877 | 0.223 | 1.343 | 4.672 |

Paired E4-B − E4-A CRPS, per seed, with 95% day-bootstrap CI:

| Seed | Overall | Sunny | Cloudy | Ramp |
|---|---|---|---|---|
| 42 | −0.220 [−0.244, −0.196] | −0.272 [−0.314, −0.232] | −0.183 [−0.213, −0.153] | −0.193 [−0.230, −0.155] |
| 123 | −0.206 [−0.226, −0.184] | −0.244 [−0.284, −0.206] | −0.178 [−0.204, −0.152] | −0.125 [−0.162, −0.088] |
| 456 | −0.212 [−0.235, −0.189] | −0.243 [−0.278, −0.210] | −0.190 [−0.225, −0.157] | −0.182 [−0.221, −0.142] |
| Seed mean ± SD | −0.212 ± 0.007 | −0.253 ± 0.017 | −0.183 ± 0.006 | −0.167 ± 0.037 |

Uncalibrated 90% coverage: E4-A 0.723 ± 0.024; E4-B 0.867 ± 0.015.
Median-forecast RMSE: E4-A 2.138–2.154; E4-B 2.012–2.035.

**Finding:** the K representation's advantage replicates. Every seed's CI excludes 0 in every stratum, and the seed-to-seed SD is about 30× smaller than the effect.
- **Against persistence:** E4-B is better overall (−16%), on cloudy days (−22%) and on ramps (−48%).
- **Sunny days:** E4-B remains worse than persistence (0.295 vs 0.223).

---

## 2. Calibration gain (E4-B-fit, trained on fit days only; calibrated on held-out calibration days)

### Cost of holding out calibration days
E4-B-fit uncalibrated CRPS is 0.751 ± 0.002, against 0.733 ± 0.006 for E4-B trained on all training days: **+0.018 kW** from training on about 20% fewer days.

### Coverage and width on the untouched validation samples (mean over 3 seeds)
| Stratum | Method | Cov 50% | Cov 80% | Cov 90% | Width 50/80/90 (kW) | CRPS |
|---|---|---|---|---|---|---|
| Overall | E4-B (full train) | 0.330 | 0.755 | 0.867 | 1.12 / 3.09 / 5.45 | 0.733 |
| | E4-B-fit uncalibrated | 0.318 | 0.772 | 0.872 | 1.05 / 3.36 / 5.89 | 0.751 |
| | **+ CQR-kW (primary)** | **0.407** | **0.809** | **0.890** | 1.18 / 3.49 / 6.00 | – |
| | + CQR-K | 0.411 | 0.805 | 0.887 | 1.22 / 3.52 / 6.01 | – |
| | + recalibration | 0.380 | 0.699 | 0.865 | 1.43 / 2.75 / 4.07 | 0.786 |
| Sunny | uncalibrated → CQR-kW | 0.322 → 0.442 | 0.811 → 0.855 | 0.891 → 0.911 | | 0.305 (recal 0.305) |
| Cloudy | uncalibrated → CQR-kW | 0.315 → 0.383 | 0.745 → 0.776 | 0.859 → 0.876 | | 1.068 (recal 1.129) |
| **Ramp** | uncalibrated → CQR-kW | 0.144 → 0.193 | 0.432 → 0.474 | **0.637 → 0.662** | | 2.448 (recal 2.479) |
| Non-ramp | uncalibrated → CQR-kW | 0.338 → 0.431 | 0.810 → 0.847 | 0.899 → 0.916 | | 0.560 (recal 0.597) |
| **Fold 1 (winter)** | uncalibrated → CQR-kW | 0.190 → 0.228 | 0.590 → 0.596 | **0.730 → 0.713** | | 1.315 (recal 1.426) |
| Fold 0 / 2 / 3 / 4 | CQR-kW, 90% | | | 0.930 / 0.922 / 0.908 / 0.936 | | |

Across seeds, the SD of calibrated 90% coverage is ≤ 0.01 in every stratum. No inverted CQR intervals occurred. Recalibration produced no crossing quantiles and no saturated levels.

### Findings
1. **CQR fixes pooled 80/90% coverage** (0.77/0.87 → 0.81/0.89) at a modest width cost (+0.11 kW at 90%). CQR-K and CQR-kW are indistinguishable.
2. **The 50% interval stays too narrow** (0.41 after CQR). The forecast distribution is too peaked in the centre: uncalibrated reliability at τ = .2/.8 is .33/.74. A constant CQR shift cannot fix the shape.
3. **Fold 1 (Sep 2017 to May 2018, winter): calibration does nothing** (90% coverage 0.73 → 0.71).
   - The calibration days, drawn from the fold's training period, show the model as almost calibrated: q̂ at 90% is about −0.01 kW.
   - The fold-1 miscoverage is therefore a **distribution shift** between the validation period and the rest of the data, not a stable miscalibration.
   - Split conformal assumes exchangeability between calibration and validation and cannot correct this.
4. **Ramp-conditioned coverage is poor:** 90% → 0.64 (0.66 after CQR). Marginal CQR improves the pooled rate largely through non-ramp samples (0.92). This is the expected lack of conditional coverage of split conformal; no conditional guarantee is claimed.
5. **Monotone full-quantile recalibration does not transfer.**
   - It worsens CRPS overall (0.751 → 0.786), on cloudy days (1.068 → 1.129) and on fold 1 (1.315 → 1.426).
   - It lowers 80% coverage (0.77 → 0.70).
   - The cause: the calibration-day reliability curve (lower tail too wide there; e.g. τ = .05 is mapped to .135) differs from the validation-period curve.
   - **It is rejected.**

### Gate 2 closing rule (pre-registered): **NOT MET, 0/3 seeds**
| Seed | Cov 50 / 80 / 90 (CQR-kW) | Min fold 90% | Pass |
|---|---|---|---|
| 42 | 0.421 / 0.810 / 0.890 | 0.702 (fold 1) | no |
| 123 | 0.387 / 0.801 / 0.887 | 0.716 (fold 1) | no |
| 456 | 0.413 / 0.816 / 0.894 | 0.722 (fold 1) | no |

- **Conditions met:** 80% and 90% pooled coverage within ±3 pp, in all seeds.
- **Conditions failed:** 50% coverage within ±5 pp (0.39 to 0.42), and every fold ≥ 0.80 at 90% (fold 1 is 0.70 to 0.72), in all seeds.

---

## 3. Computational cost (RTX 5000 Ada)
| Item | Value |
|---|---|
| Parameters (E4-A and E4-B identical) | 90,761 |
| Batch-1 latency | 1.85 ms |
| Batched throughput (batch 1024) | ≈ 10,600 samples/s |
| Calibration fit (CQR ×3 + recalibration, per fold) | 0.8 s |
| Calibration application | 0.28 µs/sample |
| Training: E4-A, 5 folds | 57 ± 4 epochs, 6,020 ± 961 s |
| Training: E4-B, 5 folds | 46 ± 3 epochs, 5,456 ± 352 s |
| Training: E4-B-fit, 5 folds | 46 ± 2 epochs, 3,526 ± 1,669 s |

Training wall times come from logs of runs that shared the GPU two at a time, so they are indicative only. E4-B early-stops sooner than E4-A. Calibration cost is negligible against training and inference.

---

## 4. Conclusions and recommendation
- **Representation:** E4-B (K quantiles, K inputs) is the forecaster of record. Its CRPS gain over E4-A is large and stable across seeds and strata.
- **Calibration:** CQR-kW is a cheap, worthwhile post-processing step for marginal 80/90% coverage. Full-quantile recalibration is rejected.
- **Gate 2 stays open.** The two remaining failures are not fixable by marginal split conformal:
  1. **Temporal/regime shift (fold 1).** An online, adaptive conformal method that updates from recently observed outcomes, e.g. adaptive conformal inference, is the standard tool for non-exchangeable time series. It needs no retraining. Candidate experiment, to be pre-registered first.
  2. **Ramp-conditional under-coverage** (90% → 0.64). This is the failure the SHADOW-MoE event-time/severity decomposition (Phase 6) is designed to address. It is a strong motivation for, and a pre-specified evaluation target of, that phase, rather than something to patch post hoc.
- **Central sharpness (50% coverage):** the predictive distribution is too peaked; a shape problem, not a width problem. Candidates: more central quantile levels (e.g. .25/.4/.6/.75) or a parametric or mixture head. That overlaps with the Phase 6 severity head.
- **Persistence-residual forecasting** remains a separate follow-up for the sunny-day gap.
