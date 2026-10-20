# E4-CAL2 report: online adaptive calibration under temporal shift

2026-10-20.
- **Protocol:** `reports/e4_cal2_prereg.md` (commit df2b0a8; amendment 1, ACI anti-windup, at 614edd5; both before any validation outcome).
- **Code:** `src/cal2.py`, `src/cal2_run.py` (commit 6506a07).
- **Raw results:** `reports/e4_cal2_results.json`. Per-sample intervals, chosen parameters and chronological update traces are in `experiments/E4_cal2/v1/seed*/fold_*/`.

**The Gate 2 verdict (NOT MET, `reports/gate2_report.md`) is preserved and is not re-opened by this experiment.**

**Scope:**
- E4-B-fit predictions reused (no retraining), seeds 42/123/456, 5 development folds.
- Same 164,362 fold-validation samples as Gate 2 (asserted).
- The official test set was not read.
- No stop condition triggered:
  - every online update used only outcomes observable at the issue time (asserted per update);
  - hyperparameters were chosen on calibration days only and written to disk before validation replay;
  - evaluation samples are identical;
  - no non-finite or inverted interval.

## Chosen hyperparameters (calibration data only)
- **ACI γ:** 0.005 (folds 0 to 2) and 0.01 (folds 3 and 4), identical across seeds.
- **Rolling window W:** mostly 990 samples (≈ 3 days); 330 on some fold/seed combinations and 4620 on one.

## Results (mean over 3 seeds; IS = mean interval score, lower is better)
| Stratum | Method | Coverage 50 / 80 / 90 | Width 50 / 80 / 90 (kW) | IS 50 / 80 / 90 |
|---|---|---|---|---|
| **Overall** | uncalibrated | 0.318 / 0.772 / 0.872 | 1.05 / 3.36 / 5.89 | 3.36 / 5.17 / 7.52 |
| | fixed CQR-kW | 0.407 / 0.809 / 0.890 | 1.18 / 3.49 / 6.00 | 3.32 / 5.17 / 7.54 |
| | ACI | **0.500 / 0.800 / 0.898** | 1.58 / 3.54 / 6.07 | **3.21 / 4.95** / 7.31 |
| | rolling conformal | **0.504 / 0.793 / 0.890** | 1.52 / 3.45 / 5.94 | 3.26 / 4.98 / **7.28** |
| **Fold 1** (Sep 2017 to May 2018) | uncalibrated | 0.190 / 0.590 / 0.730 | 1.37 / 4.72 / 8.10 | 6.01 / 8.94 / 12.02 |
| | fixed CQR-kW | 0.228 / 0.596 / 0.713 | 1.48 / 4.75 / 8.02 | 5.95 / 8.91 / 12.15 |
| | ACI | 0.501 / 0.798 / **0.891** | 2.70 / 5.76 / 9.07 | 5.62 / 8.37 / 11.31 |
| | rolling conformal | 0.513 / 0.804 / **0.900** | 2.51 / 5.53 / 8.84 | 5.69 / 8.37 / 11.19 |
| Sunny | fixed CQR → ACI / rolling | 0.442/0.855/0.911 → 0.507/0.809/0.905 / 0.518/0.808/0.897 | | IS90 4.47 → 4.15 / 4.15 |
| Cloudy | fixed CQR → ACI / rolling | 0.383/0.776/0.876 → 0.495/0.795/0.894 / 0.494/0.782/0.884 | | IS90 9.73 → 9.56 / 9.51 |
| **Ramp** | fixed CQR → ACI / rolling | 0.193/0.474/**0.662** → 0.264/0.474/**0.649** / 0.267/0.431/**0.613** | | IS90 17.81 → 17.66 / 17.72 |
| Non-ramp | fixed CQR → ACI / rolling | 0.431/0.847/0.916 → 0.526/0.837/0.926 / 0.530/0.833/0.921 | | IS90 6.39 → 6.15 / 6.11 |

**Per-fold 90% coverage:**
- **ACI:** 0.903 / 0.891 / 0.901 / 0.896 / 0.899.
- **Rolling:** 0.894 / 0.900 / 0.885 / 0.878 / 0.893.
- **Fixed CQR:** 0.930 / 0.713 / 0.922 / 0.908 / 0.936.

The seed-to-seed SD of 90% coverage is ≤ 0.017 in every stratum.

### Paired differences vs fixed CQR-kW (95% day-block bootstrap, stratified sunny/cloudy; per seed 42 / 123 / 456)
| Comparison | ΔIS, 90% overall | ΔIS, 90% fold 1 | Δcoverage, 90% fold 1 | Δcoverage, 90% ramp |
|---|---|---|---|---|
| ACI | −0.236 [−0.31, −0.17] / −0.244 [−0.32, −0.17] / −0.221 [−0.28, −0.17] | −0.90 / −0.92 / −0.73 (all CIs < 0) | +0.187 / +0.175 / +0.170 (all CIs > 0) | −0.018 / −0.021 / −0.000 |
| Rolling | −0.282 [−0.36, −0.21] / −0.261 [−0.34, −0.19] / −0.233 [−0.29, −0.17] | −1.13 / −0.95 / −0.82 (all CIs < 0) | +0.197 / +0.184 / +0.178 (all CIs > 0) | −0.052 / −0.045 / −0.048 (all CIs < 0) |

Interval-score gains at 50% and 80% are also negative with CIs below 0 in every seed.

### Time-resolved coverage (`reports/e4_cal2_fold1_weekly.png`)
Fold-1 90% coverage by month (seed mean):

| Method | Sep 2017 | Oct | Nov | Dec | Jan 2018 | Feb | Mar | May |
|---|---|---|---|---|---|---|---|---|
| fixed CQR | 0.43 | 0.54 | 0.81 | 0.69 | 0.83 | 0.86 | 0.88 | 0.96 |
| ACI | 0.87 | 0.89 | 0.89 | 0.89 | 0.89 | 0.90 | 0.90 | 0.91 |
| rolling | 0.89 | 0.91 | 0.93 | 0.85 | 0.90 | 0.90 | 0.90 | 0.90 |

**Correction to earlier wording:** the fold-1 failure is concentrated in **September–October 2017** (weekly coverage as low as 0.3 to 0.6), not in winter. Gate 2 described fold 1 as "winter". That report is left unchanged, and this correction is recorded here.

The cause is not established. Candidate hypotheses for a dedicated audit:
- conditions absent from the training period, e.g. smoke from the October 2017 Northern California wildfires;
- a data or acquisition change early in the dataset.

### Width-floor events
Degenerate (point) intervals from the pre-registered width floor:
- **ACI:** 9,644 events; **rolling:** 13,406 events. Each is across 3 seeds × 3 levels × 164,362 samples, i.e. 0.7–0.9% of issued intervals.
- They occur mainly in calm conditions, when recent scores are strongly negative. They count as misses when y differs from the point.
- **Not an inversion; no stop condition.**

## Decision (pre-registered rule)
Both methods qualify in **3/3 seeds**:
- the pooled 90% interval score is lower than fixed CQR, with CI < 0;
- fold-1 90% coverage improves.

**Recommendation:** **causal rolling-window conformal**, which has the lower seed-mean pooled IS90 (7.283 vs 7.308 for ACI). ACI is a close, equally valid alternative and slightly better on ramp coverage.

- **Retain** adaptive conformal (rolling window, W selected on calibration days, typically ≈ 3 days) as the **deployment-time calibration component** for E4-B and later models.
  - It needs only the stream of realised PV and costs microseconds per forecast.
  - It must run causally: outcomes are used only after the 15-min target is observed.
- **What it does not fix:** ramp-conditional coverage. At 90%, ramp coverage stays about 0.61–0.66, and rolling conformal lowers it by about 0.05.
  - Adaptive marginal calibration re-balances width over time, not across forecast situations.
  - **Ramp-conditional coverage therefore remains a modelling target for Phase 6.**
- **Post-hoc observation, not a verdict change:** E4-B + rolling conformal numerically satisfies the Gate 2 coverage thresholds (50/80/90 = 0.504/0.793/0.890; every fold ≥ 0.878 at 90%).
  - Gate 2 was pre-registered on CQR-kW, so its NOT MET verdict stands.
  - Phase 6 should pre-register **E4-B + rolling conformal** as its calibrated reference.
- **No guarantee claimed:** no finite-sample or conditional coverage guarantee is claimed for either adaptive method.
