# Phase C report: calibration under temporal shift and in ramp regimes (development data only)

- **Protocol:** `reports/phaseC_calibration_prereg.md` (frozen 1d1c1a4, before execution).
- **Code:** `src/phaseC_streams.py`, `src/phaseC_calib.py` (4b25a76).
- **Results:** `reports/phaseC_results.json`.
- **Data:** out-of-fold development streams, 164,362 samples, 497 days, 5 blocked folds. Each sample is predicted only by the 3 members that never trained on it.
- **No test or external data were used.** All hyperparameters were chosen leave-one-fold-out. Initial scores always come from other folds.

## Pre-registered hypothesis C1
C1 asks whether regime-aware (Mondrian) calibration raises ramp-conditional 90% coverage by ≥ 0.05 over M1, with a CI lower bound > 0, without worsening the overall 90% interval score (IS) by more than 2%.

| Stream | Picked (by the LOFO criterion) | Ramp cov90 − M1 [95% CI] | Overall IS90 vs M1 | Verdict |
|---|---|---|---|---|
| **G3″-F (primary)** | M6 (bins by event probability) | **+0.001 [−0.013, +0.015]** | −5.6% | **NOT confirmed** |
| E4-B-fit (replication) | M5 (bins by forecast spread) | +0.074 [+0.057, +0.091] | −9.2% | Confirmed |

**The primary hypothesis fails.** For G3″-F, regime-binned calibration does not improve ramp coverage. For E4-B-fit, a single expert without a gate, it does. A plausible reading is that G3″-F's gate already conditions the forecast spread on the same forecast-time signals (disagreement, event probability). The calibration step then has little regime information left to add.

## Main descriptive finding: the daily reset is fragile when its initial scores are far in time

90% coverage, overall and per fold:

| Method | G3″-F overall | G3″-F per fold 0–4 | E4-B-fit overall | E4-B-fit per fold 0–4 |
|---|---|---|---|---|
| M0 uncalibrated | 0.903 | – | 0.884 | – |
| **M1 rolling, daily reset** (the Gate 5 method) | **0.867** | 0.98 / **0.61** / 0.90 / 0.85 / 0.93 | **0.859** | 0.95 / 0.78 / 0.88 / 0.77 / 0.88 |
| M2 rolling, continuous | 0.893 | 0.90 / 0.88 / 0.90 / 0.90 / 0.89 | 0.892 | 0.90 / 0.90 / 0.88 / 0.89 / 0.89 |
| M3 ACI | 0.904 | 0.92 / 0.90 / 0.90 / 0.90 / 0.90 | 0.893 | 0.92 / 0.82 / 0.90 / 0.90 / 0.90 |
| M4 DtACI-style | 0.865 | 0.89 / 0.83 / 0.86 / 0.86 / 0.88 | 0.844 | – |
| M5 Mondrian by spread | 0.892 | 0.90 / 0.88 / 0.89 / 0.90 / 0.89 | 0.881 | 0.88 / 0.88 / 0.88 / 0.87 / 0.88 |
| M6 Mondrian by event probability | 0.892 | 0.89 / 0.88 / 0.90 / 0.90 / 0.90 | 0.883 | 0.88 / 0.89 / 0.88 / 0.88 / 0.89 |

**Interval score at 90%** (G3″-F / E4-B-fit):

| Method | G3″-F | E4-B-fit |
|---|---|---|
| M0 | 4.98 | 7.37 |
| M1 | **5.23** | **7.67** |
| M2 | 4.96 | 7.15 |
| M3 | 4.95 | 7.35 |
| M4 | 4.93 | 6.90 |
| M5 | 4.95 | 6.96 |
| M6 | 4.94 | 7.04 |

M1 has the worst score of all methods, worse even than no calibration. Every other method beats M1, with all CIs below 0 (`bootstrap_vs_M1`).

**Interpretation.**
- In this study M1 had to start each day from the nearest scores in *other folds*. Folds are contiguous blocks, so those scores can be months away, and the daily reset stops the window from adapting during the day. Fold 1 (Sep 2017–May 2018) falls to 0.61 coverage.
- In the Gate 5 dry run, M1 initialised from *neighbouring days* and reached 0.90. On the SKIPP'D test the test days sit inside the development period, with near-in-time initial scores, and coverage was 0.935.
- On the SkyGPT external days the nearest initial scores were weeks old (late October), and coverage fell to 0.77.
- **The three settings tell one consistent story.** The frozen calibration works only when recent, similar scores are available. Continuous rolling (M2) and ACI (M3) stay at 0.88–0.92 in every fold without that assumption.

**Other observations:**
- **G3″-F without any calibration (M0) is already well calibrated on development data:** 0.903 at 90%. The gated, ensembled quantiles carry most of the calibration.
- **Ramp-conditional coverage remains about 0.64–0.71 at nominal 90%** for every method on G3″-F. No calibration method in this family fixes ramp under-coverage. This is consistent with the lag analysis (`reports/lag_analysis.json`): the forecasts do not anticipate changes.

## Consequences
1. The Gate 5 and Phase A results stand as reported. They used the frozen M1 and are not re-evaluated.
2. **For any future sealed evaluation (Phase E), use continuous causal rolling conformal (M2) or ACI (M3) instead of the daily reset.** That choice must be pre-registered before the new test set is opened. This study only motivates it; it does not validate it on test data.
3. Paper framing:
   - online calibration restores marginal coverage only if it may adapt continuously;
   - daily-reset windows initialised from distant periods fail;
   - regime-aware calibration helps a single expert but adds nothing on top of a disagreement-gated mixture;
   - ramp coverage remains the open problem.
