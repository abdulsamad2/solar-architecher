# Gatton E5 lag analysis: frozen procedure (written before the sealed test; audit fix C)

**Pre-registered as:** E5 (descriptive) in `reports/phaseE_gatton_prereg.md` §5: "lag analysis (lag-0 and lag-15 correlation of forecast increments), as in `reports/lag_analysis.json`".

**Input.** Only the immutable raw outputs of the one-shot evaluation:
- `experiments/E_gatton/test_predictions_v1.npz` (sample IDs, observed PV `y`, smart persistence, 9 quantiles per arm and model);
- the per-day sunny labels in `reports/phaseE_gatton_test_results.json`.

No model is run and nothing is fitted.

**Code:** `src/gatton_lag.py`. It reuses `src/lag_analysis.best_lag` unchanged, the same function used for SKIPP'D.

**Command (after the official run):**
`python -m src.gatton_lag experiments/E_gatton/test_predictions_v1.npz reports/phaseE_gatton_test_results.json reports/phaseE_gatton_test_lag.json`

**Fixed rules.** None of these may be changed after the sealed outcomes are seen.

| Rule | Value |
|---|---|
| Alignment | Each forecast at its **target** time (t + 15 min) on the record grid: **1-min step** for test samples; 2-min for development samples (rehearsal only) |
| Signal | First differences of the median forecast (quantile index 4) and of observed PV, per day; gaps stay missing |
| Lags | L = −5 … +20 min, in whole grid steps; correlations need > 30 paired increments |
| Per-day statistics | Best lag (argmax of the correlation), correlation at L = 0, correlation at L = 15 |
| Summary per model | Median best lag over days; mean lag-0 and mean lag-15 correlation; for **all days** and for **cloudy days** (test-day labels from the frozen E-2 rule) |
| Models | F1 G3″-F, F1 E4-B-fit, F1 E6-S, F0 G3″-F, Z0 G3″-F, smart persistence |
| Interpretation | Descriptive only; no hypothesis test |

**Rehearsal.** The identical command was run on the development rehearsal outputs (`reports/phaseE_gatton_rehearsal_lag.json`).
