# C07 / C19 report: conformal initialisation and window composition

Protocol: `revision/protocols/R-C07_conformal_pastonly.md` (d537511). Code: `src/revision_c07.py`, `src/revision_posthoc_test.py c19`. Raw outputs: `revision/results/c07_conformal_pastonly.json`, `revision/results/posthoc_test_c19.json`. Label: post hoc (test).

## C07: was the frozen initialisation past-only? No.
`src/final_eval.py::nearest_init` takes the W development out-of-fold scores nearest in time to each test day's first forecast, **in either direction**. The official test days lie inside the development period, so later development days were used.

| Model (W) | Mean share of initial scores dated after the test day | Max |
|---|---|---|
| G3″-F (4,620) | 46.8% | 66.4% |
| E4-B-fit (990) | 41.3% | – |
| E6-S, G4 (4,620) | 46.8% | – |

This affects **calibrated intervals only** (H5 and the calibrated rows of the interval table). Raw CRPS, H1–H4 and all point metrics are unaffected.

## Past-only rerun (identical rule, initial scores = latest W development scores matured before the day's first forecast)
The frozen H5 numbers were first reproduced exactly (all levels and strata, |Δ| < 1e−9). The most recent past-only scores are 0.45–0.64 days old (the previous development day).

G3″-F, coverage (raw → frozen → past-only), width (kW) and interval score of the past-only result:

| Nominal | Stratum | Raw | Frozen | **Past-only** | Width | IS |
|---|---|---|---|---|---|---|
| 90% | overall | 0.930 | 0.935 | **0.926** | 6.23 | 7.60 |
| 90% | sunny | 0.956 | 0.968 | 0.950 | 0.76 | 0.83 |
| 90% | cloudy | 0.903 | 0.902 | 0.902 | 11.70 | 14.37 |
| 90% | ramp | 0.740 | 0.746 | 0.746 | 12.98 | 21.07 |
| 90% | rising ramp | 0.688 | 0.698 | 0.696 (misses all above) | | |
| 90% | falling ramp | 0.789 | 0.792 | 0.795 (misses all below) | | |
| 90% | sunrise ramp | 0.642 | 0.653 | 0.687 | | |
| 80% | overall | 0.824 | 0.846 | **0.823** | 4.40 | 6.29 |
| 50% | overall | 0.491 | 0.542 | **0.526** | 2.03 | 4.53 |

- **H5 under the strictly causal rule:** 0.926 lies within 0.90 ± 0.03. Under the frozen (non-causal) rule it was 0.935 (not met). The frozen verdict stays as recorded; the manuscript reports both and identifies the past-only value as the strictly causal result.
- Other models at 90% (frozen → past-only): E4-B-fit 0.927 → 0.933; E6-S 0.927 → 0.917; G4 0.934 → 0.926.
- Ramp-conditioned under-coverage is unchanged by the initialisation (0.75 at 90%; 0.16 at 50%), and it is directional (rising ramps miss above, falling ramps below).

## C19: window composition and ensemble-size mismatch
- With W = 4,620 and about 660 forecasts per test day, **83–89% of the window is still development scores at the end of each day** (median 86.5%). The initial scores dominate every test day, as the reviewer noted.
- **Ensemble-size sensitivity** (3-member ensembles = the 3 seeds of one fold, same past-only initial scores):

| Ensemble | CRPS | Raw 90% coverage | Calibrated 90% |
|---|---|---|---|
| 15 members | 1.006 | 0.930 | 0.926 |
| 3 members, fold 0 … 4 | 1.004–1.026 | 0.866–0.942 | 0.885–0.939 |

The 15-member ensemble is within the range of the 3-member ensembles. Differences between folds (0.87–0.94) are larger than the 3- vs 15-member difference. **The ensemble-size mismatch is not supported as the cause of the over-coverage**; the manuscript must not attribute it to that.

## Manuscript implications
- §3.5 / Algorithm 1: state the exact initialisation rule; replace "nearest in time" by the past-only rule for the reported causal result; disclose the frozen rule and the 47% future-dated share.
- §4.4.4: report raw, frozen-calibrated and past-only-calibrated coverage; conditional coverage by rising/falling ramps; state that outcome-defined ramp strata are not covered by a marginal guarantee.
- Remove the ensemble-mismatch explanation of over-coverage (or state it was tested and not supported).
