# Protocol R-E3: forward-in-time evaluation and calibration (SKIPP'D)

Committed before any E3 training. Label: **forward** (new evaluation; not pre-registered at the original freeze).

## Chronological split (development days of `folds_blocked5_v1.json`; official test days excluded from fitting)
| Period | Dates | Use |
|---|---|---|
| Fit | 2017-03-09 … 2019-06-30 | expert training; clear-sky anchor; event thresholds; feature standardisation |
| Purge | 2019-07-01 | unused |
| Selection / calibration | 2019-07-02 … 2019-07-31 | early stopping; gate fitting (incl. λ); conformal initial scores |
| Purge | 2019-08-01 | unused |
| Evaluation | 2019-08-02 … 2019-10-26 | scored once per method after all fitting is done |

The 3 official test days in the evaluation window (2019-08-10, 2019-10-14, 2019-10-19) are reported as a separate subset (1-min cadence). The camera change (2019-05-22) lies in the fit period (≈ 5 weeks of post-change data).

**Disclosure:** architectures and hyperparameters were chosen during development with blocked CV whose validation folds included the evaluation dates. E3 refits everything on earlier data; it is not a never-seen period in the design sense.

## Models (each 5 seeds: 42, 123, 456, 789, 1011; frozen recipes)
E4-B-fit (direct 15 min), E6-S (event mixture; τ from fit days), A2 (3D, 60 min), P15 (PV-only, from R-E1). Gates refitted on the selection period with the frozen procedure: G3 (E4-B-fit + E6-S, [g, d]), G3″ (A2 + E6-S, [g, d]), G4 (E4-B-fit + E6-S, [g, s_D, s_R, d]); routing G3″-F as frozen (timestamp rule). Ensemble: 5-member level-wise quantile average.

## Calibration methods (same raw forecasts; W frozen from development: 4,620 for gated/E6-S, 990 for E4-B-fit)
None (raw quantiles); daily-reset rolling conformal initialised from the latest matured selection-period scores; continuous rolling conformal (no reset); ACI (Gibbs & Candès, γ = 0.005, as Phase C M3). Levels 50/80/90.

## Metrics
Raw CRPS (overall, sunny, cloudy, ramp, non-ramp, rising/falling ramps), RMSE/MAE; coverage, mean width, Winkler (interval) score by the same strata; day counts and dates; provenance of initial scores. Paired day bootstrap (10,000, stratified, seed 20261022): G3″-F − E4-B-fit, − E6-S, − G4, − P15, − smart persistence.

## Pre-specified questions (descriptive verdicts, CI rule)
F1: G3″-F − E4-B-fit overall CRPS < 0. F2: G3″-F − persistence < 0. F3: 90% coverage within 0.90 ± 0.03 for each calibration method.
