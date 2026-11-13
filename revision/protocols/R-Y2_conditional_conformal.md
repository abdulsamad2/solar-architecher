# Protocol R-Y2: conditional (Mondrian) conformal calibration on forecast-time state

Committed before any run. Labels: **development** (selection), **post hoc** (test).
Origin: the calibrated 90% intervals of the proposed forecaster (G3''-F) cover about 0.75 of ramp outcomes on the official test (C07 past-only rule) with directional misses; marginal conformal calibration gives no guarantee for strata.

## Question
Can calibration that conditions on information available at issue time (the expert disagreement or the forecaster's own predicted spread) raise interval coverage on ramps, without losing overall coverage or sharpness?

## Inputs
- Development out-of-fold raw quantiles of G3''-F, reconstructed from the frozen per seed x fold `preds_val.npz` of E4-B-fit, E6-S and A2 with the frozen G3 / G3'' gate parameters and the timestamp routing, averaged over the three out-of-fold members. **STOP check:** the reconstructed conformity scores must reproduce `experiments/final/dev_oof_calibration_v1.npz` (`G3''-F|50/80/90`) to 1e-6 kW for >= 99.9% of samples, max 5e-3 kW.
- Official test: frozen 15-member G3''-F quantiles (`experiments/final/test_predictions_v1.npz`) and the per-member expert dumps (`experiments/R_posthoc`) for the features.

## Features (computed at issue time from the forecast only)
- **F-d (primary):** disagreement d = |mean of ramp-capable expert - mean of E6-S|, averaged over members (the gate feature).
- **F-w (secondary):** raw 90% width (q95 - q05) of the G3''-F ensemble.

## Methods (all strictly past-only, daily reset as in the C07 past-only rerun)
- **U (reference):** unconditional rolling CQR, the C07 past-only rule (W = 4620), reproduced to 1e-9 (else STOP).
- **M-d / M-w:** Mondrian rolling CQR. Feature bins from development terciles (K = 3, edges frozen from development). Each bin keeps its own window of the W_b most recent matured scores of that bin. Each test day starts from the W_b latest matured development scores of the bin dated strictly before the day's first issue time.
- W_b is chosen on development from {990, 1540, 2310, 4620} by the mean 90% interval score of a chronological rehearsal (other days only, as in `final_dryrun`). The selected W_b is frozen before the test is touched.

## Metrics
Coverage, below / above miss rates, mean width and interval score at 50/80/90%, overall and for sunny, cloudy, ramp, non-ramp, rising and falling ramp, and per feature bin. Paired day-block bootstrap of the 90% interval-score difference M - U, overall and on ramps.

## Decision rule (fixed now)
A Mondrian method **improves conditional reliability** if, on the test, its 90% ramp coverage exceeds that of U by >= 0.05 while its overall 90% coverage stays within 0.87–0.93 and its overall 90% interval score is not worse than U by more than 2%. Otherwise it **does not**. Reported whichever way it falls; the frozen method and verdicts are not changed.
