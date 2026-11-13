# Protocol R-C22 / R-C21: deployment latency and external comparison

Committed before any run.

## C22 latency (measurement)
- Batch-1, end to end, on the RTX 5000 Ada: from 16 raw image records + PV history to calibrated interval. Steps timed separately: preprocessing (clear-sky anchor evaluation, K, window assembly, image normalisation, feature table where used), 15 members × each expert, gate, ensemble, conformal update.
- Both branches: G3″ (A2 + E6-S) and fallback G3 (E4-B-fit + E6-S). 50 warm-up, then 500 test samples; median and p95 (CUDA-synchronised wall clock). Also CPU-only timing for the gate + calibration.
- Parameters: per member and resident totals (all 15 members of every expert collection that must be loaded).

## C21 external comparison (contextual, unpaired)
- Our 15-member quantile forecasts → m samples by inverse-transform sampling of the piecewise-linear quantile function (constant tails as in the scorer). Sample-based CRPS with the standard (biased) and the fair estimator (Ferro 2014), m ∈ {10, 30, 100}, 100 random repeats (mean ± SD), on the published sample set where it is defined (Phase A) and on all test samples.
- Raw sample intervals vs calibrated intervals reported separately. No state-of-the-art claim.

## Outputs
`revision/results/c22_latency.json`, `revision/results/c21_external.json`; reports in `revision/reports/`.
