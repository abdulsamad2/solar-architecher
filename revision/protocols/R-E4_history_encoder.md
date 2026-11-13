# Protocol R-E4: history length × encoder factorial

Committed before training the missing arm. Labels: **development** (primary), **post hoc** (test).

## Design (16 frames each; common 60-min population; same recipe, seeds 42/123/456 × 5 folds)
| | 15-min history (1-min stride) | 60-min history (4-min stride) |
|---|---|---|
| 2D CNN + GRU | E8 D0 (exists) | E8 D2 (exists) |
| 3D CNN + GRU | **E10 A2-H15 (new)** | E9 A2 (exists) |

New arm: `--e8-window H15 --encoder cnn3d_gru --e8-features none --head quantile --calib-holdout`, same budget as A2 (≤ 15 epochs, patience 5). Recorded code differences between the existing arms' training commits are listed in the report.

## Metrics
Overall/sunny/cloudy/ramp/non-ramp CRPS, RMSE and MAE of the median, ramp recall (R-C18 definition), parameter count, batch-1 latency and throughput (same GPU, same timing code).
Effects (per seed, paired day-block bootstrap): history effect within each encoder (H60 − H15), encoder effect within each history (3D − 2D), interaction ((3D60 − 3D15) − (2D60 − 2D15)). Seed mean ± SD.
Post hoc test: 15-member ensembles on 60-min-eligible test samples.

## Interpretation rule
The 60-min 3D design is credited with the gain only for effects whose CI excludes 0 in ≥ 2/3 seeds. Otherwise the manuscript attributes improvement to the combination, not to the history length or encoder.
