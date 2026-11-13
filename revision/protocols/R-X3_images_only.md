# Protocol R-X3: images-only probabilistic model (optional E1 diagnostic)

Committed before any training. Labels: **development** (primary), **post hoc** (test).
Origin: review E1, "An images-only model ... [is a] useful optional diagnostic".

## Question
How much of the image experts' skill comes from the images alone, without any PV measurement at forecast time?

## New arms (identical recipe to their image counterparts except that no PV information enters the model)
| Arm | Encoder | History window | Population | Counterpart (images + K history) | PV-only counterpart |
|---|---|---|---|---|---|
| I15 | `cnn_gru` (2D) | manifest 15-min (16 x 1 min) | full | E4-B-fit | P15 |
| I60 | `cnn3d_gru` (3D) | E8 H60 (16 x 4 min) | common 60-min | A2 | P60 |

Implementation: new `--input none` option in `src/train.py`; the per-step history feature is a constant 0 (same tensor shape, so the architecture and parameter count are unchanged); the E8 per-frame features stay disabled (`--e8-features none`, zero-filled), as in A2. What remains at forecast time: the 16 sky images and the target-time clear-sky PV of the fold-fitted anchor (used only to reconstruct kW from the predicted clear-sky index, as in every arm). The fold anchor itself is fitted on training-day PV, as in every arm. No PV value or K value of the forecast day enters the network.
Same as the counterparts: K target, 9-level non-crossing quantile head, AdamW lr 1e-4, wd 1e-4, batch 128, cosine, EMA 0.999, clip 1.0, <= 15 epochs, patience 5 on validation CRPS, calibration hold-out, blocked folds `folds_blocked5_v1.json`, seeds 42/123/456 (15 members).

## Cost rule
One member is timed first. If I15 + I60 together are estimated above ~6 GPU hours, only I15 is trained and I60 is reported as not run.

## Comparisons
Development (per seed, out-of-fold validation predictions, paired day-block bootstrap stratified sunny/cloudy, as `src/revision_e1_e4.py`): I15 - E4-B-fit, I15 - P15 (full); I60 - A2, I60 - P60 (common). Strata overall, sunny, cloudy, ramp, non-ramp; MAE, RMSE of the median; ramp recall (R-C18 definitions).
Post hoc test: 15-member ensembles; paired test-day bootstrap I15 - E4-B-fit, I15 - P15 (full); I60 - A2, I60 - P60 (60-min eligible).

## Interpretation rule
Descriptive. The decomposition reported: images-only, PV-only and images + PV on the same samples. No claim in the paper is changed in either direction without stating the result.
