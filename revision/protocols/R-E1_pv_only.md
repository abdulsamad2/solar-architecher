# Protocol R-E1: PV-only probabilistic baseline

Committed before any training. Labels: **development** (primary), **post hoc** (test).

## Question
Do sky images add information beyond PV history, clear-sky normalisation, probabilistic training and ensembling?

## New arms (identical recipe to their image counterparts except the image branch)
| Arm | Encoder | History | Population | Image counterpart |
|---|---|---|---|---|
| P15 | `pv_gru`: per-step Linear(1 + extra → 64) + ReLU → GRU(64 → 64) | manifest 15-min (16 × 1 min) | full | E4-B-fit |
| P60 | same | E8 H60 window (16 × 4 min) | common 60-min | A2 |

Same: K input and target, fit-day clear-sky anchor, 9-level non-crossing quantile head initialised at climatology, AdamW lr 1e−4, wd 1e−4, batch 128, cosine, EMA 0.999, clip 1.0, ≤ 15 epochs, patience 5 on validation CRPS, calibration hold-out, blocked folds `folds_blocked5_v1.json`, seeds 42/123/456 (15 members). Parameter counts reported (no capacity matching beyond the shared GRU/head; disclosed).

## Comparisons
Development (per seed, paired day-block bootstrap, stratified): P15 − E4-B-fit; P60 − A2; P15/P60 − G3″ (common population) and − G3 (full). Strata: overall, sunny, cloudy, ramp, non-ramp. Also MAE and RMSE of the median, ramp recall (median-change detection, definitions in R-C18).
Post hoc test: 15-member ensembles; paired test-day bootstrap vs E4-B-fit, A2 (60-min eligible), G3″-F.

**Optional diagnostic (post hoc test):** frozen E4-B-fit and A2 with images shuffled across samples within the test set (PV history unchanged), CRPS change.

## Interpretation rule
If P15 (P60) is within 0.02 kW of E4-B-fit (A2) overall and on ramps (CI covering 0), the image-information claim is narrowed accordingly. The result is reported whichever way it falls.
