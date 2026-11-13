# Protocol R-X1: image shuffling diagnostic (optional E1 diagnostic)

Committed before any run. Label: **post hoc** (official test inputs; frozen models; nothing refitted).
Origin: review E1, "An images-only model and temporal image shuffling are useful optional diagnostics"; pre-listed as the optional diagnostic in `R-E1_pv_only.md`.

## Question
Do the frozen image experts use the content of the current sky images and their temporal order, or mainly the PV history?

## Frozen inputs
15-member (3 seeds x 5 folds) E4-B-fit (2D, 15 min), A2 (3D, 60 min, 60-min-eligible test samples) and E6-S, run exactly as in `src/revision_posthoc_dump.run_member` (frozen fold anchors, K history, frozen gates). The deployed G3''-F path is recomputed from the perturbed experts with the frozen G3 / G3'' gate parameters (`final_eval.blend`, `gate_params`).
Reproduction check: the unperturbed run must reproduce the frozen test predictions of E4-B-fit and G3''-F to 1e-6 kW, otherwise STOP.

## Perturbations (images only; PV/K history, anchors, targets and gate inputs other than the expert outputs unchanged)
| ID | Perturbation |
|---|---|
| S-perm | whole image stack of each sample replaced by the stack of a randomly drawn other test sample (permutation of samples; 60-min experts permuted among 60-min-eligible samples); 5 permutation seeds (0-4), results averaged per sample |
| T-rev | frame order reversed within each sample (newest frame first) |
| T-rand | frame order randomly permuted within each sample (seed 0) |
| T-static | all 16 frames replaced by the latest frame (no motion, current sky kept) |

## Metrics
Raw CRPS (9 quantiles, `crps_quantile_interp`) overall, sunny, cloudy, ramp, non-ramp (frozen test ramp definition), MAE/RMSE of the median; change = perturbed - original per sample; paired test-day bootstrap (`revision_common.day_bootstrap`, 10,000 resamples, stratified sunny/cloudy). For S-perm also the range over the 5 permutation seeds.

## Interpretation rule
Image content is used if S-perm worsens overall CRPS with CI > 0; temporal order is used if T-rev or T-rand worsens ramp CRPS with CI > 0. Results are reported whichever way they fall. Diagnostic only: perturbed inputs are out of distribution, so a large deterioration shows reliance, not the size of the information gain (that is E1 / R-X3).
