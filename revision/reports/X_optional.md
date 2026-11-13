# Optional experiments X1–X4 (report)

Protocols: `revision/protocols/R-X1_image_shuffling.md`, `R-X2_linear_pool.md`, `R-X3_images_only.md`, `R-X4_forward_gate_split.md`, committed (local d780838; public dd20ae5) before any run. Code: `src/revision_x1_shuffle.py`, `src/revision_x2_linear_pool.py`, `src/revision_x3_images_only.py`, `src/revision_x4_forward_gates.py`; training queue `scripts/revision_x_queue.ps1` (`--input none` in `src/train.py`; split `revision/manifests/folds_forward_v2.json`). CRPS in kW; differences are (perturbed or alternative) − reference unless stated; [95% paired day-bootstrap CI].

## X1 Image shuffling diagnostic (post hoc, official test; frozen models)
Unperturbed run reproduces the frozen E4-B-fit and G3''-F test forecasts exactly (max |diff| 0.0 kW). Output `revision/results/x1_image_shuffling.json`.

| Model (n) | Original | Images permuted across samples (mean of 5) | Frames reversed | Frames randomly permuted | All frames = latest frame |
|---|---|---|---|---|---|
| E4-B-fit, 2D 15 min (13,266) | 1.117 | +1.066 [+0.930, +1.223] | +0.052 [+0.027, +0.085] | +0.024 [+0.011, +0.041] | −0.010 [−0.022, −0.000] |
| A2, 3D 60 min (12,355) | 1.135 | +1.292 [+1.153, +1.444] | +0.169 [+0.109, +0.234] | +0.091 [+0.052, +0.134] | +0.123 [+0.078, +0.174] |
| G3''-F, proposed (13,266) | 1.006 | +0.749 [+0.660, +0.838] | +0.082 [+0.045, +0.122] | +0.048 [+0.024, +0.075] | +0.041 [+0.017, +0.068] |

Ramp samples:

| Model | Permuted | Reversed | Random order | Latest frame only |
|---|---|---|---|---|
| E4-B-fit | +0.565 [+0.374, +0.750] | +0.181 [+0.043, +0.346] | +0.110 [+0.033, +0.202] | −0.095 [−0.162, −0.032] |
| A2 | +0.798 [+0.655, +0.927] | +0.069 [−0.140, +0.305] | +0.005 [−0.095, +0.114] | +0.349 [+0.213, +0.472] |
| G3''-F | +1.408 [+1.074, +1.700] | +0.022 [−0.099, +0.153] | −0.023 [−0.088, +0.049] | +0.132 [+0.059, +0.199] |

Range of the 5 permutations (overall CRPS): E4-B-fit 2.166–2.198, A2 2.407–2.439, G3''-F 1.749–1.771.

Interpretation (protocol rule): image content is used by every expert (permutation roughly doubles CRPS, CI > 0 in every stratum). Temporal order is used: reversing or permuting frames worsens overall CRPS of all three models. The two encoders differ: the 3D 60-min expert relies on the frame sequence (replacing the sequence by the latest frame worsens ramp CRPS by 0.349 kW), whereas the 2D 15-min expert does not gain from motion (latest frame only is slightly better, also on ramps, −0.095 kW), although it is sensitive to a reversed sequence (+0.181 kW on ramps). This supports the E4 attribution of the ramp advantage partly to the 3D encoder over a longer window. Perturbed inputs are out of distribution: the deteriorations show reliance, not the size of the information gain.

## X2 Gated linear pool vs gated quantile blend (development primary; post hoc test)
Output `revision/results/x2_linear_pool.json`. G3'' reproduced exactly in every seed (development, 1e-9 vs `reports/e9_gate.json`) and on test (max |diff| < 1e-6 vs the frozen G3''-F forecasts on the 60-min-eligible samples). Checks (scratch, not saved): mixture quantiles at w = 1 and w = 0 equal the single expert's quantiles (max |diff| ≤ 9e-16) with identical CRPS; exact mixture CRPS = atom CRPS = midpoint-pinball CRPS at w = 1; mixture quantiles never cross; calibration and validation days do not overlap. Gate refit and constant weight use calibration-day predictions only; test outcomes are used only for scoring.

Development (out-of-fold, common 60-min population; seed mean):

| Method | Overall | Sunny | Cloudy | Ramp | Non-ramp |
|---|---|---|---|---|---|
| G3'' (quantile blend, proposed) | 0.5865 | 0.1214 | 0.9227 | 2.917 | 0.3763 |
| LP-frozen (mixture, same gate weights) | **0.5798** | **0.1180** | 0.9135 | **2.862** | 0.3739 |
| LP-refit (mixture, gate refitted under mixture CRPS) | **0.5796** | 0.1205 | **0.9113** | 2.888 | **0.3714** |
| LP-const (mixture, constant weight) | 0.6004 | 0.1442 | 0.9301 | 3.095 | 0.3754 |

G3'' − LP-refit, overall: +0.0063 [+0.0052, +0.0075], +0.0084 [+0.0063, +0.0111], +0.0062 [+0.0050, +0.0074] (seeds 42, 123, 456); ramp +0.042, +0.016, +0.030 (all CI > 0). G3'' − LP-frozen overall +0.0065, +0.0072, +0.0066 (all CI > 0). Exact atom-grid CRPS gives the same ordering (G3'' 0.5834 vs LP-frozen 0.5757, seed 42). Protocol decision: the linear pool is **better** than the quantile blend (CI > 0 overall in 3/3 seeds), by about 1.1%.

Post hoc test (15 members, 12,355 60-min-eligible forecasts):

| Method | Overall | Ramp | Cloudy | Raw 90% coverage / width (kW) |
|---|---|---|---|---|
| G3'' | 1.0537 | 3.950 | 2.036 | 0.933 / 6.45 |
| LP-frozen | **1.0466** | **3.888** | 2.023 | 0.926 / 6.16 |
| LP-refit | 1.0471 | 3.933 | **2.021** | 0.932 / 6.22 |
| LP-const | 1.0738 | 4.237 | 2.058 | 0.939 / 6.64 |

G3'' − LP-frozen: overall +0.0071 [+0.0035, +0.0104], ramp +0.062 [+0.045, +0.083], non-ramp −0.0014 [−0.0042, +0.0013]. G3'' − LP-refit: overall +0.0066 [+0.0027, +0.0109], ramp +0.017 [−0.000, +0.041]. G3'' − LP-const: −0.020 [−0.032, −0.007].

Interpretation: with the same experts and the same gate inputs, a probability mixture scores slightly better than the quantile blend, on development and on test, mainly on ramps and cloudy samples, with narrower raw intervals. The gating itself remains essential (a constant-weight mixture is worse than both gated combinations). The proposed combination is not changed post hoc; the paper should report that a gated linear pool is a small, consistent improvement and that the quantile-blend choice is not optimal in CRPS.

## X3 Images-only model (development primary; post hoc test)
I15: 2D CNN-GRU, 15-min window, constant-zero history feature (no PV or K value of the forecast day enters the network), otherwise the E4-B-fit recipe; 15 members (3 seeds × 5 folds; 90,761 parameters, identical to E4-B-fit). Training 2026-11-13 21:09 – 2026-11-14 00:13 (91, 46, 47 min per seed; seed 42 shared the GPU with X1 and R-X4). Most folds stopped at epoch 3. Output `revision/results/x3_images_only.json`; test members `experiments/R_posthoc/test_members_I15.npz` (dump reproduction of E4-B-fit, E6-S, G4, G3''-F: 0.0 kW).
**I60 not run** (cost rule of the protocol: I15 alone took about 3 GPU hours under contention; I15 + I60 was estimated above 6 GPU hours).

Development (out-of-fold, full population; seed mean):

| Arm | Overall | Sunny | Cloudy | Ramp | Non-ramp | Ramp recall / precision (seed 42) |
|---|---|---|---|---|---|---|
| I15 (images only) | 0.924 | 0.411 | 1.290 | **2.214** | 0.780 | 0.514 / 0.421 |
| E4-B-fit (images + K history) | 0.751 | 0.305 | 1.068 | 2.448 | 0.560 | 0.335 / 0.561 |
| P15 (K history only) | **0.691** | **0.213** | **1.031** | 3.179 | **0.412** | 0.033 / 0.724 |

I15 − E4-B-fit: overall +0.179 [+0.157, +0.205], +0.155 [+0.136, +0.175], +0.188 [+0.167, +0.211]; ramp −0.229 [−0.266, −0.191], −0.245 [−0.279, −0.208], −0.231 [−0.272, −0.188] (seeds 42, 123, 456).
I15 − P15: overall +0.242, +0.209, +0.250 (all CI > 0); ramp −0.953, −0.994, −0.949 (all CI < 0).

Post hoc test (15 members, 13,266 forecasts):

| Arm | Overall | Sunny | Cloudy | Ramp | Non-ramp | RMSE / MAE of median |
|---|---|---|---|---|---|---|
| I15 | 1.191 | 0.312 | 2.072 | **3.201** | 0.852 | 3.094 / 1.646 |
| E4-B-fit | **1.117** | 0.218 | **2.017** | 3.627 | 0.693 | **3.037 / 1.534** |
| P15 | 1.239 | **0.193** | 2.286 | 4.851 | **0.629** | 3.345 / 1.586 |

I15 − E4-B-fit: overall +0.075 [+0.041, +0.110], ramp −0.426 [−0.546, −0.304], non-ramp +0.159 [+0.122, +0.200], cloudy +0.055 [−0.001, +0.113].
I15 − P15: overall −0.047 [−0.125, +0.031], ramp −1.650 [−2.152, −1.180], cloudy −0.214 [−0.348, −0.080], sunny +0.119 [+0.044, +0.209].

Interpretation: the two information sources are complementary. PV history alone gives the best calm-condition (sunny, non-ramp) forecasts; images alone give the best ramp forecasts (development and test), with the highest ramp recall and the largest predicted changes (change slope on ramps 0.44 vs 0.32 for E4-B-fit and 0.11 for P15, seed 42), at the cost of more false alarms and worse non-ramp CRPS. Adding the K history to the images (E4-B-fit) improves overall CRPS but makes the forecast more persistence-like on ramps. This is consistent with E1 (image benefit located on ramps) and with the design of the proposed forecaster, which combines a ramp-sensitive image expert with a calm-condition expert. Caveat: the ramp stratum is defined by the realised change, so a model that forecasts larger changes is favoured there and penalised on non-ramp samples; overall CRPS is the decisive score.

## X4 Forward evaluation with separate early-stopping and gate days (forward sensitivity)
E4-B-fit, E6-S, A2 and P15 retrained (5 seeds; 20 jobs, 2026-11-13 21:09 – 2026-11-14 01:29, all rc = 0) on the fit days with early stopping on selection days 1–15 (2019-07-02..07-17); gates G3, G3'', G4 (lambda included) and conformal initial scores fitted on selection days 16–29 (2019-07-18..07-31; 5,392 scores, all matured before the first evaluation origin); evaluation unchanged (83 days, 27,937 forecasts). Analysis = unchanged `src/revision_e3.py` with the gate-day predictions. Output `revision/results/x4_forward_gate_split.json`. Note: P15 selected the same epochs as in E3 and training is deterministic, so its weights are bit-identical to E3 (its scores coincide); the other three models changed (best epochs differ in 9 of 15 runs).

| Model | Overall | Sunny | Cloudy | Ramp | RMSE (median) |
|---|---|---|---|---|---|
| G3''-F | 0.380 (E3 0.374) | 0.138 | 0.718 | **1.761** | **1.347** |
| E4-B-fit | 0.491 | 0.262 | 0.810 | 1.806 | 1.403 |
| E6-S | 0.377 | **0.108** | 0.752 | 2.182 | 1.473 |
| G4 | **0.370** | 0.122 | **0.717** | 1.844 | 1.354 |
| P15 | 0.454 | 0.183 | 0.831 | 2.398 | 1.510 |
| Smart persistence | 0.665 | 0.285 | 1.195 | 3.779 | 1.683 |

G3''-F − comparator [95% CI] (E3 values in brackets for comparison):

| Comparator | Overall | Ramp |
|---|---|---|
| E4-B-fit | −0.111 [−0.125, −0.098] (E3 −0.113) | −0.045 [−0.107, +0.009] |
| E6-S | +0.003 [−0.014, +0.020] (E3 −0.003 [−0.016, +0.010]) | −0.421 [−0.532, −0.312] |
| G4 | +0.010 [+0.001, +0.018] (E3 +0.007 [+0.000, +0.014]) | −0.083 [−0.138, −0.030] |
| P15 | −0.074 [−0.095, −0.055] | −0.637 [−0.768, −0.504] |
| Persistence | −0.285 [−0.329, −0.244] | −2.018 [−2.330, −1.696] |

90% intervals (G3''-F): coverage overall / ramp / width / Winkler: none 0.921 / 0.568 / 2.28 / 3.15; daily-reset 0.884 / 0.506 / 2.16 / 3.15; continuous rolling 0.901 / 0.521 / 2.23 / 3.14; ACI 0.901 / 0.547 / 2.37 / 3.25.

Interpretation (protocol question): the ranking does not change. With gates fitted on days not used for early stopping, the proposed forecaster still beats the direct expert and persistence, still ties the calm-condition expert overall, and is still slightly worse than G4 overall, now with a CI that excludes zero (+0.010 [+0.001, +0.018] kW), while remaining better than both on ramps. The E3 finding against G4 is therefore not an artefact of reusing the selection days; it is a small but real disadvantage in this calm forward period. Calibration conclusions are unchanged (continuous rolling and ACI at nominal 90%; ramp coverage 0.51–0.57).

## Summary of deviations
- X3: I60 not trained (protocol cost rule). Seed 42 of I15 shared the GPU with X1 and R-X4 (slower, results unaffected).
- X1: the G3''-F path was included (protocol: "if straightforward").
- X4: none. X2: none.
