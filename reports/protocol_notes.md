# SKIPP'D protocol notes and reproduction log

## Raw data (data/raw, untouched; checksums in data/raw/SHA256SUMS.txt)
Source: https://purl.stanford.edu/dj417rh1007 (CC BY 4.0)

| File | Contents |
|---|---|
| 2017_2019_images_pv_processed.hdf5 | trainval/{images_log (349372,64,64,3) uint8, pv_log (349372,) kW}, test/{(14003,...)} |
| times_trainval.npy / times_test.npy | naive local datetimes, object dtype |

Paired-record checkpoints: dev 349,372 and test 14,003, both match the official numbers. There are 497 dev days and 20 test days.
PV range is 0 to 29.6 kW, with no NaN and no negative values. times_test is grouped by day and is not globally sorted.

## Official forecast construction (data_preprocess_forecast.ipynb)
- 16 frames at t - j min, j = 0..15, newest first; target PV at t + 15 min.
- Exact timestamp matches only.
- Greedy 2-min origin thinning: keep i if t_i - t_last_valid > 2 min - 1 s.
- Test = 20 fixed days (10 sunny / 10 cloudy), split after building on the full timeline.
- Built from the unreleased high-frequency (~10 s) image stream `all_times_highfreq.npy`. PV came from a separate full PV series.

SUNSET_forecast.ipynb: 10-fold CV over **randomly shuffled day blocks** (seed 1), Adam at lr 3e-6, batch 256, early stopping with patience 5. The test prediction is the mean of the 10 fold models. Its persistence reference is `Relative_output` CSI persistence.

## Why exact parity is impossible from the benchmark
The benchmark is thinned to ~1 record/min, and the second-of-minute phase drifts (:00 37%, the rest spread over :10 to :50).
The 1-min grid changes phase ~3,300 times (median unbroken run 92 min), and every change breaks exact t - j min matches.

| Matching tolerance | Dev samples | Test samples |
|---|---|---|
| exact (official rule) | 130,412 (-19.5%) | 5,569 |
| ±10 s | 143,782 | 5,918 |
| ±20 s | 157,541 | 6,383 |
| **±30 s (chosen)** | **164,362 (+1.5%)** | **6,638** (3,322 sunny / 3,316 cloudy) |
| official | 161,928 | not stated |

**Decision (2026-10-19):** the primary protocol uses ±30 s nearest-record matching for every history frame and the target (`src/skippd.py`, `tolerance_s=30`).
This is a documented deviation: target timing jitter is at most 30 s on a 15-min horizon. The exact-match manifest is kept as a sensitivity check.
Published SUNSET numbers must be compared with this caveat stated.

## Frozen manifests (data/manifests)
- `forecast_tol30_v1.npz` (primary). content_sha256 bb0f0932...2735b; rebuilding gives an identical hash.
- `forecast_exact_v1.npz` (strict-rule sensitivity check).

Each manifest stores sorted record times plus each record's HDF5 group and row. Each sample stores its origin, 16-frame history and target indices, and is_test / is_sunny flags.
The build asserts that no test sample draws on development records and no development sample draws on test records.

## Official helper (vendored)
`src/official/relative_op_func.py` is byte-identical to `Relative_op_func.py` (MIT). Its site and array constants:
- latitude 37.424107, longitude -122.174199
- tilt 22.5°, azimuth 195°, effective area 24.98 m², 1 kW/m² reference

Quirks kept for faithful reproduction:
- DST start/end days come from the *current* month's calendar instead of March/November, so boundary days are off.
- Relative output is floored at 0.05 kW on both P_theo and PV, and capped at 1.5.

## E0 persistence (manifest forecast_tol30_v1; experiments/E0_persistence/tol30_v1)
Official CSI persistence: pred = CSI(t, P_t) * P_theo(t+15 min), as in SUNSET_forecast.ipynb cell 24.
Persistence has no fitted parameters, so scoring it on the test set does not consume the locked test set; it is the skill reference.

| RMSE kW | Sunny (n=3,322) | Cloudy (n=3,316) | Overall (n=6,638) |
|---|---|---|---|
| Official CSI persistence, test | 0.414 | 5.124 | 3.634 |
| Naive persistence, test | 1.117 | 5.178 | 3.744 |
| Published SUNSET (reference only, not reproduced) | 0.61 | 4.27 | 3.03 |

Dev (overall only, no sky labels): CSI persistence 2.307, naive 2.488.

## Official SUNSET run, read from the saved notebook outputs (SUNSET_forecast.ipynb)
- forecast_dataset.hdf5: trainval **149,680**, test **12,248** (6,197 sunny / 6,051 cloudy).
  - The roadmap's 161,928 dev count does not match this run.
  - The test cadence is 1 min, against 2 min for dev.
- Parameters: 13,672,041. Our PyTorch port matches exactly.
- All 10 folds early-stopped after 9 to 13 epochs (best at epochs 4 to 8), at 527 steps/epoch. Mean best val RMSE 2.01.
- Test ensemble RMSE: 0.612 sunny, 4.265 cloudy, 3.029 overall.
- Official persistence on its test set: 0.420 / 5.140 / 3.625.

## Manifest v2 (primary, supersedes v1): data/manifests/forecast_tol30_v2.npz
±30 s matching, dev at a 2-min cadence and test at a 1-min cadence. content_sha256 d92100ef...4677f (rebuilt identically).

| | Dev | Test | Test sunny / cloudy |
|---|---|---|---|
| v2 | 164,362 | 13,266 | 6,638 / 6,628 |
| Official notebook | 149,680 | 12,248 | 6,197 / 6,051 |

E0 persistence on v2 test (experiments/E0_persistence/tol30_v2):

| | Sunny | Cloudy | Overall |
|---|---|---|---|
| Ours, CSI persistence | 0.413 | 5.127 | 3.636 |
| Official | 0.420 | 5.140 | 3.625 |

Ours is within 0.013 kW of the official values, which supports the sample construction for evaluation.

## E1 SUNSET port (src/sunset.py)
A faithful PyTorch port; see the module docstring for the matched details and deviations.
The epoch is one full pass, and val loss is the exact fold MSE. Output: experiments/E1_sunset_port/tol30_v2.

### E1 result (2026-10-19, single run, fold seeds 0 to 9, manifest v2)
| Test RMSE kW | Sunny | Cloudy | Overall |
|---|---|---|---|
| **Our port, 10-model ensemble** | **0.594** | **4.269** | **3.046** |
| Published SUNSET notebook | 0.612 | 4.265 | 3.029 |
| Difference | -0.018 | +0.004 | +0.017 |

| Forecast skill vs CSI persistence | Sunny | Cloudy | Overall |
|---|---|---|---|
| Our port | -43.8% | 16.7% | 16.2% |
| Official (from notebook numbers) | -45.7% | 17.0% | 16.4% |

- **Validation:** per-fold best val RMSE has a mean of 2.035 (official 2.01). Best epochs were 3 to 5 (official 4 to 8).
- **Spread:** single-fold test RMSE ranged from 3.066 to 3.132, so the ensemble gains about 0.04 to 0.09.
- **Gate 0 caveat:** no numeric tolerance was preregistered before this run.
  - All three differences are smaller than the spread between individual fold models.
  - The sunny RMSE is 3% lower than published.
- **Gate 0 judgement:** the pipeline reproduces the published benchmark closely enough to proceed. The remaining deviations are listed above.
- **Finding to carry forward:** SUNSET is far worse than persistence on sunny days, in both the original and the port.

## Clear-sky PV anchor v1 (src/clear_sky.py, ClearSkyPV)
Pipeline: Ineichen clear-sky irradiance → Hay-Davies POA (tilt 22.5°, az 195°) → physical IAM on the beam component → P_clear = scale(doy) · POA_eff.
- `scale(doy) = a + b·cos + c·sin` (annual harmonic). Fitted on training days only: the per-day median of PV/POA (POA ≥ 300 W/m²), iteratively dropping days > 8% below the envelope.
- **Fallback:** a constant scale when training covers fewer than 10 calendar months. The fit raises an error if the scale is not positive.
- **Timestamp audit (dev only):** on 200 clear dev days, the best PV vs clear-sky alignment is a **0 min shift**, in both DST and standard-time months. Naive-local → America/Los_Angeles localisation produced no NaT.
- **Constant-scale v0, rejected:** clear-day K showed a seasonal drift from 1.01 (Feb) to 0.85 (Aug), consistent with missing cell temperature. The harmonic replaces it.
- **Harmonic, fitted on the first 70% of dev days:** clear-day K by month is 0.97 to 1.04. Median clear-day K is 1.001 on fit days and 0.972 on later held-out days, a slow drift (soiling/degradation?) to watch.
- **K distribution:** all dev records with P_clear ≥ 1 kW give median 0.986, 1st percentile 0.135, 99th percentile 1.50, and 6.6% above 1.2. Cloud enhancement and low-sun records; K is not clipped.
- **K definition:** K = P / max(P_clear, 1 kW), which is invertible everywhere and is the prespecified dawn/dusk handling.
- **Known limitation:** no temperature model, and the altitude (30 m) is assumed.

## Temporally blocked dev folds (src/folds.py → data/manifests/folds_blocked5_v1.json, sha256 ad4ba6a2…)
- **Folds:** 5 contiguous chronological blocks of 99 or 100 days, with a 1-calendar-day purge either side. The official test days are asserted absent.
- **Day sky label** (dev stratification only, never an input): sunny ⇔ corr(PV, clear-sky POA) > 0.995 and mean |ΔPV| < 0.15 kW/min.
  - It is scale-free, so nothing is fitted on validation days.
  - It **matches the official labels on 20/20 test days**. 200 of 497 dev days are sunny.
  - "Cloudy" in dev therefore means "not clear all day", which is broader than the official cloudy test days.

| Fold | Validation span | Sunny val days | Val days after 2019-05-22 (camera hypothesis) |
|---|---|---|---|
| 0 | 2017-03-09 → 2017-09-12 | 38/100 | 0 |
| 1 | 2017-09-13 → 2018-05-15 | 26/100 | 0 |
| 2 | 2018-05-16 → 2018-09-21 | 47/99 | 0 |
| 3 | 2018-09-22 → 2019-07-16 | 30/99 | 51 |
| 4 | 2019-07-17 → 2019-10-26 | 59/99 | 99 |

## E2/E3 pilot (src/train.py → experiments/E2E3_cnn_gru/v1)
- **Model:** CNN-GRU (SmallSkyEncoder, 64 features, hidden 64) with a point head. Inputs are 16 chronological frames plus PV/30 history.
- **Training:** AdamW at lr 3e-4 and weight decay 1e-4, max 50 epochs, patience 10, seed 42, bf16 autocast.
- **Batch 128, not the roadmap's 16,** for GPU throughput; this is a starting value, not tuned.
- **E2:** MSE on kW/30. **E3:** MSE on K, reconstructed with P_clear at the target time. Selection in both is val RMSE in kW.
- **Reference:** smart persistence with the same fold-fitted anchor.

## Phase 2.6 preprocessing audit on real data (tests/test_real_data_audit.py, `pytest -m realdata`)
Run 2026-10-19 on forecast_tol30_v2 and folds_blocked5_v1: **7/7 passed** in 65 s.

| Check | What is asserted | Result |
|---|---|---|
| Image readability | Referenced rows in range, shape 64×64×3 uint8. No blank (std < 1) or saturated (mean > 250) frame anywhere. | 0 bad of 363,375 images |
| Time ordering and gap policy | Column 0 is the origin; frames strictly decreasing; each frame within ±30 s of t − j min; origins unique | pass |
| Forecast leakage | Target after every history frame; target within ±30 s of t + 15 min | pass |
| Day and split leakage | All frames and the target on the origin's day; no dev/test record mixing; test flags = official days. Folds: train/val/purge disjoint, every dev day has a role, no train day inside a val span, each day validated once. | pass |
| Units | PV finite, in [0, 35) kW; targets finite and ≥ 0 | pass |
| Clear-sky stability | Anchor fitted per fold on training days. P_clear finite and ≥ 0; K finite for every dev target; all 5 fits seasonal. | median K 0.99 to 1.01; p99 1.51 to 1.53; max ≈ 2.95; 0.11 to 0.13% of targets have K > 2 |
| Determinism | Rebuilding the manifest and the folds reproduces both stored hashes | pass |

Observation: a small tail of K up to ~3 exists; these are probably low-sun records where P_clear is floored at 1 kW. It is finite, so it is allowed, but it adds squared-error weight to E3's K loss. Check where these occur if E3 underperforms.

## E2/E3 v1 pilot: stopped, then diagnosed (2026-10-19)
The v1 recipe (lr 3e-4 constant, no clipping, bf16 autocast, global-average-pool encoder) was stopped on fold 1.
Validation RMSE swung epoch to epoch (E2: 2.94 → 3.98 → 2.94; E3: 3.28 → 5.90) while train loss fell smoothly.
Fold 0 v1 results, kept as a record:

| | Sunny | Cloudy | Overall |
|---|---|---|---|
| E2 | 0.747 | 2.226 | 1.819 |
| E3 | 0.960 | 2.569 | 2.115 |
| Smart persistence | 0.460 | 2.515 | 2.009 |

Diagnostics:
- **Data path:** the GPU batches equal the reference Dataset samples exactly (64 random samples). No bug.
- **Fold difficulty:** fold 1 (Sep 2017 to May 2018, mostly winter) is much harder; smart persistence is 3.434 vs 2.009 on fold 0. So v1 E2's 2.94 on fold 1 was already 14% skill. The problem is instability, not the level.
- **BN mode and bf16 at inference:** no effect (2.927 kW in all four combinations).
- **Visual check** (reports/visual_check_sequences.png): the sun moves left (morning) to lower right (afternoon); input windows and targets sit correctly on the day's PV curve.
- **Sensor faults:** 35 flat-PV runs of ≥ 10 records above 1 kW (554 records, 0.16%; longest 38 min on 2017-03-09). 66 one-minute jumps above 15 kW (max 22 kW), plausible cloud edges. Kept as-is, like the official pipeline.
- **Image/PV pairing offset (finding):** on cloudy dev days, image-brightness changes correlate best with PV changes **one minute earlier**.
  - Whole-image mean: r = 0.058 at +1 vs 0.031 at 0.
  - Sun-patch tracker: r = 0.068 at +1 vs 0.020 at 0. Sunny days show no signal.
  - Signals are weak (sun saturation, auto-exposure) but consistent across two proxies.
  - Likely cause: PV logged as 1-min averages labelled at interval start.
  - Kept the official pairing for comparability. Event-arrival labels (Phase 6) carry ±1 min timing uncertainty; a "shift images by 1 min" ablation is a cheap later check.

Fixes, now trainer defaults (v2) applied identically to E2 and E3:
- bf16 for the CNN only; GRU and output head always fp32, so the regression output is not bf16-quantised.
  - Full fp32 was tried first but is 7× slower on this GPU (220 vs 33 ms/step, batch 128).
  - Global bf16 did not cause the instability: inference RMSE was identical in bf16 and fp32.
- lr 1e-4 with per-step cosine decay;
- grad-norm clipping at 1.0;
- EMA of weights and BN buffers (decay 0.999 with (1+n)/(10+n) warm-up), used for validation and checkpoints.

Bugs found while adding these:
- **EMA without warm-up** left 37% of the random init after 1,000 steps; fixed.
- **GRU weights non-contiguous after the EMA deepcopy;** `flatten_parameters()` is now called in forward.

Architecture option `pool='spatial'`: keeps a 4×4 grid before the projection (position-aware, like SUNSET's flatten) instead of a global average that discards where clouds are relative to the sun. It is compared on fold 1 before adoption.

### Fold-1 check of the v2 recipe (kW target, 8 epochs, cosine over 8; experiments/diag_v2)
| | Best val RMSE (epoch) | EMA val curve | Sunny | Cloudy |
|---|---|---|---|---|
| v2, avg pool | **2.995** (4) | 3.17, 3.02, 3.00, 2.995, 3.01, 3.01, 3.03, 3.03 | 0.716 | 3.467 |
| v2, spatial pool | 3.119 (3) | 3.20, 3.14, 3.12, 3.12, 3.13, 3.14, 3.16, 3.17 | 1.060 | 3.583 |
| v1 | 2.943 (3), noisy | 3.03, 3.98, 2.94, 3.13, 2.96, 3.13 | – | – |
| Smart persistence | 3.434 | – | 0.382 | 3.999 |

- **Instability resolved:** the EMA curve is smooth while the raw weights still swing (3.96 at epoch 2).
- **Pooling:** spatial pooling overfits (lower train loss, worse val), so avg pooling is kept.
- **Training length:** peaks at epochs 3 to 4. Full runs therefore use max 15 epochs (cosine over 15) and patience 5.
- **Sunny days:** the CNN-GRU is worse than smart persistence (0.72 vs 0.38), as SUNSET is.

## E2/E3 v2 (experiments/E2E3_cnn_gru/v2)
Launched 2026-10-19 at commit 9a0681d: avg pool, lr 1e-4 cosine over 15 epochs, clip 1.0, EMA 0.999, patience 5, seed 42, 5 blocked folds.

## E3b: POST-HOC DIAGNOSTIC EXTENSION (experiments/E3b_csi_inputs/v2)
**Why it was added:** E3, as pre-specified, changes only the target. Its PV-history input stays in raw kW, with no time-of-day or clear-sky input, so the model must infer the current clear-sky level from images to predict K. That confounds the test of physical normalisation.

**E3b:** as E3, but the history input is K(t − j), computed with the same fold-fitted anchor. Everything else is identical:
- architecture, average pooling, optimiser, schedule, clipping, EMA, seed, batch, early stopping;
- evaluation samples, K definition, 1 kW floor.

It was decided after seeing E3 folds 0 and 1. It is therefore reported as a diagnostic, not as a pre-registered experiment.

Launch: a watcher starts E3b only after both E2 and E3 write summary.json, and only if `src.audit_e3b` passes (commit 9b8dd69).

### E3b pre-training audit (experiments/E3b_csi_inputs/v2/audit.json): PASS on all 5 folds
- **Leakage:**
  - The anchor refitted with all non-training PV replaced by uniform noise gives bit-identical coefficients.
  - The model's history input equals K at exactly the 16 history rows, all ≤ origin; the target is strictly after.
- **Extreme K:**
  - Input K: median 0.993, p99 1.50, p99.9 2.03, max 3.34 (16 of ~2.6 M values > 3).
  - Target K: max 2.95, with 194 targets > 2 in fold 2.
  - All below the pre-registered abort limit, K > 5.
- **Low-sun denominators:**
  - The 1 kW floor is active for 4.3% of input records and 3.3% of targets, almost all at 06–07 h (100% at 06 h, 21% at 07 h) and 18–19 h (5% and 44%).
  - Only 30 of 194 K > 2 targets have the floor active. The tail is mostly genuine enhancement or anchor error, not denominator blow-up.

### Results, 2026-10-19 (reports/E2_E3_E3b_comparison.md)
Pooled validation RMSE in kW over 164,362 identical samples. Δ vs E2 is the paired day-block bootstrap 95% CI.

| | Overall | Clear | Cloudy | Δ overall vs E2 |
|---|---|---|---|---|
| E2 (kW in, kW out) | **2.074** | 0.726 | **2.643** | – |
| E3 (kW in, K out) | 2.329 | 0.919 | 2.947 | +0.255 [+0.19, +0.33] |
| E3b (K in, K out), post-hoc | 2.087 | 0.727 | 2.661 | +0.013 [−0.02, +0.05] |
| Smart persistence | 2.289 | **0.390** | 2.977 | – |

**Gate 1 decision:** the criterion is not met.
- Do not claim that normalisation improves accuracy.
- Never use target-only normalisation.
- Keep the consistent K representation (E3b) as a representation choice, and carry kW and K variants into E4.
- Seeds 123 and 456 are needed before final claims.
- The shared weakness is clear days, where both models lose to persistence by about 0.34 kW.

Comparison tool: `src/compare.py`. It asserts identical validation samples across models. It reports per-fold and pooled RMSE (overall / clear / cloudy) and 95% CIs for paired RMSE differences, from a day-block bootstrap stratified by clear/cloudy day. Validation folds only; the official test set is not read.

## Open items
- SUNSET reproduction keeps its own shuffled day-block folds; all E2+ experiments use the blocked folds above.
