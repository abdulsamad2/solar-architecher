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

## Open items
- SUNSET reproduction keeps its own shuffled day-block folds; all E2+ experiments use the blocked folds above.
