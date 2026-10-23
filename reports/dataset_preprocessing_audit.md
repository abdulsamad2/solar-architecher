# Dataset and preprocessing audit (SKIPP'D, SHADOW-MoE)

2026-10-23.
- **Scope:** documentation and audit only. No code was modified, no model retrained, no official test forecast regenerated. All counts come from existing manifests and run records.
- **Worked example:** uses development records only.
- **Evidence convention:** `file:line` refers to the repository at commit 0f71571.
- **What each operation section records:**
  - **What:** a plain-language description.
  - **Why:** the motivation.
  - **Rule:** the exact rule or formula.
  - **Uses:** which data the operation reads: training (fit) days, calibration days, validation days or test days.
  - **Evidence:** code locations, configuration and tests.
  - **Uncertainty:** open questions, and any mismatch between code and reports.

---

## 1. Raw dataset files and formats

All raw files sit in `data/raw/`. Checksums are in `data/raw/SHA256SUMS.txt`. Source: <https://purl.stanford.edu/dj417rh1007> (CC BY 4.0).

| File | Content (as read by the code) | SHA-256 (first 16 hex) |
|---|---|---|
| `2017_2019_images_pv_processed.hdf5` (4.47 GB) | Group `trainval`: `images_log` (349,372, 64, 64, 3) uint8; `pv_log` (349,372) float, kW. Group `test`: the same with 14,003 rows | 52B68A705C56D1E7 |
| `times_trainval.npy` | 349,372 timestamps; object array of naive `datetime.datetime` (local clock time); range 2017-03-09 06:47:00 to 2019-10-26 17:59:20; daytime only (hours 6–19) | 57DF8528A32BE668 |
| `times_test.npy` | 14,003 timestamps, same type; grouped by day, **not globally chronological** | D63555E441B6BBCF |

- **Evidence:**
  - shape and dtype assertion: `tests/test_real_data_audit.py:56`;
  - loading: `src/skippd.py:96-97` and `:209-212`;
  - timestamp type: inspected directly;
  - the non-chronological test file: `reports/test_execution_deviations.md`, Deviation 2.
- **Not in the repository:** the raw 2048×2048 video or the ~10 s image stream used by the dataset authors. Every image therefore arrives already processed at 64×64 by the dataset authors. How they processed it is outside this repository.

## 2. Images: shape, colour, resizing, cropping, masks, scaling, augmentation

| Operation | Implemented? | Details |
|---|---|---|
| **Stored format** | – | uint8, 64×64×3, one image per record (`tests/test_real_data_audit.py:56`). Downsampling from the camera's 2048×2048 was done by the dataset authors before release, **not by this code**. |
| **Colour conversion** | **No** | Channels are used as stored; the code assumes the order is RGB, as documented by SKIPP'D. The only grey-scale conversion is for the D4 motion features (mean of the 3 channels, rounded to uint8; `src/final_eval.py` `run_test`, and `src/e8_features.py` record table). D4 is a descriptive comparator, not part of G3″-F. |
| **Resizing** | **No** | No resize anywhere in `src/` (searched for `resize` / `interpolate`). |
| **Cropping** | **No** | No crop. |
| **Masking** (e.g. the black area outside the fisheye circle) | **No** | No mask in the image pipeline. The `mask` variables in `src/train.py:36,231` and `src/final_eval.py:119` are E8 *feature-column* masks, not image masks. |
| **Pixel scaling** | **Yes** | `x = uint8 / 255` → float32 in [0, 1] (`src/train.py:68`, `GpuSeq.__call__`; also `src/skippd.py:228`). No mean/std normalisation. Each CNN block starts with a convolution followed by BatchNorm (`src/models.py:23`, `:73`), which normalises learned features, not inputs. |
| **Tensor layout** | – | `[batch, 16 frames, 3, 64, 64]`, frames oldest first (`src/train.py:66-71`). |
| **Data augmentation** | **No** | No flips, rotations, colour jitter, noise or random crops (searched for `augment`, `flip`, `crop`, `random`). |
| **Image quality filter** | **Check only** | `tests/test_real_data_audit.py:47-66` asserts that no blank (pixel std < 1) or saturated (mean > 250) image is used by any sample. It is an assertion, not a filter: no sample was removed by it. |

- **Uncertainty:**
  - The channel order (RGB vs BGR) is taken from the dataset documentation. Our camera-change analysis (R/B ratio) assumes channel 0 is red.
  - Any masking or processing by the dataset authors before release is not visible in this repository.

## 3. PV data: cleaning, units, filtering, clipping, clear-sky normalisation, residuals

| Operation | Rule and evidence |
|---|---|
| **Units** | kW as stored (`pv_log`). Checked finite, ≥ 0 and < 35 kW in both groups (`tests/test_real_data_audit.py:111-117`). The observed range is 0–29.6 kW (`reports/protocol_notes.md`). |
| **Cleaning or filtering of PV values** | **None in this code.** Values are used as released. No outlier removal, interpolation or smoothing. |
| **Clipping** | **None** on PV. K is **not** clipped to ≤ 1 (cloud enhancement allowed), as stated at `src/clear_sky.py:113`. |
| **Scaling (kW models only)** | `PV_SCALE = 30.0` kW (`src/train.py:34`). Used only by the early kW-target and PV-history variants (E2/E4-A). |
| **Clear-sky POA** | Ineichen clear-sky irradiance → Hay-Davies transposition (tilt 22.5°, azimuth 195°) → beam × physical IAM + diffuse, clipped ≥ 0 (`src/clear_sky.py:27-47`). Site: lat 37.424107, lon −122.174199, altitude 30 m (assumed) (`:17-18`). |
| **Clear-sky PV anchor** | `P_clear(t) = s(doy) · POA_eff(t)`, with `s(doy) = a + b·cos(ω·doy) + c·sin(ω·doy)` and ω = 2π/365.25 (`src/clear_sky.py:63-109`). Fit rule (`:82-105`): per day, the **median** of PV/POA over records with POA ≥ 300 W/m² → least squares on the harmonic → iteratively drop days more than 8% below the fit (3 iterations) → require the scale > 0 all year. |
| **Clear-sky index** | `K = PV / max(P_clear, 1 kW)`, with `min_clear_kw = 1.0` (`src/clear_sky.py:112-114`; config `src/train.py:448`). Inverse: `PV = K · max(P_clear, 1)` (`:117-118`). |
| **Model target** | `K(t+15)` at the matched target record (`src/train.py:249`). Forecasts are mapped back to kW with `P_clear(t+15)` (`:255-260`); multiplying by a positive factor preserves quantile order. |
| **Model PV input** | The K history at the 16 frame records (`--input csi`; `src/train.py:252-253`), from the same anchor. |
| **Residual / ΔK transformations** | **E6-S** (event head): labels are ΔK over t+1…t+15 with a 4-class arrival time and severity, built from the fold anchor. They are labels only, never inputs (`src/train.py:273-287`). Output quantiles are `K(t) + ΔK` quantiles (`:150-151`). Threshold τ = 90th percentile of \|K(t+15) − K(t)\| over **fit-day** samples, asserted equal to the frozen value (`:279-282`; `reports/phase6_frozen_thresholds.json`). The optional `--residual` path (`:265-272`) is not used by any final expert (`residual: False` in the configs). |
| **Smart persistence (internal)** | `P(t) · P_clear(t+15) / P_clear(t)`; plain persistence when `P_clear(t) < 1 kW` (`src/clear_sky.py:121-127`). |
| **Official persistence (reported baseline)** | The verbatim `Relative_output` from the SKIPP'D authors (A_e 24.98 m², tilt 22.5°, azimuth 195°), evaluated at exactly t+15 min (`src/baselines.py:17-24`; `src/official/relative_op_func.py`). |

**Which data each fit uses:**
- The anchor is fitted **per fold**, on that fold's **fit days** only: training days minus calibration and purged days. See `src/final_eval.py:96-100` (`member_anchor`) and `src/train.py:237-241`.
- Validation and test records are only *evaluated* with that anchor.

**Uncertainties:**
- There is no cell-temperature model, so the annual harmonic absorbs temperature and seasonal effects (`src/clear_sky.py:66-67`).
- The `fit_scale` function (`:50-60`, 95th-percentile envelope) is **legacy and unused** by `ClearSkyPV.fit`, which uses the per-day median plus cloudy-day rejection. The module docstring (`:6-7`) points to `fit_scale`. That is a documentation mismatch only.
- The 30 m altitude is an assumption.

## 4. Timestamps: parsing, time zone, sorting, pairing, tolerance, duplicates

| Operation | Rule and evidence |
|---|---|
| **Parsing** | `np.load(..., allow_pickle=True)` gives `datetime.datetime`, converted to `datetime64[us]` (`src/skippd.py:47-48`, `:96-98`). |
| **Time zone** | Treated as **naive local clock time** (PST/PDT). It is localised to America/Los_Angeles only inside the clear-sky computation. Ambiguous and nonexistent DST times become NaT, giving P_clear = 0 (`src/clear_sky.py:21-24`, `:35-46`). DST transitions happen at night and do not touch the daytime records. |
| **Sorting** | The dev and test timelines are concatenated, then stable-argsorted by time. Each record keeps its HDF5 (group, row) location (`src/skippd.py:93-102`). Sample construction requires strictly increasing times, otherwise it raises (`:66-67`). |
| **Image/PV pairing** | Each HDF5 row already pairs one image with one PV value (as released). The code does not re-pair them. |
| **Matching tolerance** | Each required time (t − j min, t + 15 min) is matched to the **nearest** record within **±30 s** (`src/skippd.py:51-54`, `:68-73`). `tolerance_s = 30.0` in the manifest (`data/manifests/forecast_tol30_v2.json`). |
| **Tie-breaking** | On an exact tie between two neighbours, the **earlier** record is chosen (`<=` in `src/skippd.py:53`). |
| **Duplicate records** | None can exist: the build raises unless times are strictly increasing (`src/skippd.py:66-67`), and the build succeeded. |
| **Duplicate frame use** | A sample is invalid if two frame slots match the same record. Frames must be strictly ordered (`np.diff(hist) < 0`, `src/skippd.py:74`; `src/e8_audit.py:28`). |

**Uncertainties and discrepancies:**
- **Timestamp drift.** The public 1-min records drift in their second-of-minute phase. Exact matching (tol 0) loses about 19% of samples (Section 8). This is why ±30 s was adopted, a documented deviation (`reports/protocol_notes.md`).
- **DST errors in the source.** The SKIPP'D authors report daylight-saving adjustment failures in the 2017–2018 data. **This repository does not correct them.**
- **Pairing offset.** Image changes correlate best with PV changes about 1 minute earlier (`reports/manuscript_draft.md` §4). This is not corrected; it is kept for comparability.

## 5. Sample-window construction

For an origin record at time t (`src/skippd.py:57-90`, called by `build_forecast_manifest` at `:105-166`):

| Element | Rule |
|---|---|
| Forecast origin t | A record time that passes all validity checks and the greedy thinning gate |
| H15 image/PV history (all models except A2/D4) | 16 records matched to t − j min, j = 0…15. Stored newest first (`history[:, 0] = t`), reversed to oldest first for the models (`src/skippd.py:198-199`). |
| Target | The record matched to t + 15 min; it must be strictly after t (`:73-75`) |
| Thinning (cadence) | Greedy: keep a valid candidate i only if `t_i − t_last_kept > interval − 1 s` (`:77-82`). Interval = **2 min for development, 1 min for test** (`dev_sampling_min = 2`, `test_sampling_min = 1`, `:109-110`, `:118-119`) |
| H60 history (A2, D4) | 16 records at t − 4j min, j = 0…15 (t − 60 … t), ±30 s, **same calendar day**, strictly ordered (`src/e8_audit.py:16-29`). Used for development (`src/e8_features.py:103-117`) and test (`src/final_eval.py:64-70`). |
| K history (input) | `K` at each of the 16 frame records (Section 3) |
| Clear-sky reference times | P_clear at the frame record times and at the target record time (`src/train.py:242-244`) |

**Uses:** only records at or before t enter the inputs. The target is the only future record (`tests/test_real_data_audit.py:80-84`).

**Uncertainty:** the target record may be up to 30 s from the nominal t + 15 min. The official persistence baseline uses exactly t + 15 min (`src/baselines.py:4`).

## 6. Missing values, invalid examples, day boundaries, eligibility, fallback, exclusions

| Rule | Evidence |
|---|---|
| **Missing frame or target:** if any of the 16 history slots or the target has no record within ±30 s, the sample is **not created** | `src/skippd.py:72-74` |
| **Day boundaries (H15 and target):** nothing in the H15 builder explicitly checks for the same day. Matches more than 30 s away are impossible across the overnight gap, and a real-data test asserts that no history frame or target crosses days | `tests/test_real_data_audit.py:87-92` |
| **Day boundaries (H60):** explicitly required to be on the same day | `src/e8_audit.py:25-27` |
| **H60 eligibility:** all 16 H60 frames are present, same day and ordered. Mainly fails in the first hour after the first record of the day (sunrise) | frozen in `data/manifests/e8_eligibility_v1.npz` (hash-checked, `src/e8_features.py:35-37`) |
| **15-min fallback (G3″-F):** if a sample is not H60-eligible, G3″-F uses G3 (E4-B-fit + E6-S). The rule depends on timestamps only | `src/final_eval.py:148-159` |
| **E8/E9 training population:** H60 experts (A2, D4) are trained and validated on the **common** population (eligible for H15, H30 and H60). H60-eligible but non-common validation samples are predicted separately | `src/train.py:221-235` |
| **Missing PV values:** none present (all finite) | `tests/test_real_data_audit.py:113` |
| **Other exclusions:** none. No clear-sky filter and no night filter (no night records exist). All days are kept, including the post-2019-05-22 camera-change days | – |

## 7. Partitioning and leakage prevention

| Element | Rule and evidence |
|---|---|
| **Official test** | 20 fixed days (10 sunny, 10 cloudy) from the SKIPP'D paper (`src/skippd.py:31-36`). The test samples are built on the full timeline, then selected by day (`:120-123`). |
| **Split-group assertion** | Every history frame and target of a sample must come from the same HDF5 group as the sample's split (`src/skippd.py:127-129`; `tests/test_real_data_audit.py:93-94`). |
| **Five blocked folds** | The 497 dev days, sorted, are cut into 5 contiguous blocks of 100/100/99/99/99 days (`src/folds.py:42-52`). Training days within **1 calendar day** of the validation block are purged. Test days are asserted absent (`:94`). |
| **Calibration hold-out (inside each fold)** | Training days in blocks of 10; block k is a calibration block iff k mod 5 = 2 (~20% of days). Fit days within 1 day of any calibration day are purged (`src/folds.py:55-73`). Calibration days are used for gate fitting and calibration scores, never for expert training. Overlaps are asserted (`src/train.py:205-219`). |
| **Day sky labels (dev)** | `sunny ⇔ corr(PV, POA_clear) > 0.995 and mean \|ΔPV/min\| < 0.15 kW`. Scale-free, no fitted parameters. **Used for stratified reporting only**, never as an input (`src/folds.py:29-39`). It reproduces the official sunny/cloudy labels on 20/20 test days (`folds_blocked5_v1.json` report). |
| **Leakage tests** | `tests/test_real_data_audit.py:69-108`: ordering, the tolerance, no target before the history, no day crossing, no split mixing, folds disjoint, and no training day inside a validation span. `tests/test_folds.py`: fold and calibration-split properties. |
| **Determinism of manifests** | Rebuilding the manifest and folds reproduces the frozen SHA-256 (`tests/test_real_data_audit.py:143-151`). |

**Fold layout** (`data/manifests/folds_blocked5_v1.json`):

| Fold | Validation span | Validation days | Training days | Purged days | Sunny validation days |
|---|---|---|---|---|---|
| 0 | 2017-03-09 – 2017-09-12 | 100 | 396 | 1 | 38 |
| 1 | 2017-09-13 – 2018-05-15 | 100 | 395 | 2 | 26 |
| 2 | 2018-05-16 – 2018-09-21 | 99 | 396 | 2 | 47 |
| 3 | 2018-09-22 – 2019-07-16 | 99 | 396 | 2 | 30 |
| 4 | 2019-07-17 – 2019-10-26 | 99 | 397 | 1 | 59 |

## 8. Sample counts after each stage

All counts come from existing manifests and run records. Nothing was regenerated.

| Stage | Development | Official test |
|---|---|---|
| Raw paired records | 349,372 | 14,003 |
| Days | 497 | 20 |
| Exact matching (tol 0 s), 2-min cadence (`forecast_exact_v1.json`) | 130,412 | 5,569 |
| ±30 s matching, 2-min cadence for both (`forecast_tol30_v1.json`) | 164,362 | 6,638 |
| **±30 s, 2-min dev / 1-min test (`forecast_tol30_v2.json`, used for all final results)** | **164,362** | **13,266** (6,638 sunny / 6,628 cloudy) |
| H15-eligible | 164,362 (100%) | 13,266 |
| H30-eligible | 159,414 (97.0%) | – |
| H60-eligible | 150,758 (91.7%) | 12,355 (93.1%; `reports/test_results.json` routing share) |
| Common population (H15 ∩ H30 ∩ H60), used for E8/E9 training | 150,618 (91.6%) | – |
| Test exact-timing subset (Phase A1) | – | 11,086 |
| Ramp samples (\|ΔK\| ≥ τ) | 10.07% of dev validation samples (per-fold τ) | 1,916 (τ_test = 0.27948) |

Official reference counts: 161,928 dev samples (SKIPP'D roadmap); 149,680 dev and 12,248 test in the SUNSET notebook (`forecast_tol30_v2.json`). These were built from the unreleased ~10 s stream.

**Per-fold split of E4-B-fit (seed 42), in samples** (`experiments/E4_calib/v1/csi/seed42/fold_*/metrics.json`):

| Fold | Fit samples | Calibration samples | Validation samples | Fit / calibration / purged days |
|---|---|---|---|---|
| 0 | 97,936 | 26,087 | 35,525 | 302 / 80 / 14 |
| 1 | 104,290 | 28,040 | 26,163 | 300 / 80 / 15 |
| 2 | 98,052 | 26,067 | 35,267 | 303 / 80 / 13 |
| 3 | 99,109 | 26,436 | 33,308 | 302 / 80 / 14 |
| 4 | 98,700 | 26,383 | 34,099 | 303 / 80 / 14 |
| Total validation | | | **164,362** | |

## 9. Random and stochastic operations

**The data preprocessing is fully deterministic:** manifests, folds, calibration split, clear-sky anchor and eligibility. This is verified by hash reproduction (`tests/test_real_data_audit.py:143-151`). Randomness appears only in the places below.

| Operation | Seed control | Evidence |
|---|---|---|
| Model weight initialisation | `torch.manual_seed(cfg["seed"])`, seeds 42 / 123 / 456 | `src/train.py:294` |
| Mini-batch order | `torch.Generator(...).manual_seed(seed)` + `randperm` per epoch | `src/train.py:321`, `:332` |
| GPU numerics | **Not fully deterministic**: bf16 CNN and TF32 matmul, with no `torch.use_deterministic_algorithms`. This produced up to about 1e−3 kW reproduction differences, documented in Amendment 2 | `src/train.py:478-479`; `reports/test_eval_prereg.md` |
| SUNSET-port fold assignment | Verbatim official logic: `np.random.seed(1)` day-block shuffle, `np.random.seed(fold)` within folds | `src/sunset.py:61-81` |
| Statistical bootstrap (evaluation, not preprocessing) | `np.random.default_rng(20261022)` | `src/final_eval.py:208-216` |
| Image augmentation | **None exists** | – |

## 10. Unit tests supporting the claims

| Test | What it checks |
|---|---|
| `tests/test_skippd.py` | Exact-rule frame stacking and target; 2-min gate; phase shift breaks exact but not tolerant matching; gaps drop samples; bad inputs rejected |
| `tests/test_skippd_dataset.py` | Dataset reads the correct frames and targets; rejects cross-split IDs; official persistence formula |
| `tests/test_folds.py` | Blocked folds contiguous, disjoint and purged; calibration split disjoint, purged and spread |
| `tests/test_clear_sky.py` | Anchor recovers a known scale; narrow-season fallback stays positive |
| `tests/test_e8_pipeline.py` | Frozen hashes; window counts; causal, same-day 16-frame windows; H15 window equals the manifest history |
| `tests/test_final_unsorted.py` | Feature mapping for non-chronological record files |
| `tests/test_real_data_audit.py` (opt-in, `-m realdata`) | Tests 1–8 on the real data (image quality, ordering, leakage, units, clear-sky stability, determinism) |

---

## Worked example (development data only)

**The sample.**
- Manifest row 67,473 (development index 61,842).
- Day **2018-05-16** (development label: cloudy); a validation day in **fold 2**.
- **Origin t = 2018-05-16 11:58:40.**

**H15 window.** The 16 frames run newest first from j = 0 to 15. All match exactly (offset 0 s), at HDF5 `trainval` rows 133,464 down to 133,449:

```
j : nominal / matched record time
0 : 11:58:40      4 : 11:54:40      8 : 11:50:40     12 : 11:46:40
1 : 11:57:40      5 : 11:53:40      9 : 11:49:40     13 : 11:45:40
2 : 11:56:40      6 : 11:52:40     10 : 11:48:40     14 : 11:44:40
3 : 11:55:40      7 : 11:51:40     11 : 11:47:40     15 : 11:43:40
target: nominal 12:13:40, matched 12:13:40 (offset 0 s)
```

**H60 window.** Eligible. Newest first: 11:58, 11:54, …, 11:02, 10:58, at a 4-min stride (the A2 input).

**Clear-sky index.**
- Fold-2 anchor coefficients, fitted on that fold's 303 fit days (80 calibration and 13 purged days excluded): (a, b, c) = (0.026059, 0.001535, 0.001478) kW per W/m².

| Quantity (oldest → newest) | Values |
|---|---|
| PV history (kW) | 4.33, 4.08, 3.84, 3.63, 3.45, 3.34, 3.27, 3.23, 3.19, 3.17, 3.17, 3.22, 3.28, 3.34, 3.42, 3.48 |
| P_clear at the frames (kW) | 23.62 … 24.32 |
| **K history (model input)** | 0.183, 0.172, 0.162, 0.153, 0.145, 0.140, 0.137, 0.135, 0.133, 0.132, 0.132, 0.133, 0.136, 0.138, 0.141, 0.143 |

- Target: PV(t+15) = 4.99 kW, P_clear(t+15) = 24.92 kW, so **target K = 0.200**.

**Tensor shapes for one sample:**

| Tensor | Shape | Content |
|---|---|---|
| Stored image | (64, 64, 3) uint8 | Range 0–237 at the origin frame |
| Image input | [16, 3, 64, 64] float32 | value / 255, frames oldest first (A2: the H60 frames) |
| K-history input | [16] float32 | One K value per frame |
| Model output | [9] | K quantiles at τ ∈ {0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95} |
| Forecast in kW | [9] | `K_q × max(P_clear(t+15), 1)` |

---

## Discrepancies and open points
1. **Manifest versions.** The project memory and early notes name `forecast_tol30_v1` as primary. All final results (Gate 5, Phase A, Phase C) use **`forecast_tol30_v2`**, which differs only in the 1-min test cadence (`src/final_eval.py:37`).
2. **Unused scaffold.** `src/data.py` contains an early, unused pipeline (CSV manifest, PIL image loader). No experiment imports it.
3. **Stale docstring.** The `src/train.py` docstring describes PV/30 history inputs. The final experts use K history (`--input csi`).
4. **Legacy function.** `src/clear_sky.py` describes `fit_scale`, but the anchor actually uses `ClearSkyPV.fit` (median plus harmonic plus cloudy rejection).
5. **Not corrected:**
   - the DST failures in the source data;
   - the ~1-min image/PV offset;
   - the 2019-05-22 camera change (kept, flagged in `src/folds.py:26`).
6. **Dataset-side processing unknown:** any processing of the images by the dataset authors before release (masking, colour handling, downsampling method).

---

## Publication-ready subsection: Dataset and preprocessing

**Dataset.**
- We use the SKIPP'D benchmark (Nie et al., 2023): 64×64 RGB sky images and the co-located PV output of a 30.1 kW-DC rooftop array at Stanford University (tilt 22.5°, azimuth 195°), released as paired records at about 1-minute resolution.
- The development set has 349,372 records over 497 days (9 March 2017 – 26 October 2019). The official test set has 14,003 records over 20 fixed days (10 sunny, 10 cloudy).
- Images were used as released: no further resizing, cropping, masking, colour conversion or augmentation was applied. Pixel values were scaled to [0, 1].
- PV values were used in kW without filtering, interpolation or clipping.

**Sample construction.**
- Each forecast sample consists of 16 images and PV values at t − j min (j = 0…15) and the PV value at t + 15 min.
- Because the released records are thinned to about one per minute with drifting second-of-minute phase, each required time was matched to the nearest record within ±30 s, with the earlier record chosen on ties. Samples with any missing slot, or with two slots matching the same record, were discarded.
- Origins were thinned greedily to a 2-min cadence for development and a 1-min cadence for testing, following the official construction. This yielded 164,362 development and 13,266 test samples. Exact matching would have retained 130,412 and 5,569.
- No sample spans two days or mixes development and test records; both were verified programmatically.

**Clear-sky normalisation.**
- PV is expressed as the clear-sky index K = PV / max(P_clear, 1 kW).
- P_clear is the Ineichen clear-sky plane-of-array irradiance (Hay–Davies transposition, physical incidence-angle modifier, pvlib) multiplied by an annual-harmonic scale.
- The scale is fitted per cross-validation fold on that fold's model-fitting days only: per-day median PV/POA for POA ≥ 300 W m⁻², with iterative rejection of days more than 8% below the fit.
- Models receive the 16 K values as inputs and predict quantiles of K at t + 15 min, which are mapped back to kW using P_clear(t + 15).

**Long-history window.**
- The 60-minute expert uses 16 frames at a 4-min stride, required to lie on the same day.
- Samples without such a window (about 8% of development and 7% of test samples, mainly in the first hour of each day) are routed to the 15-minute configuration by a timestamp-only rule.

**Partitioning.**
- Development days were split into five contiguous, chronologically ordered folds of 99–100 days, with a 1-day purge between training and validation blocks.
- Within each fold, about 20% of training days (blocks of 10 days, every fifth block) were held out as calibration days for gate fitting and interval calibration, again with 1-day purges.
- Day-level sunny/cloudy labels were derived from a parameter-free rule (correlation with clear-sky irradiance > 0.995 and mean absolute minute-to-minute change < 0.15 kW). They are used only for stratified reporting, and reproduce the official test-day labels on all 20 days.

**Reproducibility.**
- All preprocessing steps are deterministic. Re-running them reproduces the frozen manifests bit for bit (SHA-256 verified).
- Randomness enters only through model initialisation and mini-batch order (seeds 42, 123, 456) and non-deterministic GPU arithmetic.

**Known data limitations** (not corrected, to preserve comparability with published results):
- daylight-saving errors reported by the dataset authors for 2017–2018;
- an apparent ~1-min image/PV offset;
- a camera change on 22 May 2019.
