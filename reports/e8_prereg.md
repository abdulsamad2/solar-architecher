# E8 pre-registration (FROZEN): longer history and solar-relative motion

**Status: FROZEN 2026-10-21, after user approval, before any E8 feature extraction or training run.**
- **History:** this supersedes `e8_prereg_DRAFT.md` (kept). Changes vs the draft are listed in §0.
- **Frozen inputs (SHA-256):**
  - `reports/e8_geometry_frozen.json` = 308807C6…41C0;
  - `data/manifests/e8_eligibility_v1.npz` = 2F4CDB9A…BB0B.
  - Both are re-verified at run time; a mismatch is a STOP.

## 0. Approved decisions and amendments (vs DRAFT)
1. **Approved as drafted:**
   - 16 frames with a variable stride (H15/H30/H60 = 1/2/4 min);
   - the common population as primary, with the E4-B-fit fallback population as secondary;
   - stage-2 conditionality;
   - margins δ_O = 0.02 kW and δ_S = 0.015 kW;
   - the G5 motion-gate features.
2. **Amendment A (short-baseline motion):**
   - **Rule:** motion features of every window frame are computed between that frame's record and the record nearest **1 minute earlier** (±30 s, same day), **independent of the window stride**. For H15 this equals the draft definition, except that the oldest frame now also has motion (its 1-min predecessor, still before the origin).
   - **Missing predecessor:** motion columns are zero.
   - **Units:** shifts are divided by the actual time difference of the pair, in minutes.
   - **Rationale:** at a 4-min stride, cloud displacements (≈ 5–12 px) approach the 12 px quadrant size, making phase-correlation velocities unreliable and the motion definition window-dependent.
   - **Implementation:** features are computed once per image record and gathered by window rows, so every arm sees identical per-frame features.
3. **Amendment B (sunrise-ramp subgroup):** in the fallback population, results are additionally reported for **sunrise ramps**: Gate 2 ramp samples that are *not* in the common population, i.e. those lost to the 60-min eligibility requirement.
- **Preserved unchanged:** E7b (gap not closed; G3/G4 trade-off) and all earlier verdicts.
- **Frozen references:** E4-B-fit, E6-S, G3, G4.
- **Test set:** the official 20-day test set stays sealed.

## 1. Objective
Determine whether (a) longer historical context and (b) solar-relative cloud-motion features improve:
- forecast-time ramp anticipation;
- ramp CRPS;
- ramp-aware expert routing.

These are evaluated separately for **enhanced direct-K forecasting** and for **enhanced expert gating**.

## 2. CPU feasibility and integrity audit (done; development data only)
### 2.1 History-window eligibility and the common population (`src/e8_audit.py`, `reports/e8_eligibility_audit.json`)
Every window has **16 frames**, so the tensor shape, compute per sample and parameter count are matched. The stride grows with the window: **H15** 1 min, **H30** 2 min, **H60** 4 min. A sample is eligible iff every frame has a record within ±30 s on the same day, and the frames are distinct and ordered.

| Window | Eligible dev samples | Share | Validation ramp share | Validation sunny share |
|---|---|---|---|---|
| H15 (current) | 164,362 | 100% | 10.07% | 41.6% |
| H30 | 159,414 | 97.0% | 9.16% | 41.7% |
| H60 | 150,758 | 91.7% | 8.28% | 41.9% |
| **Common (H15 ∩ H30 ∩ H60)** | **150,618** | **91.6%** | **8.27%** | 41.95% |

**Per-fold validation retention (common / all):**

| Fold | Common / all | Retained |
|---|---|---|
| 0 | 32,645 / 35,525 | 92% |
| **1** | **22,200 / 26,163** | **85%** |
| 2 | 33,019 / 35,267 | 94% |
| 3 | 30,932 / 33,308 | 93% |
| 4 | 31,822 / 34,099 | 93% |

**Data-loss finding (material):**
- The 13,744 samples lost under the 60-min requirement are **early-morning**: 82% lie within the first 75 min of the day's records, with a median of 26 min after the first record.
- They are **ramp-rich: 29.8% ramps vs 10.1% overall.** Requiring H60 therefore removes about a quarter of all validation ramps (largely sunrise transitions), and the ramp share drops to 8.3%.
- The common-population result must not be read as covering those ramps.

Hence two evaluation populations (§5): the **common** population (primary, matched), and a **full-population fallback** (secondary, deployment-realistic).

### 2.2 Orientation and fisheye projection (`src/sky_geometry.py`, `src/e8_geometry_freeze.py`, `reports/e8_geometry_frozen.json`)
- **Method:** the sun was located per frame as the brightest 5×5-blurred spot (peak ≥ 240) at solar zenith < 75°. Frames were selected **by image criteria only**: no PV and no day labels, so the calibration is label-free.
- **Model:** (zenith, azimuth) is mapped to pixels by an equidistant, equisolid or orthographic lens law, with centre (cx, cy), scale f, rotation φ and handedness s. It is fitted robustly (soft-L1).

| Regime | Centre (px) | f (px/rad) | φ (rad) | Lens / handedness | n | Median / p90 error |
|---|---|---|---|---|---|---|
| before 2019-05-22 | (30.50, 28.94) | 17.89 | −1.283 | equidistant / mirrored | 40,313 | 0.71 / 1.49 px |
| from 2019-05-22 | (31.67, 31.04) | 18.00 | −1.246 | equidistant / mirrored | 13,743 | 0.63 / 1.20 px |

- **Orientation:** south at the bottom, east on the left, west on the right (a mirrored upward sky view). This is consistent with the earlier visual check (morning sun on the left).
- **Camera-change date confirmed: 2019-05-22.** The daily midday R/B ratio goes from 1.30 (May 21) to 2.29 (May 22). The optical centre moves about 2.5 px at the same date, so **solar-relative features use the regime-specific geometry**; the pre-change geometry on post-change images doubles the error.
- **Real-image check:** on 3,000 random dev frames (zenith < 70°), mean grey level vs distance from the *predicted* sun is 0.93 (0–2 px) / 0.80 / 0.67 / 0.55 (8–12 px). The sun sits at the crop centre.

### 2.3 Feature integrity and cost (CPU)
- **Values:** features are finite on 400 random samples per window. Motion magnitudes (px/min) shrink with stride as expected; the p99 of |shift| is 2.9 / 2.0 / 1.2 px/min for H15 / H30 / H60.
- **Speed:** extraction takes about 10 ms per sample single-threaded, about 27 min per window for all 164k dev samples, or about 1.3 h for three windows. It is parallelisable and precomputed once (float16, about 58 MB per window).
- **No feature–label association was examined** before freezing.

## 3. Feature-extraction specification (`src/solar_features.py`)
Per frame of the 16-frame window (chronological), 11 scalars, appended to the GRU input next to the existing K-history value:
- **P, solar position (5):** cos(zenith), sin(azimuth), cos(azimuth), sun_x / 64, sun_y / 64, using the regime geometry.
- **M, motion (6),** between frame k's record and the record ≈ 1 min before it (Amendment A; zero if missing):
  - mean |ΔI| (grey/255) in sun-centred rings of radius [0, 4), [4, 8) and [8, 12) px;
  - global shift (dx, dy) of the 24×24 sun-centred crop by Hann-windowed phase correlation with sub-pixel refinement, in **px per minute** (divided by the stride);
  - radial inflow: the mean over four 12×12 quadrant sub-crops of the shift component pointing **toward the sun**, in px/min.
- **Both crops of a pair are centred on the *current* frame's sun position,** so solar motion is not counted as cloud motion.
- **Causality:** all inputs are frames inside the window, at or before the origin.
- **Capacity matching:** arms without P and/or M receive those columns **zero-filled**, giving an identical architecture and parameter count in every arm.
- **Standardisation:** P and M columns are z-scored with mean and SD from the fold's **fit-day** samples.
- **Optional / not in primary arms:** a dense optical-flow field (e.g. Farnebäck) is *not* used. The phase-correlation and quadrant inflow summaries are the pre-specified "optical-flow summaries".

## 4. Arms
**All arms share:**
- the E4-B-fit configuration: direct K quantiles, K-history input, CNN-GRU (avg pool, 64/64), non-crossing 9-quantile head with climatology init;
- AdamW lr 1e−4 with cosine decay over 15 epochs, clip 1.0, EMA 0.999, batch 128, patience 5;
- `--calib-holdout` (same calibration days);
- seeds 42/123/456, 5 blocked folds;
- forecast horizon 15 min.

**Training and evaluation are restricted to common-population samples** in every arm, so each arm sees the same samples.

### Track A: enhanced direct-K forecasting (GPU)
Stage 1 (15 runs):

| Arm | History | P | M | Role |
|---|---|---|---|---|
| **D0** | H15 | 0 | 0 | **matched 15-min baseline** (same eligible samples) |
| D1 | H30 | 0 | 0 | history effect (30 min) |
| D2 | H60 | 0 | 0 | history effect (60 min) |
| D3 | H15 | ✓ | ✓ | solar + motion at 15 min |
| D4 | H60 | ✓ | ✓ | full |

**Stage 2 (6 runs, only if D3 or D4 passes the Track-A rule vs D0):** split P vs M at the history length of the passing arm (if both pass, the one with the lower seed-mean ramp CRPS):
- **D5:** P only;
- **D6:** M only.

### Track B: enhanced expert gating (CPU only; frozen experts; no retraining)
- **G3c:** the frozen G3 recipe (features g, d) refitted on the common population. This is the reference.
- **G5:** G3 features + motion summaries at the origin frame (radial inflow, inner-ring |ΔI|, and the shift magnitude of the last H15 step), with experts E4-B-fit + E6-S.
- **G3′ (only after Track A):** G3 with the direct expert replaced by the best passing D-arm.
- **Fitting:** exactly as in E7b. Calibration days only, chronological λ split, L2 logistic gate, exact blended CRPS.

## 5. Evaluation
- **Primary population:** common-population fold-validation samples, identical across arms (asserted).
- **Secondary population (fallback, deployment-realistic):** all 164,362 validation samples. The arm's forecast is used where the sample is eligible for its window, and **E4-B-fit** otherwise. Reported for every arm, including the matched baseline.
- **Primary endpoint:** pooled **ramp CRPS (kW)** of the raw predictive distribution (Gate 2 ramp stratum, frozen thresholds), with the same `crps_quantile_interp` scorer.
- **Secondary endpoints:**
  - overall, sunny, cloudy and non-ramp CRPS; per fold, with fold 1 and Sep–Oct 2017 separately;
  - event Brier score and **AUPRC** of the quantile-implied P(|ΔK₁₅| ≥ τ);
  - ramp-conditioned and overall 50/80/90% coverage, interval width and interval score after **causal rolling conformal** (E4-CAL2 procedure), reported separately from raw CRPS;
  - inference cost: feature extraction plus model latency.
- **Uncertainty:** paired day-block bootstrap 95% CIs (stratified sunny/cloudy) per seed, plus seed mean ± SD.

## 6. Decision rules (proposed)
**Non-inferiority margins:**
- overall δ_O = **0.02 kW** (as Gate 3 / E7);
- sunny δ_S = **0.015 kW**, about 3× the seed-to-seed SD of E4-B-fit's sunny CRPS (0.005 kW).

**Track A.** Arm X "improves ramp anticipation" vs its matched baseline D0 iff, in **≥ 2 of 3 seeds**:
1. ramp CRPS(X) − ramp CRPS(D0) < 0, with the CI entirely below 0;
2. overall CRPS(X) − overall CRPS(D0) ≤ δ_O;
3. sunny CRPS(X) − sunny CRPS(D0) ≤ δ_S.

Contrasts reported regardless of the rule:
- **history effect:** D1 − D0, D2 − D0;
- **feature effect at 15 min:** D3 − D0;
- **joint effect:** D4 − D0;
- **features given long history:** D4 − D2;
- **P vs M** (stage 2).

**Track B.** G5 "adds routing information from motion" iff, in ≥ 2 of 3 seeds:
- ramp CRPS(G5) − ramp CRPS(G3c) < 0 with CI < 0; and
- overall CRPS(G5) − overall CRPS(G3c) ≤ +0.005.

The same rule applies to G3′ vs G3c after Track A.

**Interpretation note:** a Track-A pass on the common population does not establish a benefit for sunrise ramps (excluded). The fallback population result is the deployment-relevant one.

## 7. Stop conditions (asserted)
- Any frame, feature or statistic from after the forecast origin. Window frames are ≤ origin; motion uses only in-window frames.
- Standardisation statistics, gate parameters or thresholds from non-fit / validation samples.
- Arms evaluated on different samples (asserted per population).
- A geometry other than the frozen regime geometry; a camera regime assigned by anything other than the date rule (≥ 2019-05-22).
- Non-finite features; crossing quantiles; invalid calibrated intervals; non-causal conformal updates.

## 8. Test plan
**Done (`tests/test_solar_features.py`, 11 tests passing):**
- the geometry fit recovers a known camera (median error < 0.3 px);
- the sun detector finds synthetic suns and rejects frames without a sun;
- the regime switch applies at the change date;
- solar scalars stay in range;
- sun crops are centred, with zero padding;
- phase correlation recovers integer and mixed shifts (< 0.3 px);
- radial-inflow sign is correct (toward the sun > 0);
- motion features are exactly zero for static frames and localised to the changed ring;
- motion scales with the stride (px/min);
- rings are disjoint.

**Before launch (to implement after approval):**
1. A window-builder test: the H30/H60 rows equal the expected t − stride·j, are all ≤ origin and on the same day; eligibility matches the audit counts exactly (164,362 / 159,414 / 150,758 / 150,618).
2. A feature-store test: precomputed features equal on-the-fly extraction for random samples; zero-filled columns are exact zeros; z-scoring uses fit-day statistics only.
3. A trainer-integration test: the D0 configuration reproduces the E4-B-fit data path (same frames and K-history) on common samples. A 1-epoch smoke run per arm checks shapes and parameter counts (identical across D0–D6).
4. An evaluation test: identical sample IDs across arms per population; the fallback population is assembled correctly (arm where eligible, E4-B-fit otherwise).
5. A real-data audit re-run, asserting the frozen geometry hash and eligibility hash at run time.

## 9. Compute budget (estimates from measured E4-B-fit runs on the RTX 5000 Ada, two concurrent lanes)
| Item | Hardware | Estimate |
|---|---|---|
| Feature precomputation (3 windows × 164k samples) | CPU | ≈ 1.3 h single-process (≈ 20–30 min with 4 workers) |
| Track A stage 1: 5 arms × 3 seeds = 15 runs | GPU | ≈ 0.9 h per run on the common population ⇒ **≈ 7 h** wall-clock with 2 lanes |
| Track A stage 2 (conditional): 2 arms × 3 seeds = 6 runs | GPU | ≈ **3 h** |
| Track B: G3c, G5 (and G3′) | CPU | < 0.5 h |
| Analysis and report | CPU | < 0.5 h |
| **Total** | | **≈ 7–10 h GPU + ≈ 2 h CPU** |

Per-forecast inference adds about 10 ms of feature extraction (single-thread CPU) to the model's 2.8 ms (batch 1). That is negligible for 1-min operation.

## 10. Items requiring approval before freezing
1. The 16-frame / variable-stride design for H30/H60 (vs dense 1-min frames, which would change capacity and compute).
2. The common population as primary, with E4-B-fit fallback for the secondary full-population evaluation.
3. Stage-2 conditionality (P/M split only if stage 1 passes).
4. The non-inferiority margins δ_O = 0.02 kW and δ_S = 0.015 kW.
5. The G5 motion-gate feature set (origin-frame radial inflow, inner-ring |ΔI|, shift magnitude).
