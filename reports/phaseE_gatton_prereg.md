# Phase E: second-site evaluation on Gatton (pre-registration, FROZEN before any Gatton PV value is used)

2026-10-23. **No Gatton PV value has been read.** Only the following were inspected:
- file names and image timestamps;
- the PV file's timestamp column;
- one development-day image (2020-01-24 12:00).

## 1. Data
- **Source:** "Gatton Sky Image & PV Generation 2020", University of Queensland (Xu & Ma; DOI 10.48610/bd7e108; UQ RDM dataset b051cba0, open access).
  - Package SHA-1 values verified: `data/2020.zip` 1e4d9d11…, `data/pv_2020.csv` d842953e….
  - Licence as stated on UQ eSpace; to be quoted in the paper.
- **Images:** 463,548 JPEG files, 2048×1536 RGB, every 10 s, 104 days (2020-01-23 to 2020-05-06), camera clock GMT+10.
- **PV:** `pv_2020.csv`, columns `DateTime` (format `%d/%m/%Y %I:%M:%S %p`, local GMT+10) and `Power(kW)`. 1-min cadence, 2020-01-01 to 2020-11-01. 9 duplicate timestamps (rule in §3). No PV-system metadata (array, tilt, azimuth, tracking) is provided.
- **Plant:** 3.275 MW peak (dataset description). Queensland, about −27.56°, 152.34° (SolarBench supplementary); no daylight saving.

## 2. Split: fixed now, by date only
- **Sealed test: the last 21 image days, 2020-04-16 to 2020-05-06 (inclusive).** These days' PV values are not read until the one-shot evaluation. The final evaluation is chronological and out of period, as recommended for non-stationary series.
- **Development: the 83 image days 2020-01-23 to 2020-04-15.** They are cut into 5 contiguous blocked folds with 1-day purges (`src/folds.blocked_folds`), each with a calibration hold-out (`src/folds.calibration_split`). Day labels come from the same parameter-free rule.
- **PV-only data outside the image period:**
  - 2020-01-01 to 2020-01-22 may be used, with development, for the clear-sky fit (§3).
  - **Nothing on or after 2020-04-16 is used for any fitting.**

## 3. Preprocessing: fixed rules; parameters fitted on development days only
1. **One frame per minute.** For each PV minute timestamp, take the image nearest in time within ±30 s (10-s files; ties go to the earlier image). This yields SKIPP'D-like 1-min paired records.
2. **Duplicate PV timestamps:** keep the first occurrence.
3. **Fisheye geometry.**
   - Estimate the circle centre and radius once, from the per-pixel mean brightness of 200 random noon images on development days (seed 20261023).
   - Crop the square [centre ± radius].
   - **Set pixels outside the circle to 0.** This removes the camera's text overlays; SKIPP'D backgrounds are dark naturally.
   - Resize to **64×64** (PIL `BOX` filter); keep RGB uint8.
   - The same frozen parameters are applied to every day, test days included. Image processing is input-only.
4. **Time alignment.**
   - Image and PV clocks are both nominally GMT+10.
   - On development days only, compute the correlation between first differences of mean image brightness inside the circle and PV first differences, at lags of −5…+5 min.
   - If the best lag is not 0, records are re-paired with that integer-minute shift. Either way, the lag is reported.
5. **Clear-sky model.**
   - Ineichen clear-sky irradiance at the site (pvlib).
   - The unknown geometry is chosen by a grid search on development-period records only (≤ 2020-04-15, including PV-only January days). Candidates: fixed tilt ∈ {0, 5, …, 40}° with azimuth ∈ {0 (north), ±15, ±30}°, and a horizontal single-axis tracker (N–S axis, backtracking off).
   - **Criterion:** RMSE between PV and the scaled POA envelope on development "sunny" days.
   - The scale is fitted with the existing `ClearSkyPV` procedure. A constant scale is used automatically when fewer than 10 months are present.
6. **Clear-sky index:**
   - `K = PV / max(P_clear, c_min)`, with c_min = 3.275 MW × (1 kW / 30.1 kW) ≈ **108.8 kW**. This is the SKIPP'D 1-kW floor scaled by plant capacity.
   - PV is used in kW, without filtering or clipping, except that **negative values are clipped to 0** (night-time inverter draw).
7. **Samples:**
   - The same construction as SKIPP'D (`src/skippd.official_forecast_samples`, ±30 s): 16 frames t−15…t and target t+15.
   - Development at a 2-min cadence; test at a 1-min cadence.
   - H60 windows and eligibility use the same rule (`src/e8_audit.history_rows`).
   - Origins only where P_clear(t) ≥ c_min (daytime).
8. **Ramp label:** τ_G = 90th percentile of \|K(t+15) − K(t)\| over development samples, frozen before testing.

## 4. Arms
| Arm | Description |
|---|---|
| **Z0** | **Zero-shot:** the frozen SKIPP'D G3″-F (15 members; frozen gates; G3 fallback), plus E4-B-fit and E6-S, applied without any retraining |
| Z1 (descriptive) | Z0 with the image flipped vertically (hemisphere ablation) |
| **F1** | **Fine-tuned:** each of the 15 members of E4-B-fit, E6-S and A2 is initialised from its SKIPP'D checkpoint and trained on Gatton development fit days with the frozen SKIPP'D recipe (AdamW, lr 1e−4, cosine schedule, EMA, ≤ 15 epochs, patience 5 on CRPS for the fold's validation days). G3 and G3″ gates are refitted on Gatton calibration days with the frozen gate-fitting procedure |
| F0 (secondary, only if GPU time allows) | The same experts trained from scratch on Gatton development data (data-efficiency comparison) |
| Baselines | Smart persistence with the §3.5 clear-sky model; persistence of PV |

- **Calibration:** the **continuous** causal rolling conformal method (Phase C "M2"; W chosen on Gatton development out-of-fold data by the frozen selection rule), with 15-min maturation and no daily reset. It is initialised from Gatton development out-of-fold scores.
- The 60-min branch is used where H60 windows exist; otherwise G3 is used (timestamp rule).

## 5. Hypotheses (sealed Gatton test, 21 days, 1-min cadence)
Each is tested with a paired day bootstrap: 10,000 resamples, stratified by the development-rule sunny/cloudy label, seed 20261022.
- **E1 (sole primary):** overall CRPS of fine-tuned G3″-F (F1) is lower than fine-tuned E4-B-fit (F1). Confirmed iff the 95% CI of the difference is entirely below 0. This replicates Gate 5's H1 at a second site.
- **E2:** F1 G3″-F CRPS is lower than smart persistence (CI < 0).
- **E3 (descriptive):** zero-shot (Z0) vs fine-tuned (F1) G3″-F CRPS; Z0 vs smart persistence.
- **E4 (descriptive):** calibrated 90% coverage overall and on ramps; target 0.90 ± 0.03 overall.
- **E5 (descriptive):** lag analysis (lag-0 and lag-15 correlation of forecast increments), as in `reports/lag_analysis.json`.

**Also reported:** ramp, sunny and cloudy strata, per-day CRPS, RMSE and MAE of the median, interval width and Winkler score.

## 6. One-shot rule
- All development-stage choices (preprocessing parameters, clear-sky geometry, fine-tuning, gates, W) are frozen and hash-inventoried before unsealing.
- The test PV is read by a guarded runner exactly once, from a clean committed tree. Raw outputs are committed before interpretation. Failures before labels are read may be fixed and documented; after that, only identical-code reruns are allowed.

## 7. Known limitations stated in advance
- About 3.5 months, a single season (summer to autumn).
- The PV array geometry is unknown and is estimated.
- MW-scale plant output is spatially smoothed compared with a 30-kW rooftop.
- Camera exposure differs strongly from SKIPP'D.
- The test period is late autumn, which may differ from development.
