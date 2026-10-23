# Disagreement-gated sky-image experts for probabilistic 15-minute PV forecasting: a pre-registered evaluation on SKIPP'D

**Abdul Samad Laghari**

*Draft v1, 2026-10-23. Target venue: Solar Energy / Applied Energy.*

---

## Abstract

Sky-image deep-learning models for minute-scale photovoltaic (PV) forecasting are usually evaluated loosely. Test sets are reused for model selection, uncertainty is reported without calibration, and cross-paper comparisons mix sample constructions. We present a probabilistic 15-minute-ahead PV forecaster developed and evaluated under a pre-registered protocol on the Stanford SKIPP'D benchmark, with the official 20-day test set opened exactly once.

**Method.**
- Forecasts are made in clear-sky-index units.
- A soft logistic gate, fitted by CRPS minimisation on held-out days and driven by forecast-time *disagreement between experts* and an implied event probability, blends a calm-condition event-mixture expert with a 60-minute 3D-CNN ramp expert.
- 15 ensemble members are combined by level-wise quantile averaging.
- Intervals are calibrated by causal rolling conformal prediction.

**Results on the official SKIPP'D test (13,266 forecasts).**
- The forecaster lowers CRPS against the strongest single expert from 1.117 to 1.006 kW: a paired test-day bootstrap difference of −0.111 kW [−0.149, −0.075], confirming the sole primary hypothesis.
- Under a closely matched sample construction its RMSE is 2.949 kW, against 3.03 kW for the published SUNSET baseline.
- Ramp non-inferiority and the coverage target were not met.

**Results on SkyGPT's external test set** (2,582 samples on 5 cloudy days after the end of the development data; one-shot):
- The frozen model reaches CRPS 2.455 kW and a Winkler score of 18.16.
- The published values on the same samples are 2.63 / 21.46 (PV-MM-Diffusion), 2.81 / 26.70 (SkyGPT) and 3.31 / 56.95 (SUNSET).

**Limits we report.**
- No model, ours included, anticipates changes: forecast increments correlate with observed increments 15 minutes earlier, not with concurrent ones.
- Ramp-conditional coverage stays near 0.65–0.75 at nominal 90%.
- Daily-reset conformal calibration fails when its initial scores are far in time; continuous online calibration does not.

**Benchmark issues.** We also document four previously unreported SKIPP'D issues and an apparent 2.2%/yr decline in the PV/clear-sky ratio.

---

## 1. Introduction

Minute-scale PV variability from passing clouds complicates grid balancing, storage dispatch and ramp-rate compliance. Ground-based sky imagers observe the clouds that cause it. Deep networks trained on sky-image sequences have become the standard approach for 1–30 min horizons (Sun et al., 2019; Paletta et al., 2021; Nie et al., 2023). Three gaps motivate this work.

1. **Lag and regime dependence.** Deep sky-image models behave like "very smart persistence" and lag ramp events (Paletta et al., 2021a, b; SolarBench, Nie et al., 2026). Physically motivated imager forecasts beat persistence only under high variability (Schmidt et al., 2016). Yet most models use one network for all regimes, and prior regime-specific designs route by observed sky class (Nie et al., 2020; McCandless et al., 2016) or blend with fixed per-horizon weights (Straub et al., 2024).
2. **Uncalibrated uncertainty.** Probabilistic sky-image forecasters such as SkyGPT (Nie et al., 2024) and PV-MM-Diffusion (Huang et al., 2026) build predictive distributions from generative samples without a calibration layer. Conformal methods for PV exist (Renkema et al., 2024; Suresh, 2025) but have not been applied to sky-image models under temporal shift.
3. **Evaluation practice.** Repeated test-set reuse inflates reported performance (Dwork et al., 2015; Recht et al., 2019). Leakage across temporally correlated samples is common (Kapoor & Narayanan, 2023). Published SKIPP'D numbers rest on different sample constructions and metrics.

**Contributions.**
1. A **disagreement-gated mixture** of a calm-condition expert and a 60-min 3D-CNN ramp expert, fitted by CRPS on held-out days, with a timestamp-only fallback when long history is unavailable (Section 4).
2. A **pre-registered, sealed, one-shot evaluation** on the official SKIPP'D test (Section 5–6.2), plus a pre-registered one-shot evaluation on SkyGPT's external test samples. There we report the lowest published CRPS and Winkler score we are aware of (Section 6.3).
3. A **calibration study** showing that online conformal calibration restores marginal coverage only when allowed to adapt continuously, and that regime-aware calibration adds nothing on top of a disagreement-gated mixture (Section 6.4).
4. A **lag analysis** showing that the gains come from better forecast level and spread, not anticipation (Section 6.5).
5. **Four undocumented SKIPP'D data issues** and an apparent PV/clear-sky ratio decline (Section 3.3).

## 2. Related work

**Sky-image forecasting.**
- Stanford SUNSET (Sun et al., 2019; Venugopal et al., 2019) established CNN forecasting from 16-frame sky-image stacks plus PV history. SKIPP'D (Nie et al., 2023) released the benchmark and test protocol.
- Paletta et al. (2021a) benchmarked architectures and diagnosed temporal lag. ECLIPSE (Paletta et al., 2022) predicts future segmented skies to reduce it.
- Video-prediction approaches generate future skies before mapping them to PV: SkyGPT (Nie et al., 2024); PV-MM-Diffusion (Huang et al., 2026), which denoises sky images and PV jointly.
- Ramp-focused work includes ramp-rate control from sky images (Wen et al., 2021) and ramp-stratified probabilistic evaluation at national scale (Lanzilao & Meyer, 2026).

**Combining forecasters.**
- Regime-specific models classify the sky or cloud regime and route to specialised models (McCandless et al., 2016; Nie et al., 2020).
- Hybrid nowcasts blend all-sky-imager, satellite and persistence forecasts. Straub et al. (2024) do so with lead-time-dependent linear-regression weights that are independent of conditions.
- Thorey et al. (2018) learn CRPS-optimal weights for PV ensembles, unconditionally.
- Mixture-of-experts gates in forecasting typically use inputs (Chakraborty et al., 2025) or within-expert variance (Aviv et al., 2025).
- We are not aware of a probabilistic sky-image gate driven by *between-expert disagreement*.
- Within each expert we use quantile averaging (Vincentization). It is sharper than probability averaging (Lichtendahl et al., 2013) and robust to location-shifted members (Busetti, 2017).

**Calibration.**
- CRPS and reliability are standard for solar probabilistic verification (Gneiting et al., 2007; Lauret et al., 2019).
- Split and quantile conformal prediction (Romano et al., 2019) assume exchangeability.
- Online methods adapt under shift: ACI (Gibbs & Candès, 2021), DtACI (Gibbs & Candès, 2024), AgACI (Zaffran et al., 2022), NexCP (Barber et al., 2023) and conformal PID (Angelopoulos et al., 2023).

**Clear-sky normalisation.**
- The clear-sky index is non-stationary under any clear-sky model (Yang, 2020; Voyant et al., 2025).
- Three-year clear-sky-normalised performance trends carry large uncertainty (Marion, 2021). Typical PV degradation is 0.5–0.8%/yr (Jordan & Kurtz, 2013).

## 3. Data

### 3.1 SKIPP'D
- **Site and sensors:** Stanford University. 64×64 RGB fisheye images at 1-min cadence; 1-min PV output of a 30.1 kW-DC rooftop array (tilt 22.5°, azimuth 195°).
- **Splits:** 349,372 development records over 497 days (2017-03-09 to 2019-10-26). The official test set has 14,003 records over 20 fixed days (10 sunny, 10 cloudy).
- **Forecast samples:** 16 frames *t*−15…*t* min and PV at *t* + 15 min.

### 3.2 Sample construction
- The public benchmark is thinned to about one record per minute, and its second-of-minute phase drifts. Exact timestamp matching therefore loses 19% of samples relative to the official construction, which used an unreleased ~10 s stream.
- We match each frame and target to the nearest record within ±30 s. This gives 164,362 development samples (2-min cadence) and 13,266 test samples (1-min cadence).
- Section 6.3 shows that the resulting protocol gap is ≤ 0.03 kW in RMSE and MAE for SUNSET.

### 3.3 Data findings
1. **Timestamp drift:** the second-of-minute phase of the public 1-min records changes about 3,300 times.
2. **Camera change on 2019-05-22.** The midday R/B ratio jumps from 1.30 to 2.29 overnight (Figure F5). A label-free fisheye calibration shows a ~2.5 px shift of the optical centre. The dataset documentation records only a later (2022-04-29) replacement.
3. **Image/PV offset:** image changes correlate best with PV changes about 1 minute earlier, consistent with forward-averaged PV.
4. **Non-chronological test file:** the official test HDF5 is stored in two time-ordered blocks (2019 days first).
5. **Apparent PV/clear-sky decline:** on clear days, PV relative to a fixed physical clear-sky model declines by **2.2 ± 0.3%/yr** after seasonal adjustment (196 days, 2017–2019; the naive linear fit gives 2.8%/yr; Figure F4). This exceeds typical module degradation. Given the uncertainty of short clear-sky-normalised trends (Marion, 2021), we attribute it jointly to degradation, soiling, clear-sky (aerosol) model error and data artefacts, not to degradation alone.

## 4. Method

### 4.1 Clear-sky index
`P_clear(t)` uses Ineichen–Perez clear-sky irradiance (pvlib; Holmgren et al., 2018), Hay–Davies transposition and an incidence-angle loss, scaled by an annual harmonic fitted on training days only. All models forecast `K = PV / max(P_clear, 1 kW)`; quantiles are mapped back to kW with `P_clear(t + 15)`.

### 4.2 Experts
All experts encode the frames per frame (CNN or 3D-CNN), then with a GRU over time, with the K history as an extra input. Training uses AdamW (lr 1e−4, cosine schedule), an EMA of the weights, at most 15 epochs with early stopping, and blocked 5-fold day splits × 3 seeds.

- **E4-B-fit:** 15-min history; 9-level non-crossing quantile head initialised at climatology.
- **E6-S:** 15-min history; event-mixture head (4 arrival-time classes × 2-component Gaussian severity over ΔK15) supervised with frozen event thresholds. It is sharp in calm conditions (sunny-day CRPS 0.073 kW on test, against 0.269 for persistence).
- **A2:** 60-min history (16 frames at 4-min stride); spatio-temporal 3D-CNN encoder. It improved ramp CRPS by about 0.10 kW over a 2D-CNN encoder on the same history (2/3 seeds).

### 4.3 Disagreement gate
- The forecast is `q = w·q_ramp + (1 − w)·q_calm`, with `w = σ(β·z)`.
- `z` holds standardised forecast-time features: the event probability implied by the calm expert's quantiles at the frozen ramp threshold, and the absolute difference between the experts' mean forecasts.
- β is fitted by exact minimisation of the blended CRPS on calibration-hold-out days that never enter expert training.
- **G3″** pairs A2 with E6-S.
- When no valid 60-min history exists (6.9% of test forecasts, mostly after sunrise), a timestamp rule routes to **G3**, the same gate with the 15-min expert E4-B-fit. The routed forecaster is **G3″-F**.

### 4.4 Ensemble and calibration
- The 15 members (3 seeds × 5 folds) are combined by equal-weight, level-wise quantile averaging.
- Intervals come from causal rolling-window conformal prediction. A score enters the window only after its target time (15-min maturation). The window is reset each day and initialised with development out-of-fold scores nearest in time. W was chosen on development data.

## 5. Evaluation protocol
- **Pre-registration:** every gate's hypotheses, metrics and decision rules were committed to version control before results existed. Analyses outside them are labelled post hoc.
- **Development:** blocked 5-fold day splits with 1-day purges. Out-of-fold assembly gives genuinely held-out development forecasts.
- **Sealed test:** before unsealing we froze the protocol, checkpoint hashes (140 files), ramp threshold and calibration state. We then ran a development dry run and a leakage audit.
  - Two execution attempts aborted before any test label was read: a clean-tree guard, and a non-chronological file. Both are documented.
  - The evaluation then ran once.
- **Metrics:**
  - CRPS of the piecewise-linear quantile function; it reduces to the absolute error for point forecasts.
  - RMSE and MAE of the median; RMSE skill vs smart persistence.
  - Interval coverage, width and Winkler (interval) score.
  - Paired day-block bootstrap: 10,000 resamples, stratified sunny/cloudy.
- **Hypotheses (official test):**
  - **H1 (sole primary):** overall CRPS of G3″-F below E4-B-fit's.
  - **H2:** below persistence.
  - **H3:** ramp non-inferiority, margin 0.02 kW.
  - **H4:** RMSE skill.
  - **H5:** calibrated 90% coverage within ±3 pp.

## 6. Results

### 6.1 Development path (out-of-fold, 3 seeds)
Each step below was a pre-registered gate. CRPS in kW.

| Step | Finding |
|---|---|
| Clear-sky index vs kW (quantile heads) | CRPS 0.733 vs 0.946 (−22%) |
| Residual target + event mixture (E6-S) | Overall 0.751 → 0.610; sunny 0.305 → 0.123 (persistence 0.223); ramp 2.45 → 2.93 (worse) |
| Explicit event supervision | Arrival Brier 0.202 vs 0.277 climatology; ramp CRPS not improved |
| Gating calm and ramp experts | Best overall CRPS 0.585 (−33% vs persistence) |
| 60-min history | Ramp CRPS −0.08 to −0.10, at a small calm-condition cost |
| 3D-CNN on 60-min history (A2) | Ramp 2.754 vs 2.850 (2D-CNN); confirmed 2/3 seeds |
| Ramp detection (all arms) | Event AUPRC 0.50–0.53, unchanged |
| G3″-F, out-of-fold | CRPS 0.573 vs 0.730 for E4-B-fit |

### 6.2 Official SKIPP'D test

| Hypothesis | Estimate [95% CI] | Verdict |
|---|---|---|
| H1: CRPS G3″-F − E4-B-fit | **−0.111 [−0.149, −0.075]** | **Confirmed** |
| H2: vs persistence, overall / cloudy | −0.701 [−0.823, −0.582] / −1.208 [−1.454, −0.971] | Confirmed |
| H3: ramp non-inferiority (0.02) | −0.117 [−0.244, +0.031] | Not established |
| H4: RMSE skill vs persistence | 18.6% (SUNSET port 16.2%) | – |
| H5: calibrated 90% coverage | 0.935 | Not met (over-coverage) |

**CRPS by stratum (kW):**

| Model | Overall | Sunny | Cloudy | Ramp | Sunrise ramp |
|---|---|---|---|---|---|
| **G3″-F** | **1.006** | 0.074 | **1.939** | **3.510** | 0.808 |
| E4-B-fit | 1.117 | 0.218 | 2.017 | 3.627 | 0.778 |
| G4 (15-min gate) | 1.024 | 0.073 | 1.976 | 3.782 | 0.802 |
| E6-S | 1.090 | 0.073 | 2.108 | 4.281 | 0.900 |
| SUNSET port | 1.719 | 0.475 | 2.965 | 4.815 | 1.070 |
| Smart persistence | 1.707 | 0.269 | 3.147 | 7.131 | 1.610 |

- G3″-F beats E4-B-fit on 19 of 20 days, and persistence and SUNSET on all 20 (Figure F2).
- Calibrated 90% intervals are 24% narrower than E4-B-fit's (6.26 vs 8.20 kW) at higher coverage.
- Ramp coverage is 0.746, with directional misses: on up-ramps 30% of outcomes fall above the interval; on down-ramps 21% fall below it.

### 6.3 Comparison with published results

**SUNSET on SKIPP'D.**
- On the exact-timing subset (the official matching rule, n = 11,086), our SUNSET port reproduces the published numbers to within 0.03 kW.
- On that subset G3″-F reaches **RMSE 2.949 kW, MAE 1.373 kW and forecast skill 18.1%**, against 3.03 / 1.71 / 16.4% published.
- The advantage is concentrated on sunny days (RMSE 0.156 vs 0.61). On cloudy days RMSE is similar (4.21 vs 4.27).

**SkyGPT's external test** (5 cloudy days, Nov–Dec 2019, 2,582 samples).
- These days lie after the end of our development data and were never seen.
- G3″-F's frozen routing yields G3 on every sample, because no 60-min history exists in the file.
- Population checks: our persistence scores 3.672 (published 3.67), and our SUNSET port scores 3.271 / 56.87 (published 3.31 / 56.95).

| Model (same 2,582 samples) | CRPS (kW) | Winkler 90% |
|---|---|---|
| **G3″-F (= G3 here; ours, one-shot)** | **2.455** | **18.16** |
| Real-future-image → U-Net oracle (Huang et al., 2026) | 2.52 | 30.72 |
| PV-MM-Diffusion (Huang et al., 2026) | 2.63 | 21.46 |
| SkyGPT (Nie et al., 2024) | 2.81 | 26.70 |
| SUNSET (Nie et al., 2024) | 3.31 | 56.95 |
| Smart persistence | 3.67 | – |

Our quantile-interpolation CRPS is 1.2% *higher* than the empirical-ensemble CRPS used by SkyGPT, measured on our SUNSET port's members, so the comparison does not favour us. Per-sample predictions of the published models are not available, so these are comparisons of point estimates, not paired tests. On these winter cloudy days the calibrated 90% coverage is only 0.77 (Section 6.4).

### 6.4 Calibration under temporal shift (development data only)
We compared seven causal calibration methods on out-of-fold streams with leave-one-fold-out hyperparameters (pre-registered):
- no calibration;
- rolling conformal with daily reset (the frozen method);
- continuous rolling conformal;
- ACI;
- DtACI-style aggregation;
- two Mondrian variants, binned by forecast spread or by event probability.

**Findings:**
1. **The daily-reset scheme is fragile.** When its initial scores come from distant periods, 90% coverage falls to 0.61 in one fold, and its interval score is the worst of all methods, even below no calibration. Continuous rolling and ACI keep 0.88–0.92 in every fold. This explains the contrast between the SKIPP'D test (test days embedded in the development period; 0.935) and the external winter days (initial scores weeks old; 0.77).
2. **Regime-aware calibration did not improve ramp coverage for G3″-F:** +0.001 [−0.013, +0.015]. The pre-registered hypothesis was not confirmed. It did for the single expert E4-B-fit: +0.074 [+0.057, +0.091]. The gate appears to capture the regime signal already.
3. **G3″-F is well calibrated without any calibration step** on development data (0.903 at 90%). No method in this family raises ramp-conditional coverage above about 0.71.

### 6.5 Do the forecasts anticipate changes?
On cloudy days we cross-correlated first differences of each model's median forecast, aligned at target time, with the observed PV differences.

| Model | Lag-0 correlation | Lag-15 correlation |
|---|---|---|
| G3″-F | ≈ 0.00 | 0.87 |
| E6-S | ≈ 0.00 | 0.97 |
| E4-B-fit | ≈ 0.00 | 0.58 |
| SUNSET port | ≈ 0.00 | 0.24 |
| Smart persistence | ≈ 0.00 | 0.98 |

- For every model, including ours, the correlation is ≈ 0 at lag 0 and peaks at a lag of about 15 minutes.
- The pattern holds on the external days.
- **The gains reported above therefore come from a better forecast level and spread, not from anticipating cloud arrivals.** This is the "very smart persistence" behaviour of Paletta et al. (2021a), now quantified on SKIPP'D.

## 7. Discussion
- **Why the mixture works.** The calm expert is sharp and nearly persistence-like. The 60-min ramp expert carries information about recent cloud motion. Their disagreement is a cheap forecast-time signal of an unsettled sky, and the CRPS-fitted gate turns it into a weighting that improves both the level and the width of the distribution. Because the regime information is already used in the forecast, a regime-aware calibration step adds nothing further (Section 6.4).
- **What does not work.** No configuration improves ramp *detection* (AUPRC ≈ 0.5), and no model anticipates changes. Ramp-conditional intervals under-cover in a directional way: up-ramps are missed from above and down-ramps from below. These are the open problems. Promising directions include optical-flow or video-prediction inputs (Paletta et al., 2022; Huang et al., 2026) and ramp-aware losses, evaluated on a new sealed set.
- **Practical cost.** The full 15-member, five-model evaluation ran at about 36 ms per forecast on one GPU. That is about three orders of magnitude cheaper than diffusion-based generation (45 s per forecast for 10 samples; Huang et al., 2026).
- **Calibration guidance.** Online conformal calibration should adapt continuously. A daily reset needs recent, similar initial scores, which a deployment that starts after a gap will not have.

## 8. Limitations
- A single site, low-resolution (64×64) images, and three years of data.
- The SkyGPT external test covers only 5 cloudy days. The 60-min branch could not be evaluated there.
- Comparisons with published models rest on point estimates.
- The ±30 s sample construction differs from the original benchmark (gap ≤ 0.03 kW for SUNSET).
- Calibration findings come from development data. A changed calibration method requires a new sealed evaluation.
- A second-site validation is in progress.

## 9. Conclusion
A disagreement-gated mixture of a calm-condition expert and a 60-min 3D-CNN ramp expert, evaluated under a pre-registered one-shot protocol, gives the best probabilistic scores we know of on SkyGPT's test set. It also clearly beats the strongest single expert on the official SKIPP'D test. The forecasts still do not anticipate cloud-driven changes, and ramp-conditional uncertainty remains poorly calibrated. We release code, frozen protocols and all raw outputs.

## Data and code availability
- **Code, pre-registrations, reports and raw test outputs:** <https://github.com/abdulsamad2/solar-architecher> (MIT).
- **SKIPP'D:** <https://purl.stanford.edu/dj417rh1007> (CC BY 4.0).
- **SkyGPT test data:** the authors' public folder (<https://github.com/yuhao-nie/SkyGPT>).

## References (to be formatted)
- Angelopoulos, Candès, Tibshirani (2023). Conformal PID control for time series prediction. NeurIPS.
- Aviv, Goldberger, Shavit (2025). MoGU. arXiv:2510.07459.
- Barber, Candès, Ramdas, Tibshirani (2023). Conformal prediction beyond exchangeability. Ann. Statist.
- Busetti (2017). Quantile aggregation of density forecasts. Oxf. Bull. Econ. Stat. 79(4).
- Chakraborty et al. (2025). MoWE. arXiv:2509.09052.
- Dwork et al. (2015). Generalization in adaptive data analysis and holdout reuse. NeurIPS.
- Gibbs & Candès (2021). Adaptive conformal inference under distribution shift. NeurIPS.
- Gibbs & Candès (2024). Conformal inference for online prediction with arbitrary distribution shifts. JMLR 25.
- Gneiting, Balabdaoui, Raftery (2007). Probabilistic forecasts, calibration and sharpness. JRSS-B 69.
- Holmgren, Hansen, Mikofski (2018). pvlib python. JOSS 3(29).
- Huang, Shao, Ke, Gao, Mazzoni (2026). PV-MM-diffusion. Applied Energy 413:127816.
- Jordan & Kurtz (2013). Photovoltaic degradation rates. Prog. Photovolt. 21.
- Kapoor & Narayanan (2023). Leakage and the reproducibility crisis. Patterns 4(9).
- Lanzilao & Meyer (2026). National-scale solar ramp events. arXiv:2603.26596.
- Lauret, David, Pinson (2019). Verification of solar irradiance probabilistic forecasts. Solar Energy 194.
- Lichtendahl, Grushka-Cockayne, Winkler (2013). Is it better to average probabilities or quantiles? Manage. Sci. 59(7).
- Marion (2021). Clear-sky and satellite irradiance for PV degradation. Solar Energy 223.
- McCandless, Haupt, Young (2016). Regime-dependent ANN for solar irradiance. Renewable Energy 89.
- Nie, Sun, Chen, Orsini, Brandt (2020). Sky-condition-specific sub-models vs end-to-end. JRSE 12:046101.
- Nie, Li, Scott, Sun, Venugopal, Brandt (2023). SKIPP'D. Solar Energy 255.
- Nie, Zelikman, Scott, Paletta, Brandt (2024). SkyGPT. Adv. Appl. Energy 14:100172.
- Nie et al. (2026). SolarBench. arXiv:2609.06187.
- Paletta, Arbod, Lasenby (2021a). Benchmarking of deep learning irradiance forecasting models from sky images. Solar Energy 224.
- Paletta, Hu, Arbod, Lasenby (2022). ECLIPSE. Applied Energy 326.
- Recht, Roelofs, Schmidt, Shankar (2019). Do ImageNet classifiers generalize to ImageNet? ICML.
- Renkema, Visser, AlSkaif (2024). Conformal prediction for probabilistic PV forecasts. Solar Energy Adv. 4.
- Romano, Patterson, Candès (2019). Conformalized quantile regression. NeurIPS.
- Schmidt, Kalisch, Lorenz, Heinemann (2016). Sky-imager-based irradiance forecasts. ACP 16.
- Straub, Herzberg, Dittmann, Lorenz (2024). Blending ASI, persistence and satellite nowcasts. Solar Energy 269:112319.
- Sun, Venugopal, Brandt (2019). Short-term solar power forecast with deep learning. Solar Energy 188.
- Suresh (2025). PatchTST with adaptive conformal inference for PV. Energies 18(18).
- Thorey, Chaussin, Mallet (2018). Ensemble PV forecast with online CRPS learning. IJF 34(4).
- Venugopal, Sun, Brandt (2019). Short-term solar PV forecasting using computer vision. JRSE 11.
- Voyant et al. (2025). Clear-sky model in short-term forecasting. Solar Energy 289.
- Wen et al. (2021). Sky-image forecasting for PV ramp-rate control. IEEE TII 17(2).
- Yang (2020). Choice of clear-sky model in solar forecasting. JRSE.
- Zaffran et al. (2022). Adaptive conformal predictions for time series. ICML.
