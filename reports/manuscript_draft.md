# Manuscript draft (working): probabilistic sky-image PV forecasting under ramps and temporal shift

*Working draft, 2026-10-21.*
- **What it contains:** only findings already established on development folds, with 3 training seeds and paired day-level uncertainty. Claims that depend on E8 (Track A) are marked **[PROVISIONAL]**.
- **Test set:** the official 20-day SKIPP'D test set is sealed. **No test-set result exists yet**, and all numbers below are development-fold (blocked cross-validation) results.

---

## Abstract (provisional; test-set sentences pending)
We study 15-minute-ahead probabilistic forecasting of rooftop PV power from ground-based sky images, on the Stanford SKIPP'D benchmark, using a blocked, pre-registered development protocol.

1. **Clear-sky normalisation:** expressing inputs and quantile outputs in the clear-sky index K does not improve point accuracy, but it **reduces CRPS by 0.21 kW (22%)** relative to the same model in kW.
2. **Persistence-residual forecasting with a mixture head** reduces overall CRPS by a further 19% and is, to our knowledge, the first model in this setting to beat smart persistence on clear-sky days. Its cost is a replicated degradation on cloud-induced ramps.
3. **An explicitly supervised event-time/attenuation-severity decomposition** learns informative arrival probabilities but **does not improve ramp CRPS**. The binding constraint is ramp *detection* from the image history.
4. **A gate driven by forecast-time event probability and expert disagreement** routes between a ramp-robust and a calm-condition expert, closing about two-thirds of the ramp gap while giving the best overall CRPS (0.585 kW, 33% below smart persistence).
5. **Temporal shift:** split-conformal calibration fails under the shift we identify (a 2.5–3%/yr decline in the PV/clear-sky ratio), while causal rolling-window conformal restores nominal marginal coverage. Ramp-conditional coverage remains about 0.6–0.66 at 90% for all methods.

We also report three data findings for SKIPP'D users:
- a ±30 s sample-matching protocol that recovers the official sample counts;
- an apparent 1-minute image/PV pairing offset;
- a camera change on 2019-05-22 that shifts both colour statistics and the optical centre.

## 1. Data and protocol
- **Benchmark:** SKIPP'D (Stanford, 2017–2019; 64×64 sky images and 1-min PV from a ≈ 30 kW array). There are 349,372 development and 14,003 test paired records, matching the official counts exactly.
- **Forecast samples:**
  - **Construction:** 16 frames (t−15…t) and the PV target at t+15, built with the official construction rules.
  - **Timestamp matching:** each frame is matched to the nearest record within ±30 s. Exact matching on the public 1-min benchmark loses 19% of samples, because the per-minute timestamps drift in their seconds. The official samples were built from an unreleased ~10 s image stream.
  - **Counts:** 164,362 development samples (2-min cadence) and 13,266 test samples (1-min cadence).
  - **Validation:** under this protocol, smart persistence on the test split is within 0.013 kW of the official values, and a PyTorch port of SUNSET is within 0.02 kW of the published RMSE (3.046 vs 3.029 kW).
- **Development folds:** 5 contiguous day blocks with 1-day purges. A parameter-free day sky label reproduces the official sunny/cloudy labels on 20/20 test days.
- **Clear-sky anchor:** Ineichen + Hay-Davies POA + IAM, scaled by an annual harmonic fitted on training days only. The timestamp audit gives a 0-minute alignment between PV and clear sky.
- **Integrity:** an 8-point real-data audit covers leakage, ordering, units, determinism and related checks. Every gate was pre-registered and committed before its results existed, and post-hoc analyses are labelled as such.

## 2. Methods (summary)
- **Backbone:** a per-frame CNN, average-pooled, feeding a GRU over 16 frames.
- **Training recipe:**
  - AdamW at lr 1e−4 with cosine decay, gradient clipping, and an EMA of the weights for validation and checkpoints;
  - the CNN runs in bf16 while the GRU and heads stay fp32.
  - An earlier constant-lr recipe was unstable epoch to epoch; this is documented.
- **Heads:**
  - point (MSE);
  - non-crossing 9-level quantile head, initialised at the training climatology;
  - event-time/severity mixture: 4 arrival classes × 2-component Gaussian severity over ΔK₁₅.
- **Scoring:** CRPS of the piecewise-linear quantile function, which reduces exactly to |error| for point forecasts, so persistence and deterministic models are scored on the same scale.
- **Calibration:**
  - split-conformal CQR, fitted on held-out training-period calibration days;
  - causal rolling-window conformal and ACI, which update only from outcomes observed by issue time.
- **Gating:**
  - quantile averaging between a direct-K expert and a residual-mixture expert;
  - a logistic gate on forecast-time features, fitted by exact blended-CRPS minimisation on calibration days only.

## 3. Results (development folds; mean ± SD over 3 seeds; paired day-bootstrap 95% CIs)
**Point forecasting (Gate 1):** clear-sky normalisation with K inputs and K targets is indistinguishable from kW.
- Pooled RMSE is 2.059 ± 0.025 for K against 2.091 ± 0.020 for kW; the cloudy-day difference CI includes 0 in 3/3 seeds.
- Target-only normalisation is clearly worse (+0.255 kW).

**Probabilistic representation (Gate 2):**
- **CRPS:** 0.733 ± 0.006 kW in K, against 0.946 ± 0.014 in kW. The paired difference is −0.212 ± 0.007, with CIs below 0 in every stratum and seed.
- **Calibration:** both representations are over-confident (90% coverage 0.87 for K).

**Calibration under temporal shift (E4-CAL2):**
- **Fixed CQR** restores pooled 80/90% coverage but leaves fold-1 90% coverage at 0.71.
- **Causal rolling-window conformal** restores 0.90 on fold 1 and 0.50/0.79/0.89 pooled (50/80/90%), and lowers the 90% interval score in 3/3 seeds.
- **Cause of the fold-1 failure:** a one-sided under-prediction in Sep–Oct 2017 (41% of outcomes above Q.95 in October), explained by an inter-annual decline in clear-day K (October: 1.008 → 0.971 → 0.948 for 2017–2019). Wildfire smoke is not supported as the driver.

**Residual target and mixture form (Phase 6 controls):**
- **Overall CRPS:** 0.751 (direct K, calibration hold-out) → 0.646 (residual quantiles) → 0.610 (residual mixture).
- **Sunny-day CRPS:** 0.305 → 0.149 → 0.123, against 0.223 for persistence.
- **Ramp CRPS:** 2.448 → 2.771 → 2.93, i.e. worse.

**Event-time/severity decomposition (Gate 3, not retained):**
- **Arrival probabilities improve:** 4-class Brier 0.202 against 0.277 for climatology and 0.238 for unsupervised.
- **Severity given the true arrival class improves:** MAE 0.29 against 0.32.
- **Ramp CRPS does not improve:** it is worse than residual quantiles in 3/3 seeds, and the supervision effect changes sign across seeds.

**Gated mixture of experts (E7, E7b):**
- **Gating vs constant blending:** gating beats it in 3/3 seeds.
- **Best overall:** the combined-feature gate, CRPS 0.585 kW (−22% vs direct K, −33% vs persistence), with a rolling-conformal 90% interval score of 5.06.
- **Best ramp-protecting gate:** expert disagreement plus event probability, ramp CRPS 2.545.
- **Routing gap:** no gate reaches ramp non-inferiority to the direct-K expert. The remaining gap is about 0.09–0.12 kW.

**Ramp-conditional coverage:** about 0.61–0.66 at nominal 90% for every model and calibration method.

## 4. Data findings
1. **Sample construction:** the ±30 s sample construction recovers the official sample counts and persistence values.
2. **Pairing offset:** image changes correlate best with PV changes 1 minute earlier (two independent proxies).
3. **PV/clear-sky decline:** about 2.5–3% per year.
4. **Camera change on 2019-05-22:**
   - **Colour:** the midday R/B ratio goes from 1.30 to 2.29 overnight.
   - **Geometry:** a label-free fisheye calibration (equidistant lens, mirrored, about 0.7 px error) shows the optical centre moving about 2.5 px at the change.

## 5. Longer history and solar-relative motion (E8; established on development folds)
**Setup:** matched-capacity direct-K experts (16 frames at a 1/2/4-min stride), evaluated on the common population that is eligible for all windows.

**Results:**
- **Longer history:** 30 and 60 min improve ramp CRPS by 0.03–0.05 and 0.08–0.10 kW over a matched 15-min baseline (3/3 seeds), with small calm-condition costs within the pre-registered margins (2/3 seeds).
- **Hand-crafted solar-relative motion features** improve ramps further (up to −0.19 kW together with 60-min history), but cost 0.03–0.05 kW overall. They fail the margins and add no routing value as gate inputs.
- **Ramp detection quality** (implied event probability) does not change across arms. The gains are distributional.
- **Best gate:** a disagreement gate pairing the 60-min direct expert with the residual mixture expert (G3′) improves ramp CRPS by 0.07–0.09 kW over the 15-min pairing at equal overall CRPS (0.596; 3/3 seeds). It comes within about 0.03 kW of the 15-min direct expert's ramp CRPS.

**Caveat:** the common population excludes early-morning, ramp-rich samples (29.8% ramps). Deployment uses the 15-min models there.

## 6. Limitations
- A single site and 64×64 images.
- Development-fold results only, until the sealed test evaluation.
- Ramp labels depend on a clear-sky anchor without a temperature model.
- The 1-min image/PV offset is kept for comparability.
- Split-conformal guarantees are marginal and approximate under temporal dependence.

## 7. Pending before submission
1. ~~Finish E8 (Track A) and write the Gate 4 report.~~ Done (`reports/gate4_report.md`).
2. **Freeze the final forecaster(s) and protocol, then a single sealed test-set evaluation**, pre-registered.
   - Candidate: G3′ + causal rolling conformal, with a 15-min fallback where a 60-min history is unavailable.
   - Comparators: persistence, the SUNSET port, E4-B-fit, G4.
3. Figures: reliability, weekly fold-1 coverage, the ramp trade-off, gate weights, and a camera geometry and colour figure.
4. Related-work section and citations (not yet drafted).
