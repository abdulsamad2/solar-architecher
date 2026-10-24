# Development history and technical details

This page holds detail that used to sit in the main README. Each step has its own pre-registration and report in [`reports/`](../reports/); those files are authoritative. Unless stated otherwise, every number here comes from **development folds**: blocked 5-fold CV, 3 seeds, paired day-bootstrap CIs on the Stanford SKIPP'D development period.

## 1. Experts and training recipe

| Name | Input | Architecture | Output | Role |
|---|---|---|---|---|
| **E4-B-fit** | last 15 min (16 frames, 1-min stride) + K history | per-frame CNN → GRU | 9 quantiles of K | direct expert; strongest single model at Stanford; G3 fallback partner |
| **E6-S** | last 15 min + K history | CNN → GRU + event-mixture head (4 arrival-time classes × 2-component Gaussian severity over ΔK) | mixture → quantiles | residual expert; very sharp in calm and clear conditions |
| **A2** | last 60 min (16 frames, 4-min stride) + K history | 3D-CNN (spatio-temporal) → GRU | 9 quantiles of K | ramp expert; sees cloud motion |
| D4 (descriptive only) | last 60 min + hand-crafted sun-position and motion features | CNN → GRU | 9 quantiles | helps ramps but costs overall; not used |

**Training recipe (all experts)**
- **Optimiser:** AdamW, lr 1e−4 with cosine decay, weight decay 1e−4, gradient clipping 1.0, weight EMA 0.999.
- **Schedule:** batch 128, at most 15 epochs, early stopping with patience 5 on the pinball loss.
- **Precision:** CNN in bf16; GRU and heads in fp32.
- **Ensemble:** 5 blocked folds × 3 seeds (42, 123, 456) = **15 members** per expert.
- **Calibration hold-out:** every 5th block of 10 training days is held out of training and used only to fit the gates.

**Gates** ([`src/gate7b.py`](../src/gate7b.py)):
- **Form:** w = σ(b₀ + β·z). The gate blends the two experts' quantile functions level by level.
- **Features (G3/G3″):** z = [logit g, log d].
  - g is the event probability implied by the E6-S quantiles.
  - d is the absolute difference between the two experts' mean forecasts (their disagreement).
- **Fitting:** exact blended-CRPS minimisation plus an L2 penalty, on calibration days only. λ is chosen on a chronological calibration split.

**Calibration** ([`src/cal2.py`](../src/cal2.py), [`src/final_eval.py`](../src/final_eval.py), [`src/gatton_eval.py`](../src/gatton_eval.py)):
- Rolling-window conformal correction using only outcomes that have matured (15-minute delay).
- **Stanford test:** daily reset, initialised from development out-of-fold scores.
- **Gatton test:** continuous, with no daily reset. This follows Phase C ([`reports/phaseC_report.md`](../reports/phaseC_report.md)).
- No finite-sample coverage guarantee is claimed under temporal dependence.

## 2. Experiment sequence (Stanford development)

| Step | Question | Key result | Decision |
|---|---|---|---|
| E0 / E1 (Gate 0) | Can we reproduce the benchmark? | Smart persistence matches; SUNSET port RMSE 3.046 vs 3.029 kW published | Benchmark reproduced |
| E2 / E3 / E3b (Gate 1) | Does clear-sky normalisation help point forecasts? | RMSE 2.059 (K) vs 2.091 (kW); CI includes 0 | No benefit for point forecasts |
| E4 (Gate 2) | Quantiles in K or in kW? | CRPS 0.733 vs 0.946 (−22%) | K representation adopted |
| E4-CAL2 | Calibration under temporal shift | Static conformal: 90% coverage 0.71 on one fold; rolling conformal gives 0.89–0.90 | Causal rolling conformal adopted |
| Phase 6 (E5/E6) | Residual target and mixture head | CRPS 0.751 → 0.646 → 0.610; sunny 0.305 → 0.123; ramps worsen 2.45 → 2.93 | E6-S kept as the calm-condition expert |
| Gate 3 | Explicit event-time/severity supervision | Arrival Brier 0.202 vs 0.277 climatology; ramp CRPS not improved | Not retained |
| E7 / E7b | Gating the two experts | Best overall CRPS 0.585; best ramp gate 2.545 | Disagreement gate adopted |
| E8 (Gate 4) | Longer history; sun-relative motion features | 60-min history improves ramps by 0.08–0.10; motion features cost 0.03–0.05 overall | 60-min history kept; features dropped |
| E9 | Encoders on the 60-min history | 3D-CNN (A2) confirmed in 2 of 3 seeds, ramps about −0.10; other arms not confirmed | A2 adopted → G3″ |
| Dry run + rehearsal | Frozen pipeline on held-out development data | G3″-F 0.573 vs E4-B-fit 0.730; all checks pass | Approved for the sealed test |
| Gate 5 | One-shot official test | H1 confirmed | Final ([`reports/gate5_test_report.md`](../reports/gate5_test_report.md)) |

**Data findings**
- **Sample matching:** ±30 s matching recovers the official sample counts ([`reports/protocol_notes.md`](../reports/protocol_notes.md)).
- **Image/PV offset:** images change about 1 minute before PV, which suggests a 1-minute pairing offset.
- **Camera change on 2019-05-22:** the R/B colour ratio jumps from 1.30 to 2.29, and the optical centre shifts by about 2.5 px.
- **Apparent PV decline:** the PV/clear-sky ratio declines about 2.2 %/yr (season-adjusted, ±0.3). This is more than typical module degradation, so it is not attributed to degradation alone ([`reports/e4_cal2_report.md`](../reports/e4_cal2_report.md), [`reports/related_work.md`](../reports/related_work.md), Figure F4).

## 3. Stanford test: detailed strata (kW)
From [`reports/test_results.json`](../reports/test_results.json):

| Model | Overall | Sunny | Cloudy | Ramp | Non-ramp | Sunrise ramp | RMSE |
|---|---|---|---|---|---|---|---|
| G3″-F | 1.006 | 0.074 | 1.939 | 3.510 | 0.583 | 0.808 | 2.959 |
| E4-B-fit | 1.117 | 0.218 | 2.017 | 3.627 | 0.693 | 0.778 | 3.037 |
| G4 (15-min gate) | 1.024 | 0.073 | 1.976 | 3.782 | 0.558 | 0.802 | 3.033 |
| E6-S | 1.090 | 0.073 | 2.108 | 4.281 | 0.551 | 0.900 | 3.274 |
| SUNSET (our port) | 1.719 | 0.475 | 2.965 | 4.815 | 1.197 | 1.070 | 3.046 |
| Smart persistence | 1.707 | 0.269 | 3.147 | 7.131 | 0.791 | 1.610 | 3.636 |

- **Ramp definition:** a ramp is \|ΔK over 15 min\| ≥ 0.279. The threshold was frozen before testing.
- **Per day:** G3″-F beats E4-B-fit on 19 of 20 test days.
- **Calibrated 90% coverage:** overall 0.935, on ramps 0.746.

## 4. Context: published results (Phase A)
These figures come from published point estimates, not paired tests. The protocols differ in sample matching and in the CRPS estimator. They are context, not a claim of superiority. Full protocol and caveats: [`reports/phaseA_report.md`](../reports/phaseA_report.md); estimator check: [`reports/estimator_check.json`](../reports/estimator_check.json).

**SUNSET, SKIPP'D test.** On the exact-timing subset, our SUNSET port reproduces the published numbers to within 0.03 kW. G3″-F has RMSE 2.949 kW and forecast skill 18.1%. Published SUNSET has 3.03 kW and 16.4%.

**SkyGPT's external test set** (5 cloudy days, Nov–Dec 2019, 2,582 samples; frozen models run once):

| Model | CRPS (kW) | Winkler 90% |
|---|---|---|
| G3″-F, 15-min configuration (competitors' 30-sample estimator) | 2.49 | 22.0 |
| PV-MM-Diffusion (published) | 2.63 | 21.46 |
| SkyGPT (published) | 2.81 | 26.70 |
| SUNSET (published) | 3.31 | 56.95 |
| Smart persistence | 3.67 | – |

- **Winkler score:** comparable to PV-MM-Diffusion, not better.
- **Interval coverage:** on these days the calibrated 90% intervals under-cover (0.77).
- **Ramps:** ramp CRPS is not better than with E4-B-fit.

## 5. Physics-informed training objective (Phase F)

**Pre-registration and outputs**
- Pre-registration: [`reports/phaseF_physics_prereg.md`](../reports/phaseF_physics_prereg.md)
- Report: [`reports/phaseF_report.md`](../reports/phaseF_report.md)
- Code: [`src/physics_losses.py`](../src/physics_losses.py), [`src/physics_metrics.py`](../src/physics_metrics.py), `src/train.py --physics-config`

**What changed and what did not.** Physics terms are added to A2's training objective only. The architecture, gate, outputs, folds, seeds, ensemble, calibration and sealed protocols are all unchanged.

| Term | Definition | Role |
|---|---|---|
| L_quantile | mean ρ_q(K − K̂_q), with ρ_q(e) = max(q·e, (q − 1)·e) | primary objective |
| L_phys | mean \|P_true − K̂₅₀ · max(P_clear(t+15), 1)\| / 30 kW | physical consistency of the median |
| L_bound | mean ReLU(−K̂_q) + ReLU(K̂_q − 1.5) | soft plausibility; K > 1 allowed |
| L_ramp | mean w_i · \|ΔK_true − ΔK_pred\|, with w_i = 1 + α·r_i | ramp-weighted change error |
| L_direction (optional) | Σ r_i · ReLU(m − sign(ΔK_true) · ΔK_pred) / Σ r_i | wrong-direction penalty |

**Training and evaluation**
- **Total loss:** L_quantile + λ_phys·L_phys + λ_ramp·L_ramp + λ_bound·L_bound (+ λ_dir·L_direction).
- **Arms and search:** A0 (baseline), A1 (+phys), R (+ramp), A3 (+phys+ramp), A4 (+bounds), A5 (+direction). The weights were chosen by a staged search on development folds.
- **Evaluation:** development out-of-fold, with the G3″ gate refitted per arm.

**Result:**
- **CRPS:** every variant changes it by ≤ 0.6%. Physics consistency alone gives −0.14%; its CI excludes 0, but the effect is negligible in practice.
- **Winner rule:** no variant met the pre-registered rule.
- **Ramps:** small effects on ramp coverage (up to +0.8 percentage points) and ramp direction (82.2% → 83.5%).
