# Phase 6: event-time / attenuation-severity architecture (DESIGN + DRAFT PRE-REGISTRATION)

**Status: DRAFT, 2026-10-20. Not frozen.** It will be committed as a frozen pre-registration only after review, and before any Phase 6 training run.

The official 20-day test set is not used. Gate 2 (NOT MET) and E4-CAL2 (adaptive conformal retained) are preserved as-is.

## 1. Research question
Does an explicit decomposition of the 15-minute PV change into **when** a ramp arrives and **how severe** it is improve the forecast distribution? The target is cloud-induced ramps, where every model so far is weakest:
- ramp-stratum CRPS is 2.42 kW for E4-B (persistence: 4.67);
- ramp 90% coverage is about 0.64, and calibration does not fix it (Gate 2, E4-CAL2).

## 2. Architecture (implemented: `src/models.py::EventMixtureHead`, `src/events.py`, `src/train.py --head event`)
Shared backbone: the E4-B CNN-GRU, with K-history inputs, average pooling and 64-d state h.
- **Arrival head:** P(c | h) over c ∈ {(0,5], (5,10], (10,15] min, no event}.
- **Severity head:** for each class c, a 2-component Gaussian mixture over ΔK₁₅ = K(t+15) − K(t).
- **Predictive distribution:** p(ΔK₁₅) = Σ_c P(c) Σ_m w_cm N(μ_cm, σ_cm); K(t+15) = K(t) + ΔK₁₅; kW = K · max(P_clear, 1 kW).
- **Quantiles:** taken from the mixture CDF by bisection, at the same 9 levels as E4. They are scored with the same CRPS function and calibrated with the same E4-CAL2 rolling conformal.
- **Loss (supervised):** CE(arrival) + NLL(ΔK₁₅ | true class) + λ · marginal NLL, with λ = 1.
- **Initialisation:** training climatology (class log-frequencies, per-class ΔK quantiles and spread), as for the quantile head.
- **Labels (`src/events.py`), labels only and never inputs:**
  - K at t+1…t+15 (nearest record within ±30 s; fold anchor);
  - τ = 90th percentile of |ΔK₁₅| on the fold's **training** samples, the Gate 2 ramp threshold;
  - causal first crossing of |K(t+h) − K(t)| ≥ τ.
  - Measured on dev data: valid for ≥ 99.3% of samples; events are 15–16% of training samples (6–7% / 5% / 4% across the three bins). Fold-1 validation has 24% events.
- **Timing caveat:** the benchmark appears to pair images about 1 minute after PV (data finding). Arrival labels therefore carry about ±1 min uncertainty, which motivates 5-min bins.

## 3. Arms: one design that separates three effects
A smoke test showed that the persistence-residual parameterisation (predicting ΔK instead of K) has a large effect on its own. Every arm is therefore matched on it except the E4-B reference.

| Arm | Head | Target | Event supervision | Head params |
|---|---|---|---|---|
| E4-B-fit (existing, 3 seeds) | non-crossing quantiles | K(t+15) | – | 585 |
| **E6-QR** | capacity-matched MLP quantiles (64→24→9) | ΔK₁₅ (residual) | – | ≈ 1,785 |
| **E6-U** | EventMixtureHead | ΔK₁₅ | no (marginal NLL only) | 1,820 |
| **E6-S** | EventMixtureHead | ΔK₁₅ | **yes** | 1,820 |

**Contrasts:**
- **E6-QR vs E4-B:** the effect of persistence-residual forecasting. This is the separate follow-up requested earlier, reported as its own result.
- **E6-S vs E6-QR (primary):** the event-time/severity decomposition vs direct residual quantiles at matched capacity.
- **E6-S vs E6-U:** the value of explicit arrival/severity supervision, with architecture and parameter count identical.

**Common to all arms (identical to E4-B-fit):**
- `--calib-holdout` (the same calibration days, so E4-CAL2 rolling conformal applies to every arm);
- K inputs, avg pool, AdamW lr 1e-4 with cosine decay over 15 epochs, clip 1.0, EMA 0.999, batch 128, patience 5;
- checkpoint selection on fold-validation CRPS (kW);
- seeds 42/123/456; 5 blocked folds; identical evaluation samples (asserted).

**Budget:** 9 new runs, roughly 5–6 h on the RTX 5000 Ada with two concurrent lanes.

## 4. Endpoints
- **Primary:** pooled **ramp-stratum CRPS (kW)**, with paired day-block bootstrap CIs (stratified sunny/cloudy) per seed.
- **Secondary:**
  - overall, sunny and cloudy CRPS;
  - 50/80/90% coverage, width and interval score after E4-CAL2 rolling conformal (W chosen on calibration days), overall and ramp;
  - per-fold results, with fold 1 and its Sep–Oct 2017 period reported separately.
- **Event quality:**
  - Brier score and log loss of the 4-class arrival forecast (E6-S, E6-U) vs the training climatology;
  - Brier score and reliability of P(|ΔK₁₅| ≥ τ) for all arms (E4-B and E6-QR via the quantile-implied probability), which is the roadmap's "event classifier derived from the direct baseline" comparator;
  - arrival-bin accuracy among true events.
- **Cost:** parameters, latency and training time.

## 5. Gate 3 decision rule (proposed)
Keep the event-time/severity decomposition iff, in **at least 2 of 3 seeds**, all of the following hold:
1. ramp CRPS(E6-S) − ramp CRPS(E6-QR) < 0, with the day-bootstrap CI below 0;
2. ramp CRPS(E6-S) − ramp CRPS(E6-U) < 0, with CI below 0;
3. overall CRPS(E6-S) ≤ overall CRPS(E6-QR) + 0.01 kW (non-inferiority);
4. in addition, the seed-mean ramp CRPS of E6-S is lower than E6-QR's on ≥ 3/5 folds.

If (1) holds but (2) does not, the gain is attributed to the mixture form rather than to supervision, and that is reported as such. The E6-QR vs E4-B result is reported regardless, as the persistence-residual experiment.

## 6. Stop conditions (asserted)
- Any model input derived from records after the origin. Future records are used only for labels.
- Event threshold or initialisation statistics computed from non-training samples.
- Evaluation samples differ across arms.
- Overlap between fit, calibration and validation days.
- Crossing quantiles or non-finite outputs.

## 7. Separate, not part of Phase 6
- **Sep–Oct 2017 audit:** the period behind the fold-1 coverage collapse. Check image statistics, PV level and clear-sky residuals for an acquisition or regime change; the October 2017 wildfire smoke is a hypothesis only.
- **The suspected May 2019 camera shift** (Phase 8).

## 8. Implementation status
- **Done and committed:**
  - label builder;
  - EventMixtureHead (with climatology init, marginal quantiles and P(|ΔK| ≥ τ));
  - capacity-matched and residual quantile heads;
  - trainer integration (`--head event`, `--event-unsupervised`, `--head quantile_mlp --residual`).
- **Tests:** 39 unit tests pass.
- **Smoke runs:** all four arm types run end to end on a fold-1 subset (2 epochs, 40 days; numbers not meaningful).
- **Remaining before launch:**
  - review and freeze this document;
  - a Phase 6 analysis script (reusing `gate2_report`/`cal2` components and adding event metrics).
