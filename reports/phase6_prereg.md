# Phase 6 pre-registration (FROZEN): event-time / attenuation-severity decomposition

Frozen 2026-10-20 before any Phase 6 training run. It supersedes `phase6_design_DRAFT.md`, which is kept for history; the architecture is described there in full.
- **Preserved unchanged:** Gate 2 (**NOT MET**, `gate2_report.md`) and E4-CAL2 (`e4_cal2_report.md`).
- **Test set:** the official 20-day SKIPP'D test set stays sealed.

## 1. Arms (3 seeds × 5 blocked folds each = 9 runs)
| Arm | Command flags (beyond the common set) | Target | Head (params) |
|---|---|---|---|
| E6-QR | `--head quantile_mlp --residual` | ΔK₁₅ | MLP 64→24 → non-crossing 9 quantiles (1,785) |
| E6-U | `--head event --event-unsupervised` | ΔK₁₅ | EventMixtureHead (1,820) |
| E6-S | `--head event` | ΔK₁₅ | EventMixtureHead (1,820) |

**Common set:**
- `--target csi --input csi --pool avg --max-epochs 15 --patience 5 --calib-holdout --frozen-thresholds reports/phase6_frozen_thresholds.json`, with seeds 42, 123 and 456.
- AdamW lr 1e-4 with cosine decay, grad-norm clip 1.0, EMA 0.999, batch 128, CNN-only bf16.
- Checkpoint selection on fold-validation CRPS (kW); this convention is disclosed and identical for all arms and for E4-B-fit.

**Reference arm (existing, not retrained):** E4-B-fit, seeds 42/123/456 (`experiments/E4_calib/v1/csi`).

**E6-U vs E6-S:** identical architecture, mixture family (2 Gaussian components per class over ΔK₁₅, 4 classes), climatology initialisation, optimiser, budget and selection. They differ **only** in the loss:
- E6-U: marginal NLL;
- E6-S: CE(arrival) + NLL(ΔK₁₅ | true class) + 1.0 × marginal NLL.

**Observed in a pre-launch check:** the event head's raw gradient norms (p99 ≈ 117) are far larger than the quantile heads' (≈ 1), so the common clip at 1.0 binds strongly for E6-U/E6-S. This is accepted as part of the matched budget and not tuned.

## 2. Labels and thresholds (frozen)
- **K:** K = P / max(P_clear, 1 kW). P_clear is the fold's clear-sky anchor fitted on the run's fit days (calibration hold-out).
- **Exact first-crossing event definition:**
  - For origin t, take K(t+h) for h = 1…15 min, each from the record nearest t + h min within ±30 s.
  - The arrival minute is h* = min{h : |K(t+h) − K(t)| ≥ τ}.
  - Class 0 if h* ∈ [1, 5], class 1 if h* ∈ [6, 10], class 2 if h* ∈ [11, 15], class 3 (no event) if no h qualifies.
  - A missing minute cannot trigger a crossing. The label is **valid** only if every minute h ≤ h* (or all 15 if there is no event) is present. Invalid labels are excluded from the CE and conditional-NLL terms but kept in the marginal NLL.
  - **Severity** = ΔK₁₅ = K(t+15) − K(t), at the matched target record.
- **Training-label threshold τ:** the 90th percentile of |ΔK₁₅| over the fold's **fit-day** samples, using the fit-day anchor. Frozen values in `reports/phase6_frozen_thresholds.json`: fold 0–4 = 0.281472 / 0.262355 / 0.298268 / 0.272097 / 0.295155. Any mismatch at run time is a STOP.
- **Evaluation ramp stratum:** model-independent and identical to Gate 2 / E4-CAL2, i.e. |ΔK₁₅| ≥ τ_eval, with the full-training-day anchor and 90th percentile. τ_eval for fold 0–4 = 0.276943 / 0.259691 / 0.297665 / 0.270829 / 0.293260.
- **Bins:** 3 arrival bins + no event, fixed; never tuned on validation results. Arrival labels carry about ±1 min uncertainty (benchmark image/PV offset).

## 3. Endpoints
**Primary:** pooled **ramp-stratum CRPS (kW) of the raw, uncalibrated predictive distribution**. It is computed from the 9 quantile levels with `crps_quantile_interp` (the same scorer as E4).
- For E6-U/E6-S the quantiles come from the marginal mixture CDF.
- For E6-QR they are K(t) + ΔK quantiles.

**Secondary (raw distributions):** overall, sunny, cloudy and non-ramp CRPS; per-fold CRPS; fold-1 and Sep–Oct 2017 CRPS.

**Calibrated intervals (separate):** the E4-CAL2 causal rolling-window conformal procedure, exactly as pre-registered there:
- W chosen on calibration days only;
- updates only from outcomes observable at issue time.

Reported: 50/80/90% coverage, width and interval score, overall, per fold, sunny/cloudy and **ramp / non-ramp (ramp-conditioned coverage)**. The same is done for uncalibrated intervals.

**Arrival-probability quality (E6-U, E6-S):**
- 4-class Brier score and log loss vs the training-climatology forecast;
- per-class reliability;
- arrival-bin accuracy among true events.

**Event-occurrence comparator (all arms incl. E4-B-fit):** Brier score and reliability of P(|ΔK₁₅| ≥ τ_eval).
- E4-B-fit and E6-QR: implied by their quantiles (`implied_event_probability`).
- E6-U/E6-S: from the mixture.

**Severity diagnostics (true ramp samples):**
- MAE of the median ΔK₁₅;
- sign accuracy (attenuation vs enhancement) of the median;
- 80% central-interval coverage of ΔK₁₅;
- for E6-S, the MAE of the class-conditional mean severity given the true arrival class.

**Uncertainty:** paired day-block bootstrap 95% CIs (stratified sunny/cloudy) per seed, plus seed mean ± SD across seeds, reported separately.

**Cost:** head and total parameters, batch-1 latency, training epochs and wall time.

## 4. Gate 3 decision rule
**Non-inferiority margin δ = 0.02 kW on overall raw CRPS.** Justification:
- about 3× the seed-to-seed SD of overall CRPS observed for E4-B (0.006 kW);
- about 2.7% of E4-B's overall CRPS (0.733);
- about 10% of the replicated E4-B − E4-A representation gain (0.21);
- comparable to the cost of the calibration hold-out (+0.018).

It tolerates seed-level noise but not a loss of practical size.

The event-time/severity decomposition is **retained** iff, in **at least 2 of 3 seeds**:
1. ramp CRPS(E6-S) − ramp CRPS(E6-QR) < 0, with the 95% day-bootstrap CI entirely below 0;
2. ramp CRPS(E6-S) − ramp CRPS(E6-U) < 0, with the CI entirely below 0;
3. overall CRPS(E6-S) − overall CRPS(E6-QR) ≤ δ = 0.02 kW (point estimate);

and, on the seed-mean, E6-S has lower ramp CRPS than E6-QR on ≥ 3 of 5 folds.

**Attribution reported regardless of the verdict:**
- **Residual-target gain:** E6-QR vs E4-B-fit (the persistence-residual experiment).
- **Mixture-architecture gain:** E6-U vs E6-QR.
- **Event-supervision gain:** E6-S vs E6-U.
- If (1) holds but (2) does not, the gain is attributed to the mixture form, not to supervision.

## 5. Stop conditions (asserted)
- A model input derived from records after the origin. Future records are only label lookups; inputs come from the 16 history rows ≤ origin.
- An event threshold differs from its frozen value, or initialisation statistics come from non-fit samples.
- Fit, calibration and validation days overlap.
- Evaluation samples differ across arms, including E4-B-fit.
- Crossing quantiles, non-finite outputs, or inverted calibrated intervals.

## 6. In parallel, not affecting Phase 6
A CPU-only audit of the Sep–Oct 2017 coverage collapse. It is descriptive only and changes no Phase 6 training, selection or decision rule.
