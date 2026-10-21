# Gate 4 report: E8 longer history and solar-relative motion (Track A + G3′)

2026-10-22.
- **Pre-registration:** `reports/e8_prereg.md` (frozen 826c34e / fd8a2ce); readiness checklist approved before launch.
- **Code:** run code at commit 681fecd (clean tree, recorded in every run's config); analysis `src/e8_tracka_report.py`, `src/e8_g3prime.py`.
- **Results:** `reports/e8_trackA_results.json`, `reports/e8_g3prime_results.json`. Track B: `reports/e8_trackB_report.md`.
- **Scope:**
  - 15 Track-A runs (D0–D4 × seeds 42/123/456 × 5 blocked folds), each trained and evaluated on the **common population** (150,618 validation samples; identical across arms, asserted).
  - Frozen geometry, eligibility and feature hashes verified at every run start.
  - E7/E7b/Track B results unchanged; official 20-day test set sealed.
  - No stop condition triggered.

## 1. Track A verdict (pre-registered rule vs the matched 15-min baseline D0)
| Arm | History | Solar + motion | Ramp CI < 0 | Overall ≤ +0.02 | Sunny ≤ +0.015 | Seeds passing | **Verdict** |
|---|---|---|---|---|---|---|---|
| D1 | 30 min | – | 3/3 | 3/3 | 2/3 | 2/3 | **improves ramp anticipation** |
| D2 | 60 min | – | 3/3 | 2/3 | 2/3 | 2/3 | **improves ramp anticipation** |
| D3 | 15 min | ✓ | 3/3 | 0/3 | 0/3 | 0/3 | not retained |
| D4 | 60 min | ✓ | 3/3 | 0/3 | 0/3 | 0/3 | not retained |

**Stage 2 (P-vs-M split) is not triggered.** It was conditional on D3 or D4 passing.

## 2. Common population: raw CRPS, kW (mean ± SD over 3 seeds)
| Stratum | D0 (15 min) | D1 (30 min) | D2 (60 min) | D3 (15 min + PM) | D4 (60 min + PM) |
|---|---|---|---|---|---|
| Overall | **0.767 ± 0.021** | 0.775 | 0.781 | 0.807 | 0.815 |
| Sunny | **0.322** | 0.325 | 0.329 | 0.344 | 0.346 |
| Cloudy | **1.088** | 1.100 | 1.108 | 1.142 | 1.154 |
| **Ramp** | 2.933 ± 0.009 | 2.894 | 2.850 | 2.824 | **2.768 ± 0.039** |
| Non-ramp | **0.572** | 0.583 | 0.595 | 0.625 | 0.639 |

Paired contrasts vs D0 (seeds 42 / 123 / 456; every ramp CI is below 0):

| Contrast | Ramp | Overall | Non-ramp |
|---|---|---|---|
| History 30 min (D1 − D0) | −0.035 / −0.048 / −0.034 | +0.019 / +0.006 / −0.002 | +0.024 / +0.011 / +0.001 |
| History 60 min (D2 − D0) | −0.095 / −0.077 / −0.078 | +0.023 / +0.016 / +0.004 | +0.034 / +0.024 / +0.012 |
| Features at 15 min (D3 − D0) | −0.130 / −0.093 / −0.105 | +0.032 / +0.043 / +0.046 | +0.047 / +0.055 / +0.060 |
| Joint (D4 − D0) | −0.193 / −0.130 / −0.174 | +0.042 / +0.052 / +0.050 | +0.063 / +0.068 / +0.070 |
| Features given 60 min (D4 − D2) | −0.098 / −0.052 / −0.096 | +0.019 / +0.036 / +0.046 | +0.029 / +0.044 / +0.059 |

**Interpretation:**
- Longer history and solar/motion features each **improve ramp CRPS, monotonically and replicably**: the 60-min history plus features gives −0.13 to −0.19 kW.
- Each also **costs accuracy in calm conditions**. History alone has a small cost that passes the margins in 2/3 seeds; the features have a cost that fails them. This is the same ramp/calm trade-off seen in Gate 3 and E7, now *within* the direct-K expert family.
- **Ramp detection does not sharpen.** The event-occurrence probability implied by each arm has unchanged quality: Brier 0.0545–0.057 and AUPRC 0.50–0.52 for all arms (D4 slightly worse). The ramp gains come from **better-shaped uncertainty during ramps, not from earlier or sharper detection.**

## 3. Fallback population (deployment-realistic; all 164,362 validation samples) and sunrise ramps
Each arm is used where its window is eligible and E4-B-fit elsewhere. Arm share: D0/D3 100%, D1 97%, D2/D4 91.7%.

| Stratum | D0 | D1 | D2 | D3 | D4 |
|---|---|---|---|---|---|
| Overall | **0.755** | 0.761 | 0.764 | 0.792 | 0.795 |
| Ramp | 2.450 | 2.406 | 2.364 | 2.344 | **2.303** |
| **Sunrise ramps** (ramps outside the common population) | 0.976 | 0.918 | **0.884** | **0.880** | **0.884** |

- **Sunrise-ramp scores mostly measure the fallback model.** For D2 and D4, sunrise ramps are mostly scored with E4-B-fit, which was trained on all samples including early mornings.
- **D0 does worse on them (0.976)** because it was trained on the common population, which excludes early mornings. **Training-population coverage matters for sunrise ramps.**
- **Sunrise ramps are much smaller in kW** (CRPS about 0.9 vs about 2.9 for mid-day ramps).

## 4. G3′: pre-registered gating with the best passing direct expert (CPU, frozen experts)
The best passing arm is **D2** (lowest seed-mean ramp CRPS among D1 and D2). G3′ uses the G3 recipe (features g, d; calibration days only) with experts D2 and E6-S. It is compared with G3c (experts E4-B-fit and E6-S).

| Stratum | G3c | **G3′** | D2 | E4-B-fit (Track B) |
|---|---|---|---|---|
| Overall | 0.599 | **0.596** | 0.781 | 0.766 |
| Sunny | 0.121 | 0.122 | 0.329 | 0.310 |
| Cloudy | 0.944 | **0.939** | 1.108 | 1.096 |
| **Ramp** | 3.076 | **2.995** | 2.850 | 2.961 |
| Non-ramp | **0.375** | 0.379 | 0.595 | 0.568 |

- **Rule (G3′ vs G3c): passes 3/3 seeds.**
  - Ramp: −0.094 [−0.118, −0.071] / −0.080 [−0.099, −0.060] / −0.070 [−0.090, −0.047].
  - Overall: −0.003 / −0.003 / −0.003.
- **G3′ vs D2:** much better overall (−0.16 to −0.21) and on non-ramp samples, and worse on ramps (+0.13 to +0.17).
- **Descriptive (not a pre-registered test here):** on the common population, G3′'s ramp CRPS is within about 0.034 kW of E4-B-fit (2.995 vs 2.961), while its overall CRPS is 0.596. G3c is 0.115 kW behind. The E7 routing gap is therefore largely, but not fully, closed (E7 threshold 0.02).

**Calibrated intervals (causal rolling conformal), 90% coverage / width kW / interval score:**

| Stratum | G3c | **G3′** | D0 | D2 | D4 |
|---|---|---|---|---|---|
| Overall | 0.895 / 3.82 / 5.17 | 0.896 / **3.73 / 5.14** | 0.885 / 5.87 / 7.41 | 0.882 / 5.90 / 7.44 | 0.884 / 5.68 / 7.32 |
| Ramp | 0.669 / 10.40 / 21.47 | 0.666 / 10.08 / 21.53 | 0.629 / 9.87 / 22.05 | 0.634 / 9.84 / 21.55 | **0.672** / 10.06 / **21.20** |
| Non-ramp | 0.916 / 3.23 / 3.70 | 0.917 / **3.16 / 3.66** | 0.908 / 5.51 / 6.09 | 0.905 / 5.54 / 6.17 | 0.903 / 5.28 / 6.06 |

- 50/80% coverage after rolling conformal is 0.50–0.51 / 0.79–0.80 for all methods.
- **Ramp-conditioned 90% coverage stays at 0.63–0.67 for every method.**

## 5. Cost
- **Training:** about 80 min per run (two concurrent on an RTX 5000 Ada); 15 runs took about 9.5 h of wall-clock.
- **Inference model cost:** identical across arms (identical parameter counts); about 2.8 ms at batch 1.
- **Feature extraction (D3/D4):** precomputed per image record (349k records in 55 s on 32 CPU workers); online, about 10 ms per forecast single-threaded.
- **History arms (D1/D2):** no extra features, but they need the last 30/60 min of images. Early-morning forecasts fall back to the 15-min model.

## 6. Conclusions
1. **Longer image history is a genuine, replicated ramp improvement** that passes the pre-registered margins: 30 min (D1) and 60 min (D2). Most of its value is realised **in combination with the calm-condition expert**: G3′ (D2 + E6-S, disagreement gate) is the best gated forecaster, improving ramps by 0.07–0.09 kW over G3c at equal overall CRPS.
2. **Hand-crafted solar-relative motion features improve ramps further but cost too much elsewhere.** D3 and D4 are not retained under the pre-registered margins, and Track B showed they add nothing as gate inputs.
3. **No model sharpens ramp detection itself.** The gains are distributional (wider and better-shaped uncertainty when it matters). Ramp-conditioned coverage (≈ 0.63–0.67) remains the main unsolved issue.
4. **Recommended forecaster of record for the sealed test evaluation**, to be pre-registered before unsealing:
   - **Candidate:** **G3′ + causal rolling conformal**, i.e. experts D2 (60-min direct K) and E6-S (residual mixture), with a disagreement gate.
   - **Early-morning fallback:** E4-B-fit / G3 where a 60-min history is not available.
   - **Reference comparators:** persistence, the SUNSET port, E4-B-fit, and G4 (best overall in E7b).
5. **Trade-off disclosure:** if ramp accuracy alone is the priority, D4 has the best ramp CRPS (2.768) and ramp interval score but loses about 0.05 kW overall. That is a priority choice outside these rules.
