# E9 Stage S (screen) report: seed 42, 5 folds, common population

2026-10-22.
- **Pre-registration:** `reports/e9_prereg.md` (99d68da).
- **Runs:** commit 3c2c476 (clean tree).
- **Analysis:** `src/e9_report.py screen` → `reports/e9_screen.json`.
- **Scope:**
  - 150,618 common-population validation samples, identical across arms (asserted).
  - Official test set sealed.
  - No stop condition triggered.
- **Status:** Stage S only. **Stage C (seeds 123/456) has not been launched; it awaits review.**

## Factor isolation (answering the attribution question)
| Arm | Architecture | Loss | Differs from D2 in |
|---|---|---|---|
| **A0 = E8 D2 (reference; not retrained)** | CNN-GRU, 92,873 params | **unweighted** pinball | – |
| L1 | same CNN-GRU as D2 | **input-weighted** pinball (α = 2) | objective only |
| A1–A5 | new architectures | **unweighted** pinball | architecture only |

The loss weighting was applied **only** to L1, and the D2 reference is the original unweighted model. Each arm changes exactly one factor, so an A-arm result supports an **architecture** claim and L1 supports an **objective** claim. No combined (architecture + L1) claim is possible in E9.

## Pooled CRPS (kW) and paired difference vs D2 (95% day-bootstrap CI), seed 42
| Arm | Overall | Sunny | Cloudy | Ramp | Overall − D2 | Sunny − D2 | Cloudy − D2 | **Ramp − D2** |
|---|---|---|---|---|---|---|---|---|
| A0 = D2 | 0.802 | 0.349 | 1.130 | 2.839 | – | – | – | – |
| L1 weighted loss | 0.804 | 0.343 | 1.136 | 2.833 | +0.001 [−0.008, +0.011] | −0.005 [−0.017, +0.006] | +0.006 [−0.007, +0.021] | −0.006 [−0.027, +0.014] |
| A1 wide CNN-GRU | 0.811 | 0.354 | 1.142 | **2.725** | +0.009 [−0.000, +0.019] | +0.005 [−0.007, +0.016] | +0.012 [−0.002, +0.027] | **−0.115 [−0.136, −0.091]** |
| A2 3D-CNN + GRU | **0.737** | 0.295 | 1.056 | 2.773 | **−0.066 [−0.075, −0.056]** | **−0.054 [−0.065, −0.044]** | **−0.074 [−0.090, −0.059]** | **−0.066 [−0.099, −0.033]** |
| A3 ConvGRU | **0.680** | **0.226** | **1.008** | 3.098 | −0.122 [−0.139, −0.106] | −0.123 [−0.145, −0.101] | −0.121 [−0.145, −0.100] | **+0.258 [+0.218, +0.300]** |
| A4 transformer | 0.702 | 0.258 | 1.023 | 2.819 | −0.100 [−0.113, −0.088] | −0.091 [−0.106, −0.077] | −0.107 [−0.126, −0.089] | −0.020 [−0.050, +0.010] |
| A5 frame-diff CNN-GRU | 0.743 | 0.311 | 1.055 | 2.819 | −0.060 [−0.071, −0.048] | −0.038 [−0.052, −0.024] | −0.075 [−0.093, −0.057] | −0.020 [−0.053, +0.014] |

Event-occurrence quality (implied P(|ΔK₁₅| ≥ τ), AUPRC / Brier):

| Arm | AUPRC | Brier |
|---|---|---|
| A0 = D2 | 0.518 | 0.0555 |
| L1 | 0.516 | 0.0561 |
| A1 | 0.518 | 0.0568 |
| A2 | 0.515 | 0.0552 |
| A3 | 0.494 | 0.0548 |
| A4 | **0.534** | **0.0530** |
| A5 | 0.531 | 0.0531 |

## Per fold (seed 42): overall / sunny / cloudy / ramp CRPS (kW)
| Fold | A0 = D2 | L1 | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|---|---|
| 0 | 0.813 / 0.517 / 0.990 / 2.143 | 0.789 / 0.456 / 0.988 / 2.116 | 0.798 / 0.448 / 1.007 / **2.069** | 0.730 / 0.391 / 0.933 / 2.105 | **0.647 / 0.293 / 0.860** / 2.467 | 0.738 / 0.398 / 0.942 / 2.148 | 0.706 / 0.369 / 0.908 / 2.152 |
| 1 | 1.361 / 0.493 / 1.680 / 3.855 | 1.339 / 0.454 / 1.665 / 3.851 | 1.406 / 0.641 / 1.687 / **3.680** | 1.245 / 0.456 / 1.536 / 3.746 | **1.138 / 0.224 / 1.474** / 4.294 | 1.190 / 0.327 / 1.507 / 3.923 | 1.241 / 0.437 / 1.536 / 3.816 |
| 2 | 0.672 / 0.250 / 1.053 / 3.031 | 0.719 / 0.303 / 1.096 / 3.109 | 0.672 / 0.260 / 1.045 / 2.930 | 0.592 / 0.219 / 0.929 / 2.928 | 0.630 / 0.272 / 0.954 / 3.063 | **0.566 / 0.208 / 0.891** / 2.917 | 0.597 / 0.231 / 0.927 / **2.912** |
| 3 | 0.796 / 0.255 / 1.060 / 2.795 | 0.813 / 0.293 / 1.066 / 2.769 | 0.820 / 0.255 / 1.096 / 2.709 | 0.761 / 0.225 / 1.023 / **2.691** | 0.712 / **0.174** / 0.975 / 2.972 | **0.715** / 0.197 / **0.968** / 2.715 | 0.788 / 0.254 / 1.049 / 2.816 |
| 4 | 0.543 / 0.327 / 0.871 / 2.220 | 0.523 / 0.297 / 0.866 / 2.199 | 0.547 / 0.333 / 0.869 / **2.076** | 0.514 / 0.283 / 0.863 / 2.276 | **0.415 / 0.174 / 0.780** / 2.426 | 0.451 / 0.219 / 0.802 / 2.204 | 0.540 / 0.329 / 0.858 / 2.207 |

**Fold consistency of the ramp direction vs D2:**
- **A1:** lower ramp CRPS on 5/5 folds.
- **A2:** 4/5 folds (fold 4 higher).
- **A3:** higher on 5/5 folds.

## Non-inferiority and selection (pre-registered, seed-42 point estimates)
**Eligible ⇔** overall ≤ D2 + 0.02 **and** sunny ≤ D2 + 0.015 **and** ramp < D2.

| Arm | Overall margin | Sunny margin | Ramp < D2 | Eligible |
|---|---|---|---|---|
| L1 | ✓ (+0.001) | ✓ (−0.005) | ✓ (2.833 < 2.839) | yes |
| A1 | ✓ (+0.009) | ✓ (+0.005) | ✓ | yes |
| A2 | ✓ | ✓ | ✓ | yes |
| A3 | ✓ | ✓ | ✗ (+0.258) | **no** |
| A4 | ✓ | ✓ | ✓ | yes |
| A5 | ✓ | ✓ | ✓ | yes |

**Frozen selection rule:** among eligible arms, the **two with the lowest seed-42 ramp CRPS**; ties go to fewer parameters. The result is **A1 (2.725) and A2 (2.773)**, selected for Stage C. A4 and A5 tie at 2.819 and are not selected.

## Capacity, training time, latency
| Arm | Parameters | Epochs (5 folds) | Train time (min, 2 runs sharing GPU) | Batch-1 latency (ms) |
|---|---|---|---|---|
| A0 = D2 | 92,873 | 49 | 71 | 1.99 |
| L1 | 92,873 | 53 | 97 | 1.60 |
| A1 | 362,889 | 41 | 125 | 1.60 |
| A2 | 213,545 | 56 | 93 | 2.10 |
| A3 | 89,097 | 75 | 140 | 5.59 |
| A4 | 138,505 | 44 | 65 | 1.72 |
| A5 | 93,305 | 57 | 95 | 1.75 |

- **Latency** was measured sequentially on an idle GPU, 200 iterations after warm-up. Identical architectures differ by about 0.4 ms (A0 vs L1), so differences below about 0.5 ms are measurement noise.
- **Training times** come from shared-GPU runs and are indicative only.

## Interpretation (screen-level, single seed; not confirmatory)
1. **The objective is not the limitation.** Input-weighted loss (L1) changes nothing (all CIs span 0).
2. **Representation choices change the ramp/calm trade-off itself:**
   - **A2 (temporal convolution)** improves overall, sunny, cloudy **and** ramp CRPS together. It is the first model in the project that does not trade calm accuracy for ramp accuracy.
   - **A1 (more capacity)** improves ramps only (5/5 folds), with calm conditions unchanged.
   - **A3 (spatial recurrence)** gives the best calm-condition CRPS of any direct model (overall 0.680, sunny 0.226, close to persistence's sunny level), but much worse ramps.
   - **A4 (transformer)** improves calm conditions strongly and is ramp-neutral, with the best event AUPRC (0.534).
3. **Caution:**
   - These are seed-42 results used for selection. Effects of this size need the pre-registered Stage C replication before any claim.
   - Seed-to-seed SD of overall CRPS was about 0.02 kW for D2 across seeds.
