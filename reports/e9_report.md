# E9 report: ramp-aware loss and architecture screen (Stages S, C, G)

2026-10-23.
- **Pre-registration:** `reports/e9_prereg.md` (99d68da). Screen report: `reports/e9_screen_report.md` (95c9cf6).
- **Results:** `reports/e9_screen.json`, `reports/e9_confirm.json`, `reports/e9_gate.json`.
- **Runs:**
  - Stage S at commit 3c2c476; Stage C at 95c9cf6. The commits in between changed only analysis and reporting code, not training code.
  - All runs on clean trees.
- **Scope:**
  - Common population (150,618 validation samples, identical across arms).
  - Official test set sealed. All earlier verdicts unchanged.
  - No stop condition triggered.
- **Factor isolation:** the reference A0 is the original **unweighted** E8 D2. L1 changes only the loss; A1–A5 change only the architecture (unweighted loss).

## Stage S (seed 42; selection), summary
- **L1 (input-weighted loss):** null on every stratum.
- **Selected under the frozen rule:** A1 (wide CNN-GRU) and A2 (3D-CNN + GRU).
- **Not selected:**
  - A3 (ConvGRU): best calm-condition CRPS but ramp +0.258, so ineligible;
  - A4 (transformer) and A5 (frame-diff): eligible but not in the top two on ramp CRPS.
- Details are in the screen report.

## Stage C: confirmation vs D2 (seeds 42 / 123 / 456)
| Arm | Seed | Overall − D2 | Sunny − D2 (≤ 0.015) | Cloudy − D2 | **Ramp − D2** [95% CI] | Pass |
|---|---|---|---|---|---|---|
| A1 wide | 42 | +0.009 | +0.005 | +0.012 | −0.115 [−0.136, −0.091] | yes |
| | 123 | **+0.034** | **+0.051** | +0.022 | −0.117 [−0.142, −0.093] | no (overall, sunny) |
| | 456 | **+0.041** | **+0.025** | +0.053 | −0.104 [−0.125, −0.084] | no (overall, sunny) |
| A2 3D-CNN | 42 | −0.066 | −0.054 | −0.074 | −0.066 [−0.099, −0.033] | yes |
| | 123 | +0.012 | **+0.036** | −0.005 | −0.108 [−0.149, −0.061] | no (sunny) |
| | 456 | −0.025 | −0.014 | −0.032 | −0.113 [−0.143, −0.084] | **yes** |

**Verdicts:**
- **A2 improves the ramp expert: 2/3 seeds** (clean replication on seeds 123/456: 1/2). It is the **best confirmed arm.**
- **A1 is not confirmed: 1/3 seeds**, only the selection seed. Its ramp gain replicates, but its calm-condition cost exceeds the margins on both replication seeds.

**Seed-mean CRPS (mean ± SD):**

| | Overall | Sunny | Cloudy | Ramp |
|---|---|---|---|---|
| D2 | 0.781 ± 0.022 | 0.329 ± 0.026 | 1.108 | 2.850 ± 0.013 |
| A1 | 0.810 ± 0.016 | 0.356 | 1.137 | **2.738 ± 0.012** |
| A2 | **0.755 ± 0.017** | 0.318 | **1.071** | **2.754 ± 0.020** |

**What replicated, and what did not:**
- **The ramp gains of both architectures replicate strongly:** about −0.10 kW, with every seed's CI below 0.
- **A2's seed-42 "better everywhere" pattern only partly replicated.** On average it is still better in calm conditions (overall −0.026, cloudy −0.037), but seed 123 has a sunny-day cost (+0.036). Part of the seed-42 calm gain reflected D2's weak seed 42 (overall 0.802 vs 0.759 / 0.783 on the other seeds).
- **Ramp detection is unchanged:** event AUPRC 0.50–0.53 for all arms on all seeds.

**Rolling-conformal 90% intervals (seed mean), coverage / width (kW) / interval score:**

| | Overall | Ramp | Non-ramp |
|---|---|---|---|
| D2 | 0.882 / 5.90 / 7.44 | 0.634 / 9.84 / 21.55 | 0.905 / 5.54 / 6.17 |
| A1 | 0.880 / 5.57 / 7.12 | 0.652 / 9.82 / **21.02** | 0.901 / 5.19 / 5.86 |
| A2 | 0.884 / 5.54 / **7.06** | 0.629 / 9.67 / 21.20 | 0.907 / 5.17 / **5.78** |

## Stage G: G3″ (A2 + E6-S, disagreement gate) vs G3′ (D2 + E6-S)
| Stratum (seed mean) | G3′ | **G3″** |
|---|---|---|
| Overall | 0.596 | **0.587** |
| Sunny | 0.122 | 0.121 |
| Cloudy | 0.939 | **0.923** |
| **Ramp** | 2.995 | **2.917** |
| Non-ramp | 0.379 | 0.376 |

Paired G3″ − G3′:

| Seed | Ramp | Overall | Rule |
|---|---|---|---|
| 42 | −0.026 [−0.048, −0.004] | −0.014 | pass |
| 123 | −0.108 [−0.137, −0.075] | −0.004 | pass |
| 456 | −0.100 [−0.124, −0.077] | −0.011 | pass |

**G3″ is RETAINED (3/3).**

Rolling-conformal 90% interval score: overall 5.14 → **5.07**; ramp 21.53 → **21.15**. Coverage is unchanged (overall 0.896, ramp 0.666).

**Descriptive context (common population; not a pre-registered test here):**
- **Ramps:** G3″'s ramp CRPS (2.917) is **below** that of the original 15-min ramp expert E4-B-fit (2.961).
- **Overall:** its overall CRPS (0.587) is below that of the calm expert E6-S (0.621).
- **The routing gap is closed descriptively:** the E7 criteria (ramp ≤ E4-B-fit + 0.02; overall ≤ E6-S + 0.02) are met in the seed mean. The first gated model to do so.

## Conclusions
1. **The training objective is not the limitation.** Input-weighted pinball loss (L1) has no effect.
2. **Spatio-temporal modelling is.** Temporal convolution (A2) gives a replicated ramp improvement of about −0.10 kW, with on-average neutral-to-better calm-condition performance, and is confirmed under the pre-registered rule.
3. **More capacity alone (A1) buys ramp accuracy at a calm-condition cost** and is not confirmed.
4. **Ramp *detection* still does not improve with any architecture** (AUPRC about 0.52). The gains are distributional, and ramp-conditioned 90% coverage stays about 0.63–0.67.
5. **New candidate forecaster of record:** **G3″ + causal rolling conformal** (experts: A2, the 60-min 3D-CNN direct-K model, and E6-S; disagreement gate). For the sealed test evaluation, the test pre-registration draft must be updated to G3″ (fallback G3 where a 60-min history is unavailable) before it is frozen.

## Forecast check (stated before Stage C)
| Prediction | Outcome |
|---|---|
| A1 keeps its ramp gain | ✓ (about −0.11) |
| A1 passes the full rule | ✗ (calm-condition cost) |
| A2 "better everywhere" weakens but holds on average | ✓ (partially) |
| A2 passes | ✓ (2/3) |
| G3″ improves ramps by about 0.03–0.06 | ✓, and larger than predicted on seeds 123/456 (about 0.10) |
| Detection unchanged | ✓ |
