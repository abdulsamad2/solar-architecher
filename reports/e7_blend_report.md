# E7-BLEND report: post-hoc gated mixture of experts

2026-10-21.
- **Protocol:** `reports/e7_blend_prereg.md` (commit 62c8ff5, before evaluation).
- **Code:** `src/blend.py`, `src/blend_run.py` (commit 1fd44a7).
- **Results:** `reports/e7_blend_results.json`; blended predictions in `experiments/E7_blend/v1/`.
- **Scope:**
  - No retraining. Blend parameters were fitted on calibration days only.
  - Seeds 42/123/456; 164,362 identical fold-validation samples.
  - Official test set sealed. Gate 2, E4-CAL2 and Gate 3 verdicts unchanged.
  - No stop condition triggered: identical samples across experts and methods, no crossing quantiles, causal conformal updates.

## Verdict (pre-registered rule)
**The gated blend B1 is NOT retained as the forecaster of record.**
| Condition (≥ 2/3 seeds required) | Seeds met |
|---|---|
| 1. Ramp CRPS within 0.02 kW of the ramp expert (E4-B-fit) | **0/3** (B1 is +0.145 / +0.152 / +0.182 worse) |
| 2. Overall CRPS within 0.02 kW of the calm expert (E6-S) | 3/3 (B1 is better: −0.015 / −0.021 / −0.019) |
| 3. Gating beats constant blending on ramps (CI < 0) | 3/3 (−0.179 / −0.169 / −0.201) |

The "ensembling, not gating" branch does not apply, because condition 1 fails.

## Results: raw CRPS, kW (mean ± SD over 3 seeds)
| Stratum | Ramp expert E4-B-fit | Calm expert E6-S | B0 constant (w ≈ 0.2) | **B1 gated** | B2 switch |
|---|---|---|---|---|---|
| **Overall** | 0.751 | 0.610 | 0.608 | **0.592 ± 0.003** | 0.611 |
| Sunny | 0.305 | 0.123 | 0.142 | **0.122** | 0.125 |
| Cloudy | 1.068 | 0.958 | 0.940 | **0.926** | 0.957 |
| **Ramp** | **2.448** | 2.931 | 2.791 | 2.608 | 2.550 |
| Non-ramp | 0.560 | **0.351** | 0.364 | 0.366 | 0.394 |
| Fold 1 | 1.315 | 1.095 | **1.087** | 1.088 | 1.141 |
| Fold 1, Sep–Oct 2017 | 0.997 | 0.743 | 0.740 | **0.703** | 0.724 |

References: persistence is 0.877 overall and 4.672 on ramps; E4-B (full training) is 0.733 overall.

**Paired contrasts for B1** (day-bootstrap CIs; all three seeds agree in sign):
- **vs calm expert:**
  - overall −0.015 to −0.021 (CI < 0);
  - cloudy −0.025 to −0.036;
  - **ramp −0.32 to −0.33**;
  - non-ramp +0.012 to +0.018.
- **vs ramp expert:** overall −0.158 to −0.159; ramp +0.145 to +0.182.
- **vs constant blend B0:** overall −0.015 to −0.019; ramp −0.17 to −0.20; sunny −0.02. All CIs are below 0.

**Interpretation:**
- **The ramp gap:** B1 closes **about two thirds** of it between the two experts (E6-S 2.931 → B1 2.608, against E4-B-fit 2.448). It does so while being the **best model overall** (0.592), on cloudy days, and in the Sep–Oct 2017 period. On sunny days it is equal to the best.
- **Dominance:** B1 is better than E6-S both overall and on ramps, and better than B0 everywhere that matters. Against E4-B-fit it remains a trade-off: much better overall and in calm conditions, worse on ramps.
- **Hard switching (B2)** gets closer on ramps (2.550) but gives back the calm-condition gains (overall 0.611).

## Gate behaviour
- **Fitted parameters:** B1 has a ≈ 0.75–1.5 and b ≈ 0.4–0.6 in every fold and seed. Constant-blend weights are 0.15–0.30; switch thresholds are at the 60th–85th percentile of g.
- **Weight allocation:** the mean ramp-expert weight is **0.65 on ramp samples vs 0.20 on non-ramp samples**. The B2 switch selects the ramp expert on 85% of ramp samples and 21% of non-ramp samples.
- **Gate quality:** the Brier score of g as a ramp detector is 0.066 (climatology 0.091).
- **The limit:** the gate is informative but not sharp enough. Ramp detection from the input window remains the binding constraint, as in Gate 3.

## Calibrated intervals (E4-CAL2 rolling conformal), 90%: coverage / width (kW) / interval score
| Stratum | E4-B-fit | E6-S | B0 | **B1** | B2 |
|---|---|---|---|---|---|
| Overall | 0.890 / 5.94 / 7.28 | 0.897 / 3.83 / 5.44 | 0.894 / 4.10 / 5.53 | 0.896 / **3.68 / 5.13** | 0.896 / 3.75 / 5.27 |
| Ramp | 0.613 / 8.36 / **17.72** | 0.663 / 8.98 / 20.93 | 0.623 / 8.59 / 19.70 | 0.645 / 8.36 / 18.80 | 0.660 / 8.39 / 18.77 |
| Non-ramp | 0.921 / 5.67 / 6.11 | 0.924 / 3.25 / 3.71 | 0.924 / 3.59 / 3.94 | 0.924 / **3.15 / 3.60** | 0.923 / 3.23 / 3.76 |

- **B1 + rolling conformal** has the best overall interval score of any model so far (5.13, against 7.28 for E4-B-fit).
- **Ramp-conditioned coverage** stays around 0.61–0.66 for every method. No conditional guarantee is claimed.

## Cost
- **Fitting:** 15 s per fold and seed (CPU).
- **Application:** 0.31 µs per sample, on top of running both experts.
- **Inference:** about 2.8 ms + 19.7 ms at batch 1 (dominated by the E6-S mixture quantiles).

## Conclusions and recommendation
1. **Under the pre-registered rule, no blend is retained.** The ramp non-inferiority condition (≤ 0.02 kW from the ramp expert) is not met in any seed.
2. **Gating demonstrably works.** It beats plain ensembling on ramps and overall in every seed, and B1 dominates the calm expert E6-S. That supports the mixture-of-experts *principle*: forecast-time event probability usefully routes between a calm and a ramp model.
3. **The remaining ramp gap (≈ 0.16 kW)** is set by gate sharpness. The event probability assigns only about 0.65 weight to the ramp expert on actual ramps. The bottleneck is the same as Gate 3: **detecting ramps from a 15-minute, 64×64 image history.**
4. **Choosing a forecaster is now an explicit priority decision.** It is not settled by the pre-registered criterion:
   - **if overall or calm-condition accuracy matters most:** B1 (0.592 overall CRPS, best calibrated intervals);
   - **if ramp accuracy matters most:** E4-B-fit.
   - Any adoption of B1 should be recorded as a decision outside this pre-registered test.

**Next experiments, each to be pre-registered:**
- **(a) End-to-end training** of a learned gate with both experts (a trained MoE), which may sharpen routing.
- **(b) Better ramp-detection inputs:** longer history, cloud-motion or sun-region features, higher resolution. This addresses the shared bottleneck.
- **(c) A gate feature from the ramp expert's own spread**, or a combination of both experts' event probabilities.
