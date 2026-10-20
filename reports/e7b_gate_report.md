# E7b report: enhanced gate features for expert routing

2026-10-21.
- **Pre-registration:** `reports/e7b_gate_prereg.md` (commit 53cac39; synthetic scoring tests in `tests/test_gate7b.py`, both before any E7b result).
- **Runner:** `src/gate7b_run.py` (54bef45).
- **Results:** `reports/e7b_results.json`, `experiments/E7b_gate/v1/`.
- **Scope:**
  - Experts frozen (E4-B-fit, E6-S); E7-B1 frozen as a reference. No joint training.
  - Gates fitted on calibration days only (out-of-sample for both experts); λ chosen on a chronological calibration split.
  - Seeds 42/123/456; 164,362 identical fold-validation samples. Official test set sealed.
  - E7 (B1 **NOT RETAINED**) and all earlier verdicts are unchanged.
- **No stop condition triggered:**
  - the fitting CRPS matched `crps_quantile_interp` to < 1e−8 in every fit;
  - the gate feature g reproduced E7's exactly;
  - no non-finite or crossing quantiles; causal conformal updates only.
- **Run note:** an earlier start of this exact run was interrupted by the user mid-execution (partial outputs for seed 42 and seed 123 fold 0). Those outputs were deleted unexamined, and the run was restarted from scratch at the same commit; it is fully deterministic.

## Decision (pre-registered)
| Gate | Features | Ramp CRPS vs G0 with CI < 0, and overall ≤ G0 + 0.005 (seeds 42/123/456) | Adds routing information? |
|---|---|---|---|
| G1 | g, direct spread | no / no / no (ramp **worse**: +0.008 to +0.016) | no |
| G2 | g, dual spread | no / no / no | no |
| **G3** | **g, expert disagreement** | **yes / yes / yes** | **yes** |
| G4 | g, spreads, disagreement | yes / yes / yes | yes |

- **Recommended gate (lowest seed-mean ramp CRPS among passing gates):** **G3**.
- **Sufficiency check (E7 thresholds):** G3's ramp CRPS is still +0.085 / +0.088 / +0.116 kW above the ramp expert (limit 0.02), although overall it is within the limit (−0.019 to −0.022 vs E6-S). **The routing gap does not close (0/3 seeds).**

## Raw CRPS, kW (mean over 3 seeds)
| Stratum | E4-B-fit | E6-S | E7-B1 | G0 | G1 | G2 | **G3** | G4 |
|---|---|---|---|---|---|---|---|---|
| Overall | 0.751 | 0.610 | 0.592 | 0.592 | 0.590 | 0.591 | 0.590 | **0.585** |
| Sunny | 0.305 | 0.123 | 0.122 | 0.123 | 0.122 | 0.124 | 0.124 | 0.122 |
| Cloudy | 1.068 | 0.958 | 0.926 | 0.925 | 0.923 | 0.923 | 0.922 | **0.915** |
| **Ramp** | **2.448** | 2.931 | 2.608 | 2.611 | 2.624 | 2.619 | **2.545** | 2.559 |
| Non-ramp | 0.560 | **0.351** | 0.366 | 0.365 | 0.362 | 0.363 | 0.371 | 0.364 |

**Paired contrasts** (day-bootstrap; seeds 42 / 123 / 456):
- **G3 − G0:**
  - ramp **−0.065 / −0.067 / −0.068** (all CIs < 0);
  - overall −0.002 / −0.001 / −0.002;
  - non-ramp +0.005 / +0.006 / +0.005.
- **G4 − G0:** ramp −0.053 / −0.056 / −0.049; overall −0.006 (all CIs < 0); cloudy −0.009 to −0.011.
- **G3 − E7-B1:** ramp −0.060 / −0.064 / −0.066 (all CIs < 0).
- **G0 − E7-B1:** ≤ 0.005 in every stratum (the refit reproduces B1 closely).

**What each feature does:**
- **Expert disagreement** carries routing information that the event probability lacks. It closes about **40%** of B1's remaining ramp gap (2.608 → 2.545, against a ramp-expert floor of 2.448), at a small non-ramp cost (+0.005).
- **Expert spreads do not help routing** on their own. The direct-expert spread even **hurts** ramps. In G4 the spreads re-allocate gains toward cloudy/overall (G4 is the best overall model at 0.585) rather than ramps.

## Gate diagnostics (mean over seeds; ramp base rate 0.101)
| Gate | Ramp AUPRC of w | Ramp Brier of w | w on ramps: mean (p10/p50/p90) | w on non-ramps: mean (p10/p50/p90) |
|---|---|---|---|---|
| g alone (raw probability) | 0.474 | 0.066 | – | – |
| G0 | 0.449 | 0.105 | 0.64 (0.49/0.67/0.78) | 0.20 (0.03/0.06/0.64) |
| G1 | 0.413 | 0.106 | 0.62 | 0.20 |
| G2 | 0.356 | 0.115 | 0.60 | 0.23 |
| **G3** | 0.433 | **0.086** | 0.53 (0.16/0.56/**0.84**) | **0.15** (0.01/0.06/**0.49**) |
| G4 | 0.388 | 0.090 | 0.49 (0.12/0.50/0.81) | 0.15 (0.01/0.05/0.48) |

- **The gates are fitted for CRPS, not classification**, so w is not a calibrated ramp probability, and AUPRC does not rank the gates by usefulness.
- **G3 routes more decisively:** the upper decile of w on ramps rises from 0.78 to 0.84, and the upper decile on non-ramps falls from 0.64 to 0.49.
- **Fitted coefficients:**
  - G3: positive weights on both g and disagreement in every fold (e.g. seed 42: g 0.80–1.40, d 0.77–1.65).
  - G4: positive weight on the residual-expert spread and negative weight on the direct-expert spread.
  - The selected λ is 1e−4 to 1e−2 throughout.

## Calibrated intervals (rolling conformal, causal), 90%: coverage / interval score
| | E4-B-fit | E6-S | E7-B1 | G0 | G3 | G4 |
|---|---|---|---|---|---|---|
| Overall | 0.890 / 7.28 | 0.897 / 5.44 | 0.896 / 5.13 | 0.896 / 5.14 | 0.897 / 5.11 | 0.898 / **5.06** |
| Ramp | 0.613 / **17.72** | 0.663 / 20.93 | 0.645 / 18.80 | 0.643 / 18.80 | 0.661 / 18.10 | 0.657 / 18.03 |
| Non-ramp | 0.921 / 6.11 | 0.924 / 3.71 | 0.924 / 3.60 | 0.924 / 3.61 | 0.923 / 3.66 | 0.925 / 3.60 |

Ramp-conditioned coverage stays around 0.61–0.66 for every method. No conditional guarantee is claimed.

## Cost
- **Gate fitting:** 2.4 s per gate × fold × seed.
- **Feature computation:** 37 µs per sample, dominated by the quantile-implied probability.
- **Gate evaluation:** 0.05 µs per sample.
- **Total:** negligible next to running the two experts (≈ 2.8 + 19.7 ms at batch 1).

## Conclusions and recommendation
1. **Improved gate information helps, but is not sufficient.**
   - Expert disagreement is a genuine, replicated routing signal: ramp CRPS falls by 0.065 kW on every seed, without an overall cost.
   - Even with it, the blend stays 0.09–0.12 kW behind the ramp expert on ramps.
   - Expert spreads add nothing to routing.
2. **The best available forecasters now:**
   - **G4:** best overall (0.585) and best calibrated interval score (5.06);
   - **G3:** the best ramp-protecting blend (2.545).
   - Choosing between them, or keeping E4-B-fit for ramp-critical use, remains a priority decision. Neither meets the E7 ramp non-inferiority threshold.
3. **Recommendation for the next experiment: focus on ramp detection itself, i.e. longer histories and solar-relative motion features**, rather than further gate engineering.
   - Every routing improvement so far is bounded by how well *any* model anticipates a ramp from the current inputs.
   - Both experts' predictive distributions see the same 15-minute, 64×64, non-sun-aligned window, and their disagreement is the only extra signal left to extract from them.
   - Suggested pre-registered direction:
     - a longer image/PV history (e.g. 30–60 min);
     - sun-centred, solar-relative crops or features (cloud motion toward the sun's position from frame differencing or optical flow);
     - evaluated first as an improved *ramp expert* and *gate input*, with E4-B-fit / G3 as frozen references and the same ramp-first decision rules.
