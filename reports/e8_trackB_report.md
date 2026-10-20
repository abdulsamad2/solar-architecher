# E8 Track B report: motion-augmented expert gating (CPU only, frozen experts)

2026-10-21.
- **Pre-registration:** `reports/e8_prereg.md` (frozen 826c34e / fd8a2ce, before extraction and runs).
- **Code:** `src/e8_trackb.py` (5453114).
- **Results:** `reports/e8_trackB_results.json`, `experiments/E8_trackB/v1/`.
- **Feature store:** `data/features/e8_*_v1` (hashes in `e8_features_v1.json`; 9 real-data pipeline tests pass).
- **Scope:**
  - No expert retraining; gates fitted on calibration days only (E7b procedure).
  - **Common population** (150,618 validation samples, eligible for H15/H30/H60), identical across methods (asserted).
  - Seeds 42/123/456 × 5 blocked folds.
  - E7/E7b results are preserved; frozen E7b G3/G4 are evaluated on the same samples as historical references. Official test set sealed.
- **No stop condition triggered:**
  - frozen geometry and eligibility hashes verified;
  - features causal;
  - identical samples;
  - finite, non-crossing quantiles;
  - causal conformal updates.

## Decision (pre-registered Track-B rule: G5 vs G3c)
**G5 does NOT add routing information: 1/3 seeds pass.**

| Seed | Ramp CRPS G5 − G3c [95% CI] | Overall G5 − G3c (≤ +0.005) | Pass |
|---|---|---|---|
| 42 | −0.008 [−0.012, −0.004] | +0.001 | yes |
| 123 | +0.001 [−0.004, +0.007] | −0.001 | no |
| 456 | +0.003 [−0.004, +0.009] | +0.000 | no |

## End-task forecasting: raw CRPS (kW), common population, mean of 3 seeds
| Stratum | E4-B-fit | E6-S | G3 (E7b) | G4 (E7b) | G3c | G4c | **G5** |
|---|---|---|---|---|---|---|---|
| Overall | 0.766 | 0.621 | 0.599 | **0.594** | 0.599 | **0.594** | 0.599 |
| Sunny | 0.310 | **0.120** | 0.121 | 0.120 | 0.121 | 0.121 | 0.121 |
| Cloudy | 1.096 | 0.983 | 0.944 | 0.937 | 0.944 | **0.936** | 0.944 |
| Ramp | **2.961** | 3.555 | 3.080 | 3.098 | 3.076 | 3.093 | 3.075 |
| Non-ramp | 0.568 | **0.356** | 0.375 | 0.368 | 0.375 | 0.369 | 0.375 |

**Contrasts** (seeds 42 / 123 / 456):
- **G5 − G4c:** ramp −0.017 / −0.015 / −0.023 (all CIs < 0), overall +0.005 / +0.004 / +0.005, cloudy +0.006 to +0.009. G5 sits on the G3 side of the G3/G4 trade-off.
- **G4c − G3c:** overall −0.004 to −0.005 (all CIs < 0); ramp +0.009 / +0.016 / +0.026. **The E7b G3/G4 trade-off replicates on the common population.**
- **Refits vs frozen E7b gates:** G3c and G4c are within 0.004 kW of the frozen E7b gates in every stratum.

**On the common (mid-day) population, ramps are harder.** E4-B-fit ramp CRPS is 2.961 here, against 2.448 on all samples: the sunrise ramps dropped by the 60-min eligibility requirement were easier. The gap between the best blend and the ramp expert is about 0.115 kW.

## Gate discrimination, reported separately from end-task CRPS
Ramp base rate 0.083. The gate weight w is used as a ramp score (gates are fitted for CRPS, not classification).

| | Ramp AUPRC | Ramp Brier | Mean w on ramps / non-ramps | p90 of w on ramps / non-ramps |
|---|---|---|---|---|
| G3c | **0.423** | **0.076** | 0.544 / 0.140 | 0.857 / 0.462 |
| G4c | 0.369 | 0.080 | 0.497 / 0.138 | 0.826 / 0.457 |
| G5 | 0.393 | 0.077 | 0.532 / 0.139 | 0.851 / 0.465 |

Single features as ramp scores (AUPRC, sign-agnostic):

| Feature | AUPRC |
|---|---|
| g (event probability) | **0.476** |
| d (expert disagreement) | 0.363 |
| inner-ring \|ΔI\| | 0.272 |
| shift magnitude | 0.217 |
| radial inflow | 0.140 |

- **Discrimination does not improve with motion:** G5's ramp ranking is *worse* than G3c's (0.393 vs 0.423 AUPRC), and its end-task CRPS is unchanged.
- **Fitted G5 coefficients** on the motion features are small and change sign between folds (e.g. seed 42 inflow −0.111 to +0.022; ring −0.241 to +0.331). That is consistent with little stable signal beyond g and d.

## Calibrated intervals (causal rolling conformal), 90%: coverage / width kW / interval score
| Stratum | E4-B-fit | G3c | G4c | G5 |
|---|---|---|---|---|
| Overall | 0.882 / 6.06 / 7.46 | 0.895 / 3.82 / 5.17 | 0.895 / 3.77 / **5.11** | 0.894 / 3.83 / 5.17 |
| Ramp | 0.627 / 10.18 / **21.24** | 0.669 / 10.40 / 21.47 | 0.661 / 10.28 / 21.39 | 0.669 / 10.41 / 21.49 |
| Non-ramp | 0.905 / 5.69 / 6.22 | 0.916 / 3.23 / 3.70 | 0.916 / 3.18 / **3.64** | 0.914 / 3.23 / 3.70 |

## Cost
- **Gate fitting:** 2.4 s per gate × fold × seed.
- **Motion features:** precomputed once per image record (349,372 records in 55 s on 32 CPU workers); at run time, a lookup.
- **Gate evaluation:** < 0.1 µs per sample.

## Interpretation and implications for Track A
1. **The pre-registered gate question is answered: no.** Short-baseline, origin-frame motion summaries (radial inflow toward the sun, inner-ring change, shift magnitude) do not improve routing beyond the event probability g and the expert disagreement d, for ranking or for CRPS.
2. **This does not test Track A's hypothesis.** Track A asks whether a *trained* expert with a longer history and per-frame solar/motion inputs anticipates ramps better. Track B only shows that **hand-crafted origin-frame motion summaries, fed to a linear gate on top of the current experts**, carry little new information.
3. **Prior for Track A:**
   - the weak single-feature AUPRCs (≤ 0.27) and the fold-to-fold sign instability lower the expected yield of the motion features (D3/D4);
   - the history-length arms (D1/D2) are not informed by Track B.
   - See the readiness checklist (`reports/e8_training_readiness.md`) for the resulting recommendation on Track A scope.
