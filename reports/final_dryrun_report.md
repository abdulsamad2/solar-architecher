# Final test pipeline: development-only dry run and rehearsal report

2026-10-23. **No official test outcome, test PV or test image has been accessed.**
- **Protocol:** `reports/test_eval_prereg.md` (frozen 491fac2, with Amendments 1–2).
- **Code:** `src/final_eval.py`, `src/final_dryrun.py`, `src/final_rehearsal.py` (latest commit 9fd702a).
- **Raw outputs:** `reports/final_dryrun_results.json`, `reports/final_rehearsal_dev.json`, `reports/checkpoint_inventory.json`.

## 1. Numerical reproduction: a code-correctness test, NOT validation
Each member's stored fold-validation predictions are recomputed from its frozen checkpoint with the final inference code.

| Model | Members | Max \|diff\| (kW) | Samples > 1e−4 kW (all members) | Status |
|---|---|---|---|---|
| E4-B-fit | 15 | 1.84e−3 | 12 | ✅ |
| E6-S | 15 | 5.14e−4 | 5 | ✅ |
| A2 (3D-CNN) | 15 | 7.47e−4 | 24 | ✅ |
| D4 | 15 | 1.87e−3 | 130 | ✅ |
| G3 gate (frozen E7b) | 15 | 5.34e−4 | – | ✅ |
| G4 gate (frozen E7b) | 15 | 1.22e−3 | – | ✅ |
| G3″ gate (refit, `src/freeze_g3pp.py`) | 15 | E9 Stage-G CRPS reproduced **exactly** (< 1e−9) | – | ✅ |

**Criterion (Amendment 2, flagged for review):** per member, ≥ 99.9% of samples within 1e−4 kW, max ≤ 5e−3 kW, mean ≤ 1e−6 kW. Two dry-run attempts failed before it passed, and both are documented:
1. **Amendment 1:** inference used fp32 matmul while training used TF32 (5.8e−3 kW). Fixed by matching backend flags; the error dropped to 1.8e−7.
2. **Amendment 2:** isolated bf16 rounding-boundary flips (4 of 34,099 samples in one member) made a max-only tolerance ill-posed. The criterion was revised.

## 2. Out-of-fold development assembly: genuinely held out (3-member surrogate)
Each member predicts only its own fold's validation samples. The full G3″-F routing, including the fallback, ran on all 164,362 development validation samples.
- **Routing:** 91.7% G3″ (valid 60-min history), 8.3% G3 fallback.

| CRPS (kW) | Overall | Sunny* | Cloudy | Ramp | Non-ramp | Sunrise ramp | 60-min eligible | Fallback samples |
|---|---|---|---|---|---|---|---|---|
| **G3″-F** | **0.573** | 0.121 | **0.894** | **2.411** | 0.367 | 0.903 | 0.581 | 0.482 |
| E4-B-fit | 0.730 | 0.291 | 1.043 | 2.436 | 0.539 | **0.873** | 0.745 | 0.562 |
| E6-S | 0.604 | 0.121 | 0.948 | 2.917 | **0.345** | 1.015 | 0.614 | 0.487 |
| G4 | 0.578 | **0.120** | 0.905 | 2.548 | 0.358 | 0.904 | 0.587 | **0.479** |
| D4 (60-min eligible only) | – | – | – | 2.746 | – | – | 0.793 | – |

\*Development "sunny" is the development clear-day label, not the official test labels.

## 3. Calibration freeze (development only)
- **Window lengths:** W per model from the E4-CAL2 rolling selection on the development out-of-fold stream: **G3″-F 4620, E4-B-fit 990, E6-S 4620, G4 4620** samples.
- **Initialisation scores:** frozen in `experiments/final/dev_oof_calibration_v1.npz` (hash in the inventory).
- **Daily-reset sanity run** (initialised from other days only), 90% coverage / interval score:

  | Stratum | G3″-F | E4-B-fit |
  |---|---|---|
  | Overall | 0.901 / 4.93 | 0.896 / 7.12 |
  | Ramp | 0.651 / 17.70 | 0.605 / 17.47 |
  | Fallback samples | 0.836 / 4.34 | 0.735 / 5.20 |

## 4. Rehearsal of the full evaluation body (`evaluate()`, the code the one-shot run uses)
This was run on the development out-of-fold forecasts with development day labels and no SUNSET. **It is not a test result.** Bootstrap CIs span 497 development days, not the 20 test days.

| Hypothesis (rehearsal) | Estimate [95% CI] |
|---|---|
| H1: overall CRPS G3″-F − E4-B-fit | −0.157 [−0.170, −0.145] |
| H2: G3″-F − persistence, overall / cloudy | −0.359 [−0.388, −0.332] / −0.501 [−0.550, −0.455] |
| H3: ramp CRPS G3″-F − E4-B-fit (NI margin 0.02) | −0.025 [−0.054, +0.005]; upper bound ≤ 0.02, so non-inferior |
| H4: RMSE skill vs persistence | 17.0% |
| H5: calibrated 90% coverage | 0.901 |

- **Comparator CRPS:** persistence 0.931 overall; E4-B-fit on the 60-min-eligible subset 0.745 (vs D4 0.793).
- **Ramp-conditioned 90% coverage:** G3″-F 0.648.
- **Expectation for the test run:** with only 20 test days, the test CIs will be much wider than these development CIs.

## 5. Other checks
- **Synthetic fixtures** (`tests/test_final_eval.py`, 5 tests):
  - the ensemble is level-wise and monotone;
  - daily-reset calibration is causal with no carry-over;
  - nearest-day initialisation;
  - the bootstrap is deterministic;
  - the one-shot runner refuses without approval.
- **Test suite:** 75 unit tests + 16 real-data tests pass.
- **Leakage audit:** `reports/leakage_audit_final.md` (12 controls).
- **Inventory:** 140 files hashed (60 expert checkpoints, configurations, standardisation files, 30 gate-parameter files, the SUNSET test predictions, frozen thresholds and definitions, manifests, calibration scores). Every hash is re-verified by `run_test` before execution.
