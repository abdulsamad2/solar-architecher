# E8 Track A training-readiness checklist

2026-10-21. **Status: READY, awaiting explicit approval. No Track A training has been launched.** The mandatory stop is in force.

## Checklist
| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Pre-registration frozen with Amendment A (short-baseline motion) and B (sunrise-ramp subgroup) | ✅ | `reports/e8_prereg.md` (826c34e, fd8a2ce) |
| 2 | Exactly 16 frames per window; H15/H30/H60 strides 1/2/4 min | ✅ | `test_window_timestamps_causal_same_day_and_16_frames` (all three windows) |
| 3 | Short-baseline motion from the record ≈ 1 min earlier; causal; actual separation verified; missing-pair policy (zero, flagged) | ✅ | `test_one_minute_pairs_separation_causality_and_missing_policy`: all pairs 0.5–1.5 min apart, predecessor earlier, 99.8% available, missing = exact zeros |
| 4 | Common-population manifest (150,618) and full eligible population (164,362) preserved; sunrise-ramp subgroup defined | ✅ | `data/manifests/e8_eligibility_v1.npz` (hash-frozen); `test_window_counts_and_common_population`; `src/e8_tracka_report.py` |
| 5 | Solar projection, orientation and coordinate convention verified | ✅ | label-free geometry per regime (median 0.63–0.71 px); `test_coordinate_convention_against_detected_sun` (median < 1.5 px); camera change 2019-05-22 |
| 6 | CPU feature store frozen with versioned metadata and hashes | ✅ | `data/features/e8_features_v1.json` (geometry, eligibility, record-features and windows SHA-256); verified at run time (`verify_frozen`) |
| 7 | Window-builder, timestamp, causality, shape and feature-equality tests | ✅ | `tests/test_e8_pipeline.py`: 9/9 pass (`pytest -m realdata`) |
| 8 | H15 window identical to the existing E4 history, so D0 uses the same frames | ✅ | `test_h15_window_equals_existing_e4_history` |
| 9 | Trainer integration: identical parameter counts across D0–D6; exact zero-fill; fit-day-only standardisation; common-population restriction; fallback predictions | ✅ | `tests/test_e8_trainer.py` (4 pass). GPU smoke (1 epoch, 60-day subset, fold 1): D0–D4 run; identical train/val/cal counts across arms; masks correct; extra-sample counts per window as expected |
| 10 | Legacy (non-E8) trainer unchanged | ✅ | E4-B smoke reproduces pre-E8 numbers exactly (1.450 / 1.569 kW) |
| 11 | Analysis script with pre-registered rules, fallback assembly and sunrise ramps | ✅ | `src/e8_tracka_report.py`; `test_fallback_assembly_uses_arm_where_available_else_base` |
| 12 | Full test suite | ✅ | 62 unit tests + 16 real-data tests pass |
| 13 | E7/E7b results unchanged; official test set sealed | ✅ | No E7/E7b file modified; no test-split access in any E8 code path |
| 14 | Track B completed and reported before Track A | ✅ | `reports/e8_trackB_report.md` (G5 not retained, 1/3 seeds) |

## Launch plan (on approval)
**Common flags:** `--head quantile --target csi --input csi --pool avg --max-epochs 15 --patience 5 --calib-holdout --out experiments/E8/v1/<ARM>`, with seeds 42 / 123 / 456.

| Arm | Extra flags |
|---|---|
| D0 | `--e8-window H15 --e8-features none` |
| D1 | `--e8-window H30 --e8-features none` |
| D2 | `--e8-window H60 --e8-features none` |
| D3 | `--e8-window H15 --e8-features PM` |
| D4 | `--e8-window H60 --e8-features PM` |

- **Schedule:** 15 runs in 2 GPU lanes, never more than 2 concurrent (GPU memory). Arms are interleaved across lanes so partial results stay balanced.
- **Estimated wall-clock:** ≈ 7 h (≈ 0.9 h per run on the common population).
- **Stage 2:** 6 runs (≈ 3 h), only if D3 or D4 passes the pre-registered rule.
- **Analysis:** `python -m src.e8_tracka_report`, then a Gate 4 report covering direct-K vs gating, common vs fallback population, and sunrise ramps.

## Recommendation on scope (given Track B)
**Run stage 1 as pre-registered (all five arms).**
- **Motion arms (D3/D4):** Track B lowered the expected yield of the hand-crafted motion features, since single-feature ramp AUPRC is ≤ 0.27 and the signs are unstable. A negative D3/D4 result is still informative.
- **History arms (D1/D2):** their hypothesis is untouched by Track B.
- **Cost:** dropping arms would save under 3 GPU hours and would be a deviation from the frozen plan.
