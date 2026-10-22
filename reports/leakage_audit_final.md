# Leakage and causality audit for the final test pipeline

2026-10-23. **Scope:** `src/final_eval.py` (test inference, routing, ensemble, calibration, scoring) and every frozen input it reads.

| # | Possible leak | Control | Evidence |
|---|---|---|---|
| 1 | Future frames in model inputs | Windows are built from timestamps at t − j·stride (±30 s), same day, **≤ origin**; H15 rows equal the frozen manifest history. | `test_window_timestamps_causal_same_day_and_16_frames`, `test_h15_window_equals_existing_e4_history`; `Split` asserts the window origin equals the forecast origin |
| 2 | Cross-split frames (dev ↔ test) | `Split` asserts every window frame belongs to the split's own HDF5 group | assertion in `Split.__init__`; dev windows were checked when the feature store was built |
| 3 | Future records in motion features (D4) | Motion uses the record ≈ 1 min **earlier** (same day), only from in-window frames | `test_one_minute_pairs_separation_causality_and_missing_policy` |
| 4 | Labels in routing | The G3″-F route depends **only** on 60-min history availability (timestamps). Gate features g and d are functions of the expert forecasts only (no PV target). | `route_g3pp_f`; `raw_features` takes predictions only |
| 5 | Test information in clear-sky anchors | Each member's anchor is fitted on its development **fit days**; the ramp anchor on **all development days**. | `member_anchor`; `reports/test_ramp_definition_frozen.json` |
| 6 | Test information in standardisation | D4 features are z-scored with the stored fit-day statistics (`e8_standardisation.json`) | `member_predict` |
| 7 | Test information in thresholds | τ (training-label and test ramp) comes from development only | frozen JSON files; hashes in the inventory |
| 8 | Calibration using unmatured outcomes | A score enters the window only when target time ≤ issue time; asserted per update | `rolling_conformal_daily`; `test_rolling_daily_reset_causality_and_no_carryover` |
| 9 | Calibration carrying across disconnected days | The window is reset at each test day and initialised from development out-of-fold scores only | same test; `nearest_init` |
| 10 | Tuning on test | No parameter is estimated after loading test labels. Labels are first read *after* all forecasts are assembled, and are used only for scoring and for calibration feedback that has matured. | order of operations in `run_test` |
| 11 | Repeated or altered execution | The runner refuses without a recorded approval token, refuses with a dirty tree or existing results, and verifies every inventoried hash | `run_test` guards; `test_one_shot_runner_refuses_without_recorded_approval` |
| 12 | Evaluation-sample mismatch | All full-population models are scored on the same 13,266 test samples. SUNSET sample order is asserted equal. D4 is reported only on the 60-min-eligible subset, alongside E4-B-fit on that same subset. | `run_test` assertions |
