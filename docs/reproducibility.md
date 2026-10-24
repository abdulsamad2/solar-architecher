# Reproducibility and scientific integrity

This page covers four things:
- what the repository contains;
- how to rebuild the development pipeline;
- how the official SKIPP'D test was protected;
- where its provenance is recorded.

> **The sealed test has already been run, once. Do not run it again.** The command in §4 is listed for provenance only. Its guard refuses to run while results or an execution log exist.

## 1. What is and is not distributed

| Item | In the repository? |
|---|---|
| Source code, configs, unit tests | yes |
| Run records: each run's `config.json` (exact settings and git commit), metrics, training histories, logs | yes, under `experiments/` |
| Raw sealed-test predictions | yes: [`experiments/final/test_predictions_v1.npz`](../experiments/final/test_predictions_v1.npz) |
| Pre-registrations, reports, raw result JSON, execution logs, hash inventories | yes, under `reports/` |
| Raw dataset | **no**; download it from the Stanford Digital Repository (§2) |
| Derived manifests and feature files (`data/`) | **no**; rebuilt by the code below |
| Trained model weights (`*.pt`) | **no**; they are not distributed. Their SHA-256 hashes are listed in [`reports/checkpoint_inventory.json`](../reports/checkpoint_inventory.json) |

Re-training reproduces the procedure, not bit-identical weights. Non-deterministic GPU kernels make exact weights unreproducible.

## 2. Environment and data

- **Software:** Python 3.11 (`pyproject.toml` pins `>=3.11,<3.12`).
- **Hardware:** a CUDA GPU is needed to train. The work used one NVIDIA RTX 5000 Ada (32 GB).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pytest                               # unit tests; real-data audits are opt-in (pytest -m realdata)
```

**Data.** SKIPP'D is available at <https://purl.stanford.edu/dj417rh1007> (DOI 10.25740/dj417rh1007; CC BY 4.0). Put `2017_2019_images_pv_processed.hdf5`, `times_trainval.npy` and `times_test.npy` in `data/raw/`.

## 3. Development pipeline

```bash
python -m src.skippd          # forecast manifest  -> data/manifests/forecast_tol30_v2.npz (±30 s matching)
python -m src.folds           # blocked folds      -> data/manifests/folds_blocked5_v1.json
python -m src.e8_audit        # 60-min eligibility manifest -> data/manifests/e8_eligibility_v1.npz (also rewrites reports/e8_eligibility_audit.json)
python -m src.e8_features     # 60-min windows for A2 / D4
pytest -m realdata            # real-data integrity audits

# Experts: run each for --seed 42, 123 and 456 (each run trains all 5 folds)
python -m src.train --target csi --input csi --head quantile --calib-holdout --max-epochs 15 --patience 5 --seed 42 --out experiments/E4_calib/v1
python -m src.train --target csi --input csi --head event --calib-holdout --frozen-thresholds reports/phase6_frozen_thresholds.json --max-epochs 15 --patience 5 --seed 42 --out experiments/E6/v1/E6-S
python -m src.train --target csi --input csi --head quantile --calib-holdout --e8-window H60 --encoder cnn3d_gru --max-epochs 15 --patience 5 --seed 42 --out experiments/E9/v1/A2
python -m src.train --target csi --input csi --head quantile --calib-holdout --e8-window H60 --e8-features PM --max-epochs 15 --patience 5 --seed 42 --out experiments/E8/v1/D4   # descriptive comparator

# Gates, development-only dry run and rehearsal
python -m src.gate7b_run       # G3 / G4 gates (E4-B-fit + E6-S), calibration days only
python -m src.freeze_g3pp      # G3″ gate (A2 + E6-S)
python -m src.final_dryrun     # out-of-fold assembly, calibration freeze, checkpoint inventory
python -m src.final_rehearsal  # full evaluation body on development data
python -m src.figures          # reports/figures/ (reads the committed test predictions and data/raw)
python scripts/make_readme_figures.py   # docs/figures/results_summary.svg from reports/test_results.json
```

**Notes**
- **Hash checks:** `src.e8_features`, `src.freeze_g3pp`, `src.final_dryrun` and `src.final_rehearsal` check the hashes of the frozen geometry file and the 60-min eligibility manifest (`src/e8_features.verify_frozen`). They stop if either differs from the original freeze.
- **Overwrites:** several development scripts write their summary JSON into `reports/`. Run them on a separate branch or clone, so the committed research record stays untouched.
- **Physics ablation (Phase F):** `scripts/run_physics_ablation.py` and `scripts/evaluate_physics_ablation.py`, configured by `configs/physics/`.

## 4. Sealed-test execution (provenance only; do not run)

- **Command used once:** `python -m src.final_execute <TOKEN>` → `src.final_eval.run_test`.
- **Guards:** the token must equal `reports/test_eval_approval.txt`, the tree must be clean, and the command refuses to run if an execution log exists.

| Item | Record |
|---|---|
| Pre-registration | [`reports/test_eval_prereg.md`](../reports/test_eval_prereg.md) (frozen at `491fac2`) |
| Development dry run | [`reports/final_dryrun_report.md`](../reports/final_dryrun_report.md) |
| Leakage audit | [`reports/leakage_audit_final.md`](../reports/leakage_audit_final.md) |
| Approval commit | `77fa2f3` |
| Code at execution | `e4b1048` |
| Raw outputs, committed unmodified | `cd7c30a` |
| Execution log | [`reports/test_execution_log.json`](../reports/test_execution_log.json) |
| Output hashes | [`reports/test_raw_output_hashes.json`](../reports/test_raw_output_hashes.json) |
| Deviations | [`reports/test_execution_deviations.md`](../reports/test_execution_deviations.md) |
| Final report | [`reports/gate5_test_report.md`](../reports/gate5_test_report.md) |

**Execution deviations.** Two aborted attempts preceded the run, and neither read any test PV or labels.
- **Attempt 1:** a log file was written too early and tripped the clean-tree guard.
- **Attempt 2:** the test HDF5 is not stored in time order.
- **Fixes:** neither fix changed any model or scoring rule. Both are documented in the deviations file.

Commit hashes cited in reports written before 2026-10-21 refer to the history before the author rewrite. See [`reports/commit_hash_mapping.md`](../reports/commit_hash_mapping.md).

## 5. Integrity practices

- **Pre-registration:** hypotheses, metrics and decision rules were committed before each result existed. The git history is the evidence.
- **Blocked folds:** contiguous day blocks with 1-day purges. Gates are fitted only on held-out calibration days, and the calibration window only on development data.
- **One-shot sealed test:** the test days were never used for training, selection or tuning. All model files were hash-inventoried before unsealing. Raw outputs were committed before any interpretation.
- **Follow-up studies on development data:** Phases A, C and F were each pre-registered. Phase C and Phase F used development data only and did not reuse the sealed test. Phase A ran the frozen models once on SkyGPT's public external test set.
- **Outcome labels:** results are labelled as confirmatory (pre-registered primary), secondary/descriptive (pre-specified) or exploratory (post hoc).
- **Negative and null results are reported:** non-confirmed hypotheses, the missed calibration target, unsuccessful variants and the Phase F null result.
