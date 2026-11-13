<div align="center">

# SHADOW-MoE

**Research code, protocols and result files for probabilistic 15-minute PV power forecasting from ground-based sky images (Stanford SKIPP'D benchmark)**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)
[![Status: research code](https://img.shields.io/badge/status-research%20code-lightgrey.svg)](#status)

[Overview](#overview) · [What has been done](#what-has-been-done) · [Layout](#repository-layout) · [Install and reproduce](#install-and-reproduce) · [Integrity](#integrity-of-the-test-evaluation) · [Cite](#citation)

</div>

## Overview

This repository holds everything needed to inspect and reproduce the experiments: source code, configurations,
pre-registrations and protocols, sample and split manifests, run records and the raw result files. It does not
contain a paper; results are documented in the report and result files linked below.

The forecasting pipeline implemented in [`src/`](src):

- **Clear-sky normalisation** ([`src/clear_sky.py`](src/clear_sky.py)): a physical clear-sky model of the array
  (pvlib solar position, Ineichen–Perez irradiance, Hay–Davies transposition, incidence-angle modifier) turns PV
  power into a clear-sky index, which is the learning target.
- **Three separately trained sky-image experts** ([`src/models.py`](src/models.py), [`src/train.py`](src/train.py)):
  a direct quantile expert (`E4-B-fit`), a calm-condition expert with a Gaussian-mixture event head (`E6-S`) and a
  60-min 3D-CNN expert (`A2`).
- **Gated quantile blend** ([`src/gate7b.py`](src/gate7b.py), [`src/freeze_g3pp.py`](src/freeze_g3pp.py)): a
  low-capacity logistic gate, fitted after the experts by CRPS minimisation, averages the quantiles of two experts
  per forecast; a timestamp rule chooses the 60-min or the 15-min pair (`G3″-F`).
- **Calibration** ([`src/calibrate.py`](src/calibrate.py)): rolling-window conformal prediction of a 15-member
  ensemble (5 folds × 3 seeds).
- **Evaluation** ([`src/metrics.py`](src/metrics.py), [`src/final_eval.py`](src/final_eval.py)): CRPS, RMSE, MAE,
  interval coverage, width and Winkler score, paired day-level bootstrap.

## What has been done

Each study has a protocol written before its result existed, the code that ran it, and its raw result file.

**Development and the official test**

| Study | Protocol | Report | Results |
|---|---|---|---|
| Development gates (calibration, blending, gate design, longer-history expert) | [`reports/gate2_calibration_prereg.md`](reports/gate2_calibration_prereg.md), [`reports/e7b_gate_prereg.md`](reports/e7b_gate_prereg.md), [`reports/e8_prereg.md`](reports/e8_prereg.md), [`reports/e9_prereg.md`](reports/e9_prereg.md) | [`reports/gate2_report.md`](reports/gate2_report.md), [`reports/gate3_report.md`](reports/gate3_report.md), [`reports/gate4_report.md`](reports/gate4_report.md) | `experiments/`, [`reports/final_rehearsal_dev.json`](reports/final_rehearsal_dev.json) |
| Official SKIPP'D test (one-shot, pre-registered) | [`reports/test_eval_prereg.md`](reports/test_eval_prereg.md) | [`reports/gate5_test_report.md`](reports/gate5_test_report.md) | [`reports/test_results.json`](reports/test_results.json), `experiments/final/` |
| External comparability (SkyGPT test set) | [`reports/phaseA_comparability_prereg.md`](reports/phaseA_comparability_prereg.md) | [`reports/phaseA_report.md`](reports/phaseA_report.md) | [`reports/phaseA_a2_results.json`](reports/phaseA_a2_results.json) |
| Calibration under temporal shift | [`reports/phaseC_calibration_prereg.md`](reports/phaseC_calibration_prereg.md) | [`reports/phaseC_report.md`](reports/phaseC_report.md) | `experiments/phaseC/` |
| Reconstructed-power and ramp-aware loss penalties | [`reports/phaseF_physics_prereg.md`](reports/phaseF_physics_prereg.md) | [`reports/phaseF_report.md`](reports/phaseF_report.md) | [`reports/phaseF_bootstrap_all.json`](reports/phaseF_bootstrap_all.json) |
| Temporal response (lag) of the forecasts | – | – | [`reports/lag_analysis.json`](reports/lag_analysis.json) |

**Follow-up analyses** (in [`revision/`](revision); none changes the frozen test evaluation)

| Analysis | Protocol | Report | Results |
|---|---|---|---|
| Input timing audit and shifted-input sensitivity | [`R-C06`](revision/protocols/R-C06_input_timing.md) | [`C06_timing.md`](revision/reports/C06_timing.md) | [`c06_timing_audit.json`](revision/results/c06_timing_audit.json), [`c06_shifted_inputs.json`](revision/results/c06_shifted_inputs.json) |
| Conformal initialisation audit and past-only rerun | [`R-C07`](revision/protocols/R-C07_conformal_pastonly.md) | [`C07_C19_conformal.md`](revision/reports/C07_C19_conformal.md) | [`c07_conformal_pastonly.json`](revision/results/c07_conformal_pastonly.json), [`posthoc_test_c19.json`](revision/results/posthoc_test_c19.json) |
| Contribution by condition and strong comparators | [`R-C16_C17_C20`](revision/protocols/R-C16_C17_C20.md) | [`C16_C17_C20_C22.md`](revision/reports/C16_C17_C20_C22.md) | [`c16_c20_contribution_bootstrap.json`](revision/results/c16_c20_contribution_bootstrap.json) |
| Event probabilities and ramp metrics | [`R-C18`](revision/protocols/R-C18_events.md) | [`C15_C18_events.md`](revision/reports/C15_C18_events.md) | [`c18_events_dev.json`](revision/results/c18_events_dev.json), [`posthoc_test_c18.json`](revision/results/posthoc_test_c18.json) |
| External CRPS with matched estimators; end-to-end latency | [`R-C22_C21`](revision/protocols/R-C22_C21_latency_external.md) | [`C21_external.md`](revision/reports/C21_external.md) | [`c21_external_crps.json`](revision/results/c21_external_crps.json), [`c22_latency.json`](revision/results/c22_latency.json) |
| PV-only probabilistic baseline (no images) | [`R-E1`](revision/protocols/R-E1_pv_only.md) | [`E1_E4_images_history.md`](revision/reports/E1_E4_images_history.md) | [`e1_e4_dev.json`](revision/results/e1_e4_dev.json), [`posthoc_test_e1e4.json`](revision/results/posthoc_test_e1e4.json) |
| Gate-feature and constant-weight ablation | [`R-E2`](revision/protocols/R-E2_gate_ablation.md) | [`E2_gate_ablation.md`](revision/reports/E2_gate_ablation.md) | [`e2_gate_ablation_dev.json`](revision/results/e2_gate_ablation_dev.json), [`posthoc_test_e2.json`](revision/results/posthoc_test_e2.json) |
| Forward-in-time evaluation with past-only calibration | [`R-E3`](revision/protocols/R-E3_forward.md) | [`E3_forward.md`](revision/reports/E3_forward.md) | [`e3_forward.json`](revision/results/e3_forward.json), [`folds_forward_v1.json`](revision/manifests/folds_forward_v1.json) |
| History length × image encoder | [`R-E4`](revision/protocols/R-E4_history_encoder.md) | [`E1_E4_images_history.md`](revision/reports/E1_E4_images_history.md) | [`e1_e4_dev.json`](revision/results/e1_e4_dev.json) |

Hashes of the models, manifests and inputs used by these analyses are in
[`revision/hash_inventory.json`](revision/hash_inventory.json); the order of work is in
[`revision/commit_log.md`](revision/commit_log.md).

## Repository layout

```
src/          data, models, training, gating, calibration, evaluation and the follow-up analyses (revision_*.py)
tests/        unit tests (pytest) and opt-in real-data audits (pytest -m realdata)
configs/      run configurations
scripts/      ablation runners and helper scripts
experiments/  run records: configs, metrics, training histories, logs, raw predictions (no model weights)
reports/      pre-registrations, reports, audits and raw results of the development work and the official test
revision/     protocols, manifests, reports and raw results of the follow-up analyses
docs/         development history and the reproducibility guide
```

## Install and reproduce

Python 3.11 is required; training needs a CUDA GPU (the runs used one NVIDIA RTX 5000 Ada, 32 GB). The exact
environment is pinned in [`requirements-lock.txt`](requirements-lock.txt).

```bash
git clone https://github.com/abdulsamad2/solar-architecher.git
cd solar-architecher
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pytest                             # unit tests; no data required
```

Download `2017_2019_images_pv_processed.hdf5`, `times_trainval.npy` and `times_test.npy` from the
[Stanford Digital Repository](https://doi.org/10.25740/dj417rh1007) into `data/raw/`, then build the manifests:

```bash
python -m src.skippd               # forecast manifest (±30 s matching)
python -m src.folds                # blocked 5-fold development folds
pytest -m realdata                 # opt-in real-data integrity audits
```

The full training, gating and rehearsal sequence is described in [`docs/reproducibility.md`](docs/reproducibility.md).
Trained model weights are not distributed.

> [!WARNING]
> **Sealed test: do not rerun.** `src.final_execute` / `src.final_eval.run_test` is a one-shot evaluation that has
> already been executed; its guards refuse to run while results exist.

## Integrity of the test evaluation

- **Pre-registration:** every hypothesis and analysis protocol was committed before its result existed.
- **Leakage control:** blocked development folds; gates and calibration were fitted on development data only.
- **Before unsealing:** a [leakage audit](reports/leakage_audit_final.md) and SHA-256 verification of all model files.
- **One-shot run:** the test ran once and its raw outputs were committed unchanged before analysis
  ([`reports/test_execution_manifest.json`](reports/test_execution_manifest.json)).
- **Deviations:** two aborted attempts that never read test labels are documented in
  [`reports/test_execution_deviations.md`](reports/test_execution_deviations.md).
- **Follow-up analyses** are labelled post hoc or forward and are kept separate from the frozen evaluation.

## Citation

Please cite this repository (metadata in [`CITATION.cff`](CITATION.cff); archived releases on Zenodo) together with
the dataset:

```bibtex
@software{laghari_shadow_moe_2026,
  author = {Laghari, Abdul Samad},
  title  = {{SHADOW-MoE}: probabilistic 15-minute solar power forecasting from sky images},
  year   = {2026},
  url    = {https://github.com/abdulsamad2/solar-architecher}
}
```

Dataset: Y. Nie, X. Li, A. Scott, Y. Sun, V. Venugopal, A. Brandt. SKIPP'D: a SKy Images and Photovoltaic Power
Generation Dataset for short-term solar forecasting. *Solar Energy* 255:171–179, 2023.
[doi:10.25740/dj417rh1007](https://doi.org/10.25740/dj417rh1007).

## Licences

- **Code:** [MIT licence](LICENSE).
- **Vendored helper:** [`src/official/relative_op_func.py`](src/official/relative_op_func.py) is © the SKIPP'D
  authors, MIT licence ([`src/official/LICENSE_SKIPPD`](src/official/LICENSE_SKIPPD)).
- **Data:** SKIPP'D is distributed under CC BY 4.0. We thank its authors at Stanford University for making it public.

## Status

Research code for reproducing the experiments listed above, not a production forecasting system.

Author: Abdul Samad Laghari.
