<div align="center">

# SHADOW-MoE

**Probabilistic 15-minute solar power forecasting from ground-based sky images**

A disagreement-gated mixture of sky-image experts, evaluated in a pre-registered, one-shot sealed test on Stanford SKIPP'D

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)
[![Status: research code](https://img.shields.io/badge/status-research%20code%2C%20manuscript%20in%20preparation-lightgrey.svg)](#research-status)

[Results](#results-at-a-glance) · [Method](#method) · [Data](#dataset-and-preprocessing) · [Quick start](#installation-and-quick-start) · [Reports](#reports-and-figures) · [Integrity](#reproducibility-and-scientific-integrity) · [Limitations](#limitations-and-negative-results) · [Citation](#citation)

</div>

> [!IMPORTANT]
> **Key finding.** On the official 20-day SKIPP'D test, the gated forecaster G3″-F lowered overall CRPS against its strongest single expert by **9.9%**: 1.006 vs 1.117 kW, 95% CI of the difference [−0.149, −0.075] kW. The primary hypothesis is confirmed.
>
> Two pre-registered secondary targets were **not** met:
> - non-inferiority on ramps;
> - 90% interval coverage within ±3 percentage points of nominal.

## Overview

**Research question.** Can a forecaster that routes between specialised sky-image experts produce better probabilistic 15-minute PV forecasts than its strongest single expert?

**What this repository contributes**
- **A gated forecaster, G3″-F.**
  - It combines a 3D-CNN ramp expert (60-minute image history) with a residual event-mixture expert that is sharp in calm conditions.
  - A logistic gate blends the two. Its inputs include their *disagreement*.
  - A timestamp rule falls back to a 15-minute direct expert when 60 minutes of history are not yet available.
  - The output is 9 predictive quantiles, made from a 15-member ensemble and calibrated by causal rolling conformal prediction.
- **A one-shot sealed evaluation** on the Stanford SKIPP'D benchmark:
  - a frozen protocol;
  - hash-verified model inventories;
  - raw outputs committed before interpretation.
- **Follow-up studies, all pre-registered:**
  - a comparison with published methods;
  - a calibration study under temporal shift;
  - a physics-informed training-objective study.
- **Complete reporting of what did not work:** hypotheses that were not confirmed, the missed calibration target, unsuccessful model variants and a null result for physics-informed losses.

## Results at a glance

### Primary hypothesis (confirmatory, pre-registered)

| Official test | Forecasts | G3″-F CRPS | E4-B-fit CRPS | Difference [95% CI] | Verdict |
|---|---|---|---|---|---|
| SKIPP'D, 20 days (10 sunny, 10 cloudy) | 13,266 | 1.006 kW | 1.117 kW | −0.111 [−0.149, −0.075] kW (−9.9%) | **Confirmed** |

CRPS is the continuous ranked probability score; lower is better. The CI comes from a paired test-day bootstrap: 10,000 resamples, stratified sunny/cloudy. E4-B-fit is the strongest single model, a direct 15-minute CNN-GRU quantile expert.

<p align="center">
  <img src="docs/figures/results_summary.svg" alt="Forest plot of G3″-F CRPS changes with 95% CIs versus E4-B-fit and smart persistence, and a dot plot of 90% interval coverage by stratum against the 0.87–0.93 target" width="900">
</p>

The figure is generated from [`reports/test_results.json`](reports/test_results.json) by [`scripts/make_readme_figures.py`](scripts/make_readme_figures.py).

### Secondary, descriptive and exploratory results

| Question | Type | Result |
|---|---|---|
| G3″-F beats smart persistence (H2) | pre-registered secondary | **Confirmed:** overall −0.701 [−0.823, −0.582]; cloudy days −1.208 [−1.454, −0.971] |
| Ramp CRPS non-inferior to E4-B-fit (H3, margin 0.02 kW) | pre-registered secondary | **Not established:** −0.117 [−0.244, +0.031] |
| RMSE skill vs persistence (H4) | descriptive | G3″-F 18.6%; our SUNSET port 16.2% |
| Calibrated 90% coverage within 0.90 ± 0.03 (H5) | pre-registered secondary | **Not met:** 0.935 (over-covers); ramps 0.746 |
| Per-day comparison with E4-B-fit | descriptive | G3″-F better on 19 of 20 test days |
| Forecast anticipates PV changes (lag analysis) | exploratory | **No.** Forecast changes follow observed changes about 15 minutes later |

**Follow-up studies (development data only; the sealed test was not reused)**
- **Calibration (Phase C):** the frozen daily-reset conformal method is fragile when its initial scores come from a distant period. Continuous rolling or adaptive conformal calibration stays at 0.88–0.92 coverage. The pre-registered primary hypothesis was not confirmed.
- **Physics-informed losses (Phase F):** adding solar-physics terms to the ramp expert's training objective changed CRPS by ≤ 0.6%. No variant met the pre-registered selection rule, so this is a null result in practice.

Full report: [Gate 5 test report](reports/gate5_test_report.md).

## Method

<p align="center">
  <img src="docs/figures/architecture.svg" alt="SHADOW-MoE G3″-F architecture: inputs, clear-sky normalisation, three experts, disagreement-guided gates, timestamp router with fallback, 15-member ensemble, conformal calibration" width="820">
</p>

1. **Clear-sky normalisation.**
   - A physical clear-sky model gives the power the array would produce under a clear sky, P_clear(t). It combines Ineichen irradiance, Hay–Davies transposition and an incidence-angle loss.
   - Every model forecasts the clear-sky index K = PV / max(P_clear, 1 kW), which separates clouds from the daily and seasonal cycle.
   - The kW forecast is K̂(t+15) × P_clear(t+15).
   - This normalisation reduced development CRPS by 22%.
2. **Experts.** All are image encoders followed by a GRU, with a quantile or mixture head:
   - **A2:** 3D-CNN over 60 minutes of images (16 frames at 4-minute stride); the ramp expert.
   - **E6-S:** an event-mixture head on the 15-minute change ΔK; sharp in calm conditions.
   - **E4-B-fit:** per-frame CNN over 15 minutes; the direct expert.
3. **Disagreement-guided gate.**
   - w = σ(b₀ + β₁·logit g + β₂·log d), where g is the event probability implied by E6-S's quantiles and d is the gap between the two experts' mean forecasts.
   - The two quantile functions are blended level by level.
   - Gates are fitted by exact blended-CRPS minimisation on held-out calibration days only ([`src/gate7b.py`](src/gate7b.py)).
4. **Routing and fallback.**
   - G3″ (A2 + E6-S) is used whenever a full 60-minute image history exists (93% of test forecasts).
   - Otherwise, mainly just after sunrise, G3 (E4-B-fit + E6-S) is used.
   - The rule uses timestamps only ([`src/final_eval.py`](src/final_eval.py)).
5. **Ensemble and calibration.**
   - 15 members (5 blocked folds × 3 seeds) are averaged quantile by quantile.
   - Intervals are corrected by causal rolling-window conformal prediction, which uses only outcomes observed 15 minutes later.
   - The correction resets daily and is initialised from development out-of-fold scores.
   - No finite-sample coverage guarantee is claimed under temporal dependence.

<details>
<summary><b>Example: a cloudy test day</b> (forecast vs observed PV)</summary>
<br>
<p align="center">
  <img src="reports/figures/F1_test_day_forecast.png" alt="15-minute-ahead G3″-F forecast with 20–80% and 5–95% bands against observed PV and smart persistence on the cloudy test day 2019-08-10" width="820">
</p>

The plot shows the G3″-F median with its raw 20–80% and 5–95% bands, smart persistence and observed PV. The median is smoother than persistence, and the bands widen in variable conditions. Like every model tested, it follows sharp drops and recoveries rather than anticipating them (see [Limitations](#limitations-and-negative-results)).
</details>

Training recipe, expert details, development history and equations: [docs/development_history.md](docs/development_history.md).

## Dataset and preprocessing

**Dataset.** SKIPP'D (SKy Images and Photovoltaic Power generation Dataset), Stanford University, 2017–2019: [DOI 10.25740/dj417rh1007](https://doi.org/10.25740/dj417rh1007), CC BY 4.0.
- **Sky images:** 64×64 RGB fisheye, one per minute.
- **PV:** 1-minute output of a ≈ 30 kW rooftop array.
- **No other inputs:** there are no cloud labels, irradiance sensors or weather variables. The link between clouds and power is learned from the images.

| Split | Records | Days | Forecast samples |
|---|---|---|---|
| Development | 349,372 | 497 | 164,362 |
| Official test (sealed) | 14,003 | 20 (10 sunny, 10 cloudy) | 13,266 |

**Preprocessing**
- **Samples:** each sample uses 16 frames over [t − 15, t] (or 60 min for A2) and the PV target at t + 15 min.
- **Sample matching:** the public benchmark's timestamps drift by up to ±30 s, so frames are matched to the nearest record within ±30 s. This recovers the official sample counts. It is a documented deviation from the benchmark's exact-timing construction ([`reports/protocol_notes.md`](reports/protocol_notes.md), [`reports/dataset_preprocessing_audit.md`](reports/dataset_preprocessing_audit.md)).
- **Development folds:** blocked 5-fold, with 1-day purges and a calibration hold-out.

## Installation and quick start

These steps need Python 3.11. Training needs a CUDA GPU; the work used one NVIDIA RTX 5000 Ada (32 GB).

```bash
git clone https://github.com/abdulsamad2/solar-architecher.git
cd solar-architecher
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pytest                             # unit tests; no data required
```

Then download `2017_2019_images_pv_processed.hdf5`, `times_trainval.npy` and `times_test.npy` from the [Stanford Digital Repository](https://doi.org/10.25740/dj417rh1007) into `data/raw/`, and build the development manifests:

```bash
python -m src.skippd               # forecast manifest (±30 s matching)
python -m src.folds                # blocked 5-fold development folds
pytest -m realdata                 # opt-in real-data integrity audits
```

Trained model weights are **not distributed**. The full training, gating and rehearsal sequence is in [docs/reproducibility.md](docs/reproducibility.md).

> [!WARNING]
> **Sealed test: do not rerun.** `src.final_execute` / `src.final_eval.run_test` is a one-shot evaluation command. It has already been executed. Its guards refuse to run while results exist.

## Reports and figures

**Final test**
- [Gate 5 report](reports/gate5_test_report.md)
- [Pre-registration](reports/test_eval_prereg.md)
- [Raw results](reports/test_results.json)
- [Execution deviations](reports/test_execution_deviations.md)

**Follow-up studies**

| Phase | Pre-registration | Report |
|---|---|---|
| A: comparison with published methods | [pre-registration](reports/phaseA_comparability_prereg.md) | [report](reports/phaseA_report.md) |
| C: calibration under temporal shift | [pre-registration](reports/phaseC_calibration_prereg.md) | [report](reports/phaseC_report.md) |
| F: physics-informed losses | [pre-registration](reports/phaseF_physics_prereg.md) | [report](reports/phaseF_report.md) |

**Other material**
- **Audits:**
  - [Dataset preprocessing](reports/dataset_preprocessing_audit.md)
  - [Leakage](reports/leakage_audit_final.md)
  - [Lag analysis](reports/lag_analysis.json)
- **Literature:** [related work](reports/related_work.md)
- **Figures:** [`reports/figures/`](reports/figures/) contains a test-day forecast, per-day CRPS, interval coverage, the camera change, the PV/clear-sky decline and the Phase F figures. Generate them with `python -m src.figures`.

**Comparison with published methods (Phase A).** G3″-F was compared with published SUNSET, SkyGPT and PV-MM-Diffusion numbers on the SKIPP'D test and on SkyGPT's public external test set.
- **Unpaired:** these are comparisons against published point estimates. The protocols differ in sample matching and CRPS estimation. We therefore do not claim superiority over these methods.
- **Interval scores are comparable, not better.** Our 90% interval score is comparable to PV-MM-Diffusion's.
- **Coverage on SkyGPT's set:** our intervals under-cover there (0.77 at 90%).

Details: [docs/development_history.md §4](docs/development_history.md#4-context-published-results-phase-a).

## Reproducibility and scientific integrity

- **Pre-registration:** hypotheses, metrics and decision rules were committed before the corresponding results existed. Amendments are dated and justified in the same files.
- **Leakage control:** blocked folds with purge days. Gates, thresholds and calibration windows are fitted on development data only. A [leakage audit](reports/leakage_audit_final.md) was completed before unsealing.
- **One-shot sealed test:**
  - it ran once, from a clean committed tree, after an explicit recorded approval;
  - all 140 model files were SHA-256 verified first;
  - raw predictions and metrics were committed unchanged (`cd7c30a`) before any interpretation.
- **Execution deviations:** two aborted attempts preceded the run, and neither read any test PV or labels. Neither fix changed a model or scoring rule ([deviations](reports/test_execution_deviations.md)).
- **Provenance:** see [docs/reproducibility.md](docs/reproducibility.md#4-sealed-test-execution-provenance-only-do-not-run).
- **Result labels:** results are labelled confirmatory, descriptive or exploratory, and post-hoc analyses are never presented as confirmatory.

## Limitations and negative results

- **A single site:** 64×64 images and three years of data.
- **Ramps remain unsolved:**
  - ramp non-inferiority was not established;
  - at nominal 90%, ramp interval coverage was 0.746;
  - misses are directional: on up-ramps 30% of outcomes fall above the 90% interval, and on down-ramps 21% fall below it.
- **No anticipation:** forecast changes follow the observed changes about 15 minutes later (lag-0 correlation ≈ 0 on cloudy days). The gains come from a better forecast level and spread, not from predicting cloud arrivals.
- **Calibration:** the intervals over-covered on the test days (0.935 at 90%), and the daily-reset method is fragile under temporal shift (Phase C).
- **Physics-informed losses:** no practically meaningful improvement (Phase F).
- **Comparisons with published methods are unpaired**, and their protocols differ from ours.
- **Data caveats:**
  - the ±30 s sample matching differs from the benchmark's original construction;
  - the camera was replaced in May 2019;
  - the PV/clear-sky ratio declines by an apparent 2.2%/yr.

  These are documented in the [preprocessing audit](reports/dataset_preprocessing_audit.md).

## Citation

Please cite this repository (metadata in [`CITATION.cff`](CITATION.cff)) together with the dataset:

```bibtex
@software{laghari_shadow_moe_2026,
  author = {Laghari, Abdul Samad},
  title  = {{SHADOW-MoE}: probabilistic 15-minute solar power forecasting from sky images},
  year   = {2026},
  url    = {https://github.com/abdulsamad2/solar-architecher}
}
```

Y. Nie, X. Li, A. Scott, Y. Sun, V. Venugopal, A. Brandt. *SKIPP'D: a SKy Images and Photovoltaic Power Generation Dataset for short-term solar forecasting.* Solar Energy, 2023. Dataset DOI [10.25740/dj417rh1007](https://doi.org/10.25740/dj417rh1007).

## Acknowledgments and licences

- **This repository's code** is released under the [MIT licence](LICENSE).
- **Vendored helper:** [`src/official/relative_op_func.py`](src/official/relative_op_func.py) is © the SKIPP'D authors, MIT licence ([`src/official/LICENSE_SKIPPD`](src/official/LICENSE_SKIPPD)).
- **Data licence:** SKIPP'D is distributed under CC BY 4.0.
- **Acknowledgments:** we thank the SKIPP'D authors at Stanford University for making their data public.

## Research status

- **Done:** the official SKIPP'D test (Gate 5), the published-method comparison (Phase A), the calibration study (Phase C) and the physics-loss study (Phase F).
- **In progress:** a manuscript. It has not yet been peer-reviewed.
- **Nature of the code:** this is research code for reproducing the reported experiments, not a production forecasting system.

Author: Abdul Samad Laghari.
