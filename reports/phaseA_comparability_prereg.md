# Phase A: comparability with published results (pre-registration, FROZEN before execution)

2026-10-23. **Models, thresholds, calibration settings and scoring are the frozen Gate 5 versions. Nothing is retrained, refitted or re-selected.** This phase only places the frozen models next to published numbers. Its outcome cannot change the Gate 5 verdicts.

## A1: SKIPP'D 20-day test, scored by the SKIPP'D paper's metrics (re-scoring committed predictions)

**Inputs:** the committed one-shot predictions (`experiments/final/test_predictions_v1.npz`, hash in `reports/test_raw_output_hashes.json`). No new inference is run.

**Published reference:** SKIPP'D paper (Nie et al., arXiv 2207.00913), Table 5.1, SUNSET forecast on the same 20 test days.

| Stratum | RMSE (kW) | MAE (kW) | Forecast skill |
|---|---|---|---|
| Sunny | 0.61 | 0.50 | −45.80% |
| Cloudy | 4.27 | 2.95 | 17.03% |
| Overall | 3.03 | 1.71 | 16.44% |

Forecast skill = 1 − RMSE / RMSE(smart persistence).

**Analyses (descriptive):**

1. **Point metrics of every frozen model.** For G3″-F, E4-B-fit, E6-S, G4, the SUNSET port and smart persistence:
   - RMSE and MAE of the median, and RMSE-based forecast skill vs smart persistence;
   - by sunny / cloudy / overall, on our 13,266 samples.
2. **Exact-timing subset.** Restrict to samples whose 16 history frames and target lie **exactly** at t−j min and t+15 min (0 s offset). This is the official matching rule, applied to the benchmark records. All analysis 1 metrics are repeated on this subset.
3. **Protocol-gap estimate.** Compare our SUNSET port with the published SUNSET numbers in each stratum. The gap measures how far our sample construction moves the numbers.
4. **Paired day bootstrap** (frozen procedure: 10,000 resamples, stratified sunny/cloudy, seed 20261022) for G3″-F − SUNSET port and G3″-F − persistence, in RMSE-skill terms and in MAE, on both populations.

**Reporting rule:**
- A comparison with *published* SUNSET is stated as "under a closely matched protocol" only if our SUNSET port is within 0.05 kW of the published overall RMSE and MAE on the exact-timing subset.
- Otherwise it is reported as protocol-mismatched.

## A2: head-to-head with SkyGPT on its own test days (new sealed external test, one shot)

**Data:** SkyGPT's public extra test set (`test_set_2019nov_dec.hdf5` + `times_curr_test_2019nov_dec.npy`, Google Drive folder linked from <https://github.com/yuhao-nie/SkyGPT>).
- It covers 5 cloudy days in Nov–Dec 2019 at the same site.
- Our development data ends on **2019-10-26**, so these days were never seen in training, gate fitting or calibration.
- **STOP** if any sample date is ≤ 2019-10-26 or falls on a SKIPP'D test day.

**Published reference** (SkyGPT, arXiv 2306.11682, Table 1, test set; CRPS and Winkler scores in kW):

| Model | CRPS | Winkler score (90%) |
|---|---|---|
| Smart persistence | 3.67 | – |
| SUNSET | 3.31 | 56.95 |
| SkyGPT | 2.81 | 26.70 |

These values were verified verbatim against the paper's Table 1 (arXiv 2306.11682v1, p. 17).

**SkyGPT's test population, verbatim from its Section 4:**
- "5 cloudy days from 2019 November to December", 2,582 samples.
- Samples are restricted to cloudy-screened samples (a modified normalised red-blue ratio cloud-pixel detector), stepped every 2 minutes, and require complete 1-min image and PV records from t−15 to t+15.

**SkyGPT's metric definitions:**
- **CRPS:** the empirical predictive-CDF CRPS over the model's ensemble predictions (100 for SkyGPT, 10 for SUNSET).
- **Winkler score:** α = 0.1, with the 5th / 95th percentiles of the predictions as bounds.
- **Smart persistence:** the clear-sky index held constant, using the same clear-sky formula and constants as our official persistence (A_e 24.98 m², tilt 22.5°, azimuth 195°).

**Image masking:** SkyGPT's PV models mask the black background outside the fisheye circle. Our frozen models were trained on unmasked SKIPP'D images and are run on the images exactly as stored in the file, with no masking. If the file's images are already masked (background exactly 0 where SKIPP'D backgrounds are not), this is reported as an input-distribution difference.

**Frozen models evaluated** (15-member ensembles, frozen checkpoints and gate parameters, frozen clear-sky anchors per member):
- **G3** (E4-B-fit + E6-S, frozen E7b gate). The file holds only 16 frames over 15 minutes, so no 60-min history exists for any sample. G3″-F's frozen routing rule therefore outputs G3 for **every** sample: here G3 *is* G3″-F, by construction.
- E4-B-fit, E6-S and G4 as comparators.
- The SUNSET port (10 frozen fold models).
- Smart persistence (the official function).
- A2 and D4 cannot run (they need 60-min history) and are not reported.

**Inputs:**
- `images_log` (16 frames, t−15…t, oldest first, matching our model input order);
- `pv_log` (16 PV values) for the clear-sky-index history;
- the timestamps file for t;
- target = `pv_pred[:, 14]` (t + 15 min).

The file layout is checked on load (shapes, value ranges, chronological order). Any mismatch with the README description → **STOP before any model is run**.

**Metrics** (following SkyGPT's definitions where they exist):
- **CRPS (kW)** of each probabilistic model. Ours use the frozen quantile-interpolation CRPS. For the SUNSET port, the empirical ensemble CRPS over its 10 fold predictions, as SkyGPT computed it for SUNSET; its point-ensemble MAE is reported too.
- **Winkler score (90%)** from the raw 5% / 95% quantiles, as SkyGPT computed it, and also after the frozen causal rolling conformal calibration (daily reset, initialised from development scores nearest in time, frozen W).
- **CRPS forecast skill** vs smart persistence.
- **Also reported:** RMSE and MAE of the median, 50/80/90% coverage and interval width (raw and calibrated), ramp-stratified CRPS (frozen τ_test definition and anchor), and per-day CRPS.

**Comparison and decision rule (descriptive):**
- **Primary comparison:** G3 (= G3″-F) CRPS on these samples vs SkyGPT's published 2.81 kW.
- SkyGPT's per-sample predictions are not public, so no paired test is possible. The comparison is a point-estimate statement, with our day-bootstrap CI shown for our model only.
- **Population check:** our smart persistence CRPS on the file must be within ±5% of the published 3.67 kW. If not, the populations may differ, and every SkyGPT comparison is labelled "population not verified".
- Paired day bootstrap (frozen procedure) among our own models and persistence. Only 5 days are available, so these CIs are wide and stated as such.

**One-shot rule:**
- The evaluation runs once, from a clean committed tree, via a guarded runner that refuses to run if results exist.
- Raw predictions and results are committed before interpretation.
- Failures before labels are read may be fixed and documented as deviations. After labels are read, only identical-code reruns are allowed.

## What Phase A cannot claim
- No new state-of-the-art claim on SKIPP'D. A1 is a re-scoring of an already-reported test.
- A2 compares against published numbers, not per-sample predictions. It is evidence, not a paired significance test.
- A2 tests G3″-F only in its 15-minute fallback configuration (G3). It says nothing about the 60-min A2 branch.
