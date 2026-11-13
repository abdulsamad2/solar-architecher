# E1 / E4 report: do images add information; history length × encoder

Protocols: `revision/protocols/R-E1_pv_only.md`, `R-E4_history_encoder.md`. Code: `src/revision_e1_e4.py`, `src/revision_posthoc_test.py e1e4`. Raw outputs: `revision/results/e1_e4_dev.json`, `revision/results/posthoc_test_e1e4.json`. New arms: P15 was trained at commit 123da81 with a clean `src`; P60 and 3D@15 min (A2-H15) at commit e252459 with uncommitted changes in the working tree (`src_dirty=True` in their `config.json`; which files were modified was not recorded). `src/train.py` and `src/models.py` are identical in the two commits. (Correction 2026-11-13, found in an independent audit.)

## E1: PV-only probabilistic baseline
P15 / P60: same K input and target, quantile head, budget, folds, seeds and calibration hold-out as E4-B-fit / A2, without the image branch (25,673 / 26,377 parameters vs 90,761 / 213,545).

**Development (seed mean CRPS, kW):**

| Model | Overall | Sunny | Ramp |
|---|---|---|---|
| P15 (PV only, 15 min) | 0.691 | 0.213 | 3.179 |
| E4-B-fit (images + PV, 15 min) | 0.751 | 0.305 | **2.448** |
| G3 (E4-B-fit + E6-S, gated) | **0.590** | 0.124 | 2.545 |
| P60 (PV only, 60 min; common) | 0.737 | 0.241 | 4.223 |
| A2 (images + PV, 60 min; common) | 0.755 | 0.318 | 2.754 |
| G3″ (common) | 0.587 | 0.121 | 2.917 |

- P15 − E4-B-fit: overall −0.054 to −0.063 (CI < 0, 3/3 seeds) — **the PV-only model is better overall and on sunny days on development data**; ramps +0.72 to +0.75 (CI > 0, 3/3) — **much worse on ramps**.
- P60 − A2: same pattern (overall better or equal, ramps much worse in 3/3).
- P15 − G3: +0.098 to +0.106 overall, +0.60 to +0.66 ramps (CI > 0, 3/3) — **the full framework beats the PV-only model everywhere**.
- PV-only ramp recall (median-based) 0.03 vs 0.34 for E4-B-fit; change slope on ramps 0.11 vs 0.32.

**Post hoc test (15 members):** P15 − E4-B-fit = **+0.122 [+0.069, +0.178]** overall, +1.22 [+0.82, +1.65] on ramps; P15 − G3″-F = +0.233 overall; P60 − A2 = +0.208 overall, +2.24 ramps. On the cloudier test days the image models are better overall as well.

**Interpretation (rule in protocol):** the PV-only model is not within 0.02 kW of the image models on ramps in either data set. **Images carry the ramp information.** In calm, sunny development conditions a PV-only model is as good or better than the direct image expert, which is why the calm expert and the gate matter. The manuscript's image-information claim is supported for ramps and cloudy conditions, not for calm conditions.

## E4: history length × encoder (common population, 16 frames)
Parameters: 2D 92,873; 3D 213,545 (capacity mismatch disclosed).

| Seed-mean CRPS (kW) | Overall | Sunny | Ramp | Ramp recall* |
|---|---|---|---|---|
| 2D CNN-GRU, 15 min (E8 D0) | 0.767 | 0.322 | 2.933 | 0.35 |
| 2D CNN-GRU, 60 min (E8 D2) | 0.781 | 0.329 | 2.850 | 0.39 |
| 3D CNN-GRU, 15 min (new) | **0.738** | **0.298** | 2.811 | 0.36 |
| 3D CNN-GRU, 60 min (A2) | 0.755 | 0.318 | **2.754** | 0.37 |

*seed 42.

Effects (per-seed paired day bootstrap; count of seeds with CI < 0):
- **History (60 − 15) within 3D:** ramp 3/3; overall 0/3 (overall slightly worse at 60 min).
- **History within 2D:** ramp 3/3.
- **Encoder (3D − 2D) at 60 min:** ramp 3/3; overall 2/3. **At 15 min:** ramp 3/3.
- **Interaction:** not consistent (seed 42 positive, seeds 123/456 ≈ 0).

**Post hoc test (60-min-eligible samples):** history within 2D −0.113 ramp [−0.155, −0.079]; within 3D −0.218 [−0.299, −0.143]; encoder at 60 min −0.179 [−0.330, −0.040], at 15 min −0.074 [−0.144, −0.005]; overall differences small (CIs include 0).

**Interpretation:** both factors improve ramp CRPS independently and roughly additively; neither improves overall CRPS reliably on its own, and the longer history costs a little in calm conditions. The 60-min 3D expert's ramp advantage is attributable to both the temporal-convolution encoder and the longer history; the manuscript must not attribute it to one of them alone, and must state the capacity difference.
