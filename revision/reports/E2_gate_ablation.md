# E2 report: gate feature and constant-weight ablation

Protocol: `revision/protocols/R-E2_gate_ablation.md`. Code: `src/revision_e2.py`, `src/revision_posthoc_test.py e2`. Raw outputs: `revision/results/e2_gate_ablation_dev.json`, `revision/results/posthoc_test_e2.json`.

All methods blend the same frozen A2 and E6-S predictions; each is fitted on calibration days only (including λ). G3″ reproduces E9 Stage G to 1e−9.

## Development (common 60-min population, 3 seeds × 5 folds; seed mean CRPS, kW)
| Method | Overall | Sunny | Ramp |
|---|---|---|---|
| A2 alone | 0.755 | 0.318 | 2.754 |
| E6-S alone | 0.621 | 0.120 | 3.555 |
| B50: 50/50 | 0.638 | 0.193 | 3.074 |
| BC: CRPS-optimal constant w (0.22–0.39) | 0.616 | 0.152 | 3.248 |
| Gg: event probability only | 0.587 | 0.120 | 2.980 |
| Gd: disagreement only | 0.591 | 0.127 | **2.899** |
| **G3″: g + d (proposed)** | **0.587** | 0.121 | 2.917 |
| G4 (E4-B-fit + E6-S, 4 features) | 0.594 | 0.120 | 3.098 |

**Decision rules:**
- Gate vs constant blending: G3″ better than BC and than B50 on ramps (CI < 0) with overall ≤ +0.005 kW in **3/3 seeds** → **the gate adds value over constant blending.**
- Feature contributions: dropping d (Gg) worsens ramps in 3/3 seeds → d carries ramp information. Dropping g (Gd) does **not** worsen ramps (0/3 seeds; Gd is slightly better on ramps and 0.003 kW worse overall) → **g adds little beyond d.**
- Weights rise monotonically with forecast-time d (mean w on A2 from 0.04 in the lowest d decile to 0.65 in the highest) and with g.

## Post hoc test (60-min-eligible test samples, n = 12,355; 15-member ensembles)
G3″ − method, CRPS (kW) [95% CI]:

| Method | Overall | Ramp |
|---|---|---|
| B50 | −0.030 [−0.042, −0.019] | −0.221 [−0.258, −0.173] |
| BC | −0.038 [−0.053, −0.024] | −0.468 [−0.529, −0.397] |
| Gg | −0.002 [−0.005, +0.002] | −0.052 [−0.079, −0.036] |
| Gd | −0.004 [−0.011, +0.003] | −0.021 [−0.043, +0.012] |
| G4 | −0.019 [−0.043, +0.002] | −0.316 [−0.419, −0.202] |
| A2 | −0.081 | +0.289 |
| E6-S | −0.088 | −0.881 |

Test confirms development: the fitted gate clearly beats constant blending; the two single-feature gates are close to G3″.

## Manuscript implications (C02, C05)
- The demonstrated gating contribution is **forecast-time weighting driven mainly by expert disagreement**. The event-probability feature is not shown to add information beyond disagreement. Disagreement-based gating has precedent (GateFormer, 2026; see the literature report), so novelty must be stated as the configuration, not the feature.
- Add the E2 table (development) and the test row as post hoc.
