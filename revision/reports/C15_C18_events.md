# C15 / C18 report: event probabilities and ramp metrics

Protocol: `revision/protocols/R-C18_events.md`. Code: `src/revision_c18.py`, `src/revision_posthoc_test.py c18`. Raw outputs: `revision/results/c18_events_dev.json`, `revision/results/posthoc_test_c18.json`.

## Two events (C15)
- **Crossing event:** |K(t + h) − K(t)| ≥ τ for some h ∈ 1…15 min (first crossing; may reverse). Scored by P_cross = 1 − P(no event) from E6-S's time-class head.
- **Endpoint ramp:** |K(t + 15) − K(t)| ≥ τ. Scored by g (quantile-implied, used by the gate) and by p_mix (full mixture CDF).
Development base rates: crossing 15.5%, endpoint 10.0%.

**g (9 quantiles) vs p_mix (full mixture CDF):** correlation 0.9992, mean |difference| 0.005; Brier 0.0664 vs 0.0662; AUPRC 0.468 vs 0.469 (seed 42). The 9-quantile approximation used by the gate loses essentially nothing.

## Probability scores (Brier skill vs out-of-fold climatology; AUPRC; ROC AUC)
| Probability → label | Dev BSS (3 seeds) | Dev AUPRC | Dev ROC AUC | Test BSS | Test AUPRC |
|---|---|---|---|---|---|
| P_cross → crossing | 0.411–0.416 | 0.65 | 0.93 | 0.249 | 0.733 |
| g → endpoint | 0.263–0.274 | 0.47 | 0.90 | 0.281 | 0.513 |
| p_mix → endpoint | 0.265–0.275 | 0.47 | 0.91 | 0.282 | 0.513 |
| G3″ weight w → endpoint (as a ramp score) | < 0 | 0.43–0.46 | 0.91 | – | – |

Test climatology = leave-one-test-day-out base rate (development rates are on the fold thresholds, not on the frozen test threshold). Reliability tables and PR curves are in the raw outputs.

**Interpretation:** the event probabilities carry substantial forecast-time information (positive Brier skill, ROC AUC ≈ 0.9) even though the predictive median lags. A lagging median does not mean the distribution has no advance information (reviewer's point confirmed). The gate weight itself is not a calibrated probability (negative BSS) and should not be presented as one.

## Median-based ramp metrics (definitions)
Δ̂ = Q50 − K(t), Δ = K(t + 15) − K(t). Predicted ramp iff |Δ̂| ≥ τ. Recall = TP/(TP + FN) (denominator: observed ramps); precision = TP/(TP + FP) (denominator: predicted ramps); false-alarm rate = FP/(FP + TN) (denominator: observed non-ramps); direction accuracy over observed ramps with Δ̂ ≠ 0; change slope = OLS slope of Δ̂ on Δ for observed ramps. Confusion counts in the raw outputs.

| Model | Recall | Precision | False-alarm rate | Direction accuracy | Change slope (ramps) |
|---|---|---|---|---|---|
| Dev (seed 42): E4-B-fit | 0.335 | 0.561 | 0.029 | 0.791 | 0.322 |
| E6-S | 0.118 | 0.636 | 0.008 | 0.793 | 0.182 |
| A2 (common) | 0.365 | 0.569 | 0.025 | 0.806 | 0.363 |
| G3″ (common) | 0.266 | 0.613 | 0.015 | 0.824 | 0.305 |
| Test: G3″-F | 0.285 | 0.652 | – | 0.879 | 0.340 |
| E4-B-fit | 0.322 | 0.631 | – | 0.852 | 0.336 |
| E6-S | 0.106 | 0.690 | – | 0.825 | 0.190 |
| G4 | 0.216 | 0.690 | – | 0.871 | 0.289 |

High direction accuracy (0.8–0.9) coexists with low recall (0.1–0.35) and change slopes of about 0.3: the median moves in the right direction but by about a third of the observed change.

## Manuscript implications
- Name the two events separately; evaluate each probability against its own label; explain g is from 9 quantiles and is numerically equivalent to the mixture CDF probability.
- Add the event-probability table, the ramp-metric definitions with denominators, and confusion counts (supplement).
