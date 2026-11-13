# Protocol R-C18: event probabilities and ramp metrics (C15, C18)

Committed before any run. Labels: **development** (OOF validation), **post hoc** (test).

## Two events (named separately)
- **Crossing event** (E6-S time-class label): |K(t + h) − K(t)| ≥ τ for some h ∈ 1…15 min (first crossing; may reverse).
- **Endpoint ramp**: |K(t + 15) − K(t)| ≥ τ (the event behind g and the gate).
τ = frozen fold threshold (development); frozen test τ (`reports/test_ramp_definition_frozen.json`) for the test ramp stratum.

## Probabilities scored against their matching label
| Probability | Label |
|---|---|
| P_cross = 1 − time_probs[no event] (E6-S head) | crossing event |
| g = quantile-implied P(|ΔK₁₅| ≥ τ) (9 quantiles, used by the gate) | endpoint ramp |
| p_mix = full mixture CDF P(|ΔK₁₅| ≥ τ) (`p_abs_change_ge_tau`) | endpoint ramp |
| gate weight w (G3″) as a ramp score | endpoint ramp |

Metrics: base rate, Brier score, Brier skill vs climatology (fit-day base rate), reliability (10 equal-count bins), PR curve and AUPRC, ROC AUC; confusion counts at 0.5 and at the threshold that equals the base rate.

## Median-based ramp metrics (all point/probabilistic models)
Predicted change Δ̂ = Q50 − K(t); observed Δ = K(t + 15) − K(t).
- Detection: predicted ramp iff |Δ̂| ≥ τ. Recall = TP / observed ramps; precision = TP / predicted ramps; false-alarm rate = FP / observed non-ramps.
- Direction accuracy = share of observed ramps with sign(Δ̂) = sign(Δ) (denominator: observed ramps with Δ̂ ≠ 0).
- Change slope: OLS slope of Δ̂ on Δ (all samples; ramps only).
Confusion counts reported.

## Outputs
`revision/results/c18_events.json` + reliability/PR figure data; report `revision/reports/C15_C18_events.md`.
