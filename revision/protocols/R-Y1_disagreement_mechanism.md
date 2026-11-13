# Protocol R-Y1: does expert disagreement explain the probability-mixture advantage?

Committed before any run. Labels: **development** (primary), **post hoc** (test).
Origin: R-X2 found that a gated probability mixture (linear pool, LP) of the 60-min ramp expert (A2) and the calm-condition expert (E6-S) scored about 1% better than the frozen gated quantile blend (G3''), mainly on ramps. Scoring-rule theory (Knüppel & Krüger 2022; Krüger 2024, Prop. 5.1) gives, for the CRPS and a two-component pool with weight w,

CRPS(LP_w, y) = w CRPS(A, y) + (1 - w) CRPS(R, y) - w (1 - w) D(A, R),

where D(A, R) = E|A - R| - (E|A - A'| + E|R - R'|) / 2 is the Cramér (CRPS) divergence between the two expert distributions. The quantile blend has no such disagreement term.

## Questions
1. Identity check: does the decomposition hold numerically on our atom representation (sanity, not a finding)?
2. Mechanism: does the per-forecast advantage of LP over the quantile blend, Δ = CRPS(G3'') - CRPS(LP-frozen) (same weights w), increase with the disagreement term w(1 - w) D?
3. Does D explain why the advantage is concentrated on ramps?

## Frozen inputs
Exactly as R-X2 (`src/revision_x2_linear_pool.py`): A2 and E6-S per seed x fold `preds_val.npz` (development, common 60-min population, out-of-fold), frozen G3'' gate weights, 200-atom grid quantile functions (`gate7b.grid_quantiles`), exact atom CRPS (`MixCRPS`, `atom_crps`). Post hoc test: per-member test dumps (`experiments/R_posthoc`), 60-min-eligible test samples, per-member atom CRPS averaged over the 15 members.

## Analysis
- Identity: max |LP CRPS - (weighted component CRPS - w(1-w)D)| over all samples; must be < 1e-9 (else STOP).
- Primary: per seed, decile bins of the disagreement term w(1 - w)D (bin edges from that seed's development samples). Report mean Δ per decile, the Spearman correlation between decile index and mean Δ, and the paired day-block bootstrap CI of (mean Δ in the top decile - mean Δ in the bottom decile).
- Secondary: the same with the gate feature d (absolute difference of expert means) instead of D; share of the total LP advantage contributed by the top 20% of D; mean D on ramp vs non-ramp samples.
- Post hoc test: the same decile table and top-minus-bottom bootstrap on the official test.

## Interpretation rule (fixed now)
The mechanism is **supported** if, in >= 2 of 3 seeds, the Spearman correlation is > 0.8 and the top-minus-bottom CI lies above 0, and the post hoc test shows the same sign. It is **not supported** if the correlation is <= 0.5 or the CI includes 0 in >= 2 of 3 seeds. Anything else is reported as **partly supported**. Reported whichever way it falls.
