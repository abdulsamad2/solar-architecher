# Physics-informed training objective: paper-ready methodology subsection

*Results are inserted after the pre-registered ablation completes; nothing below depends on them.*

### X.Y Physics-informed training objective

**Clear-sky index.**
- All experts forecast the clear-sky index K = P / max(P_clear, 1 kW).
- P_clear is the plane-of-array clear-sky power: Ineichen clear-sky irradiance, Hay–Davies transposition and a physical incidence-angle modifier, scaled by an annual harmonic fitted on training days only.
- The PV forecast is reconstructed as P̂(t+15, q) = K̂(t+15, q) · max(P_clear(t+15), 1 kW).
- This identity makes physically motivated penalties on K̂ directly interpretable in power units.

**Objective.**
- We retain the multi-quantile pinball loss L_q as the primary objective. We add three auxiliary terms, plus one optional term, to the training of the image-based ramp expert only; the architecture, the gate and the calibration are unchanged.

1. **Power consistency.** L_phys penalises the mean absolute deviation between observed power and the power implied by the predicted median, P_true − K̂₀.₅ · P_clear(t+15). It is normalised by a fixed scale of 30 kW, the array's approximate nameplate rating. A Huber variant is available.
2. **Soft physical bounds.** L_bound = mean over q of [ReLU(−K̂_q) + ReLU(K̂_q − K_max)], with K_max = 1.5.
   - Values of K̂ above 1 are permitted, since cloud-edge enhancement produces them.
   - Only negative or implausibly large indices are penalised.
3. **Ramp consistency.** L_ramp = mean of w_i · \|ΔK_i − ΔK̂_i\|, with:
   - ΔK = K(t+15) − K(t) and ΔK̂ = K̂₀.₅ − K(t);
   - weights w_i = 1 + α · r_i that up-weight ramp samples;
   - r_i = 1 iff \|ΔK_i\| ≥ τ, where τ is the existing fold-specific ramp threshold (the 90th percentile of \|ΔK\| on the fold's model-fitting days).

   Because K(t) is common to both changes, L_ramp is a ramp-weighted absolute error of the median.
4. **Ramp direction (optional).** L_dir = Σ r_i · ReLU(m − sign(ΔK_i) · ΔK̂_i) / Σ r_i, with margin m = 0.02, penalising a median that moves in the wrong direction during a ramp.

The total loss is L = L_q + λ_phys·L_phys + λ_ramp·L_ramp + λ_bound·L_bound (+ λ_dir·L_dir).

**Hyperparameter selection.**
- The weights were selected by a pre-registered staged search on two development folds (seed 42):
  1. λ_phys ∈ {0.01, 0.05, 0.1, 0.25}, minimising overall CRPS;
  2. λ_ramp ∈ {0.01, 0.05, 0.1, 0.25} and α ∈ {1, 2, 4}, minimising ramp CRPS subject to overall CRPS no more than 1% worse than the baseline;
  3. λ_bound ∈ {0.001, 0.005, 0.01};
  4. optionally λ_dir.
- The selected configurations were then trained with the full protocol: 3 seeds × 5 blocked folds.

**Evaluation.**
- Each variant replaces the ramp expert inside the gated mixture. The gate is refitted on calibration days with the unchanged procedure.
- Forecasts are evaluated out-of-fold on the development period, so that the previously used test sets are not reused for model comparison.
- We report CRPS by condition (overall, sunny, cloudy, ramps, sunrise ramps, rising and falling ramps), point errors, calibrated interval coverage and width, and physical diagnostics:
  - median power-consistency error;
  - the share of K̂ < 0 or K̂ > K_max;
  - ramp-direction accuracy;
  - ramp-magnitude error.
- Differences from the baseline are assessed with a paired, day-block bootstrap stratified by sky condition.
