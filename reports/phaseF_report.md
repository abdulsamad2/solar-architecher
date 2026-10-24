# Phase F report: physics-informed training objective (development, out-of-fold)

2026-10-24.
- **Protocol:** `reports/phaseF_physics_prereg.md` (c19998f).
- **Search:** `configs/physics/selection.json`.
- **Results:** `reports/phaseF_results.json`, `reports/phaseF_table.md`, `reports/phaseF_bootstrap_all.json`.
- **Figures:** `reports/figures/F_phys1–5`.

**Scope.** All results are development out-of-fold G3″-F forecasts: 164,362 samples, 497 days, 3 members per sample, none trained on the sample. The official test sets were **not** reused. The G3″ gate was refitted for each arm with the frozen procedure.

**Pipeline check.** The A0 rebuild reproduces the previously reported development figures exactly (CRPS 0.5725, ramp 2.411, 90% coverage 0.901). It matches the independently recomputed stream to within 1.8e−4 kW.

## Selected hyperparameters (pre-registered staged search, seed 42, folds 1 and 3)
- **λ_phys = 0.01.** Every λ_phys changed the expert's search-fold CRPS by +0.1% to +0.7%; none improved it.
- **λ_ramp = 0.1, α_ramp = 2.** Every λ_ramp raised overall CRPS by 0.9–3.5%. Ramp CRPS changed by −0.1% to +1.7%.
- **λ_bound = 0.001**, selected under its constraint.
- **λ_direction = 0.01**, **flagged**: no direction candidate met the "overall CRPS ≤ 1.01 × A0" constraint (search-fold overall +1.0% to +2.0% over A0). The smallest grid value was therefore used, per the pre-registered fallback rule.

## Main result

**Development out-of-fold G3″-F, kW** (bold = best, italic = second best; for coverage, closest to 90%):

| Method | Overall CRPS | Cloudy CRPS | Ramp CRPS | RMSE | MAE | 90% coverage | Ramp coverage | Sunrise coverage |
|---|---|---|---|---|---|---|---|---|
| Original G3″-F | *0.573* | *0.894* | **2.411** | *1.914* | *0.775* | 0.901 | 0.651 | **0.604** |
| + Physics | **0.572** | **0.893** | *2.414* | **1.913** | **0.773** | *0.901* | 0.650 | *0.602* |
| + Ramp | 0.575 | 0.899 | 2.419 | 1.918 | 0.775 | 0.901 | *0.655* | 0.598 |
| + Physics + Ramp | 0.575 | 0.900 | 2.421 | 1.919 | 0.775 | 0.901 | 0.653 | 0.601 |
| + Physics + Ramp + Bounds | 0.575 | 0.900 | 2.425 | 1.919 | 0.775 | 0.901 | 0.654 | 0.600 |
| + … + Direction | 0.575 | 0.899 | 2.420 | 1.918 | 0.775 | **0.900** | **0.658** | 0.599 |

**Pre-registered selection rule:** lowest overall CRPS among arms whose ramp CRPS is no worse than A0's. **No arm qualified**: every physics arm has a slightly higher ramp CRPS. The primary comparison therefore has no selected arm. The bootstraps below are descriptive.

**Paired day bootstrap vs the original** (arm − original; 95% CI; negative CRPS = better):

| Arm | ΔCRPS overall | ΔCRPS cloudy | ΔCRPS ramp | Δ ramp 90% coverage |
|---|---|---|---|---|
| + Physics | −0.0008 [−0.0015, −0.0002] | −0.0011 [−0.0022, −0.0001] | +0.0031 [+0.0005, +0.0057] | −0.0008 [−0.0028, +0.0011] |
| + Ramp | +0.0025 [+0.0010, +0.0041] | +0.0054 [+0.0028, +0.0081] | +0.0082 [+0.0024, +0.0143] | +0.0040 [+0.0014, +0.0067] |
| + Physics + Ramp | +0.0027 [+0.0013, +0.0042] | +0.0058 [+0.0033, +0.0084] | +0.0096 [+0.0042, +0.0153] | +0.0027 [+0.0001, +0.0054] |
| + … + Bounds | +0.0028 [+0.0014, +0.0043] | +0.0060 [+0.0037, +0.0085] | +0.0135 [+0.0076, +0.0196] | +0.0032 [+0.0004, +0.0059] |
| + … + Direction | +0.0023 [+0.0008, +0.0040] | +0.0054 [+0.0027, +0.0082] | +0.0086 [+0.0028, +0.0148] | **+0.0078 [+0.0047, +0.0109]** |

**Conditional effects** (Figure F_phys2):
- Ramp-loss arms improve **rising-ramp** CRPS by about 1.4% but worsen **falling-ramp** CRPS by about 2.1%.
- Sunrise-ramp results are identical across arms, by design: sunrise samples are served by the unchanged 15-min fallback G3.

## Physical diagnostics

| Arm | Median \|P − P̂₅₀\| (kW) | % K̂ < 0 | % K̂ > 1.5 | Ramp-direction accuracy | Ramp-magnitude error (K) |
|---|---|---|---|---|---|
| Original | 0.159 | 0.31% | 0.87% | 82.2% | 0.344 |
| + Physics | 0.158 | 0.32% | 0.87% | 82.4% | 0.344 |
| + Ramp | 0.155 | 0.34% | 0.87% | 83.2% | 0.342 |
| + Physics + Ramp | 0.155 | 0.34% | 0.86% | 83.0% | 0.342 |
| + … + Bounds | 0.154 | 0.34% | 0.85% | 83.0% | 0.342 |
| + … + Direction | 0.154 | 0.35% | 0.85% | **83.5%** | 0.341 |

- Physically implausible predictions were already rare: under 1% of quantile values. The bound loss barely changes them.
- The ramp terms slightly improve direction accuracy (82.2% → 83.5%) and the median power error. They leave ramp magnitude essentially unchanged: predicted ramps remain heavily damped (slope about 0.32, Figure F_phys4).

## Interpretation
1. **Within this architecture and data, the physics-informed objective does not materially change forecast quality.**
   - Every CRPS difference is ≤ 0.6% of the baseline.
   - The physics-consistency term alone gives a statistically detectable but negligible improvement: −0.14% overall, with ramp CRPS +0.13%.
   - The ramp terms cost 0.4–0.6% of CRPS.
2. **The ramp terms buy a small calibration gain during ramps.** Ramp 90% coverage rises by up to +0.8 percentage points (direction arm; CI +0.5 to +1.1 pp). Ramp-direction accuracy rises by about 1.3 pp. Both come at a small CRPS cost. This is a measurable but practically minor shift, not an improvement of the dynamical regime.
3. **The terms reshape, rather than improve, the ramp response.** Up-ramps improve and down-ramps worsen.
4. **Consistent with the lag analysis**, these results indicate that the model's limit during ramps lies in *perceiving* approaching clouds, not in the training objective. The images contain too little usable information about cloud arrival for a loss function to recover it. Future gains should come from the inputs: explicit cloud motion, higher-resolution views around the sun, and longer or upwind context.

**Limits.**
- Physics terms were applied only to the 3D-CNN expert. E6-S is not a quantile model, and the sunrise fallback is unchanged.
- The hyperparameter search used one seed and two folds.
- The results are development out-of-fold, not a sealed test.
