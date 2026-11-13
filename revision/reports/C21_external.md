# C21 report: harmonised external comparison (contextual, unpaired)

Protocol: `revision/protocols/R-C22_C21_latency_external.md`. Code: `src/revision_c21.py`. Raw output: `revision/results/c21_external_crps.json`. Literature basis: `revision/reports/PV nowcasting novelty and comparators.md`.

## Why the published values are not directly comparable
- No published probabilistic SKIPP'D method reports results on the official 20-day test set. SkyGPT (Nie et al. 2024) and PV-MM-Diffusion use SkyGPT's own external set: 2,582 samples on 5 cloudy days (Nov–Dec 2019).
- They use the standard empirical-ensemble CRPS with different sample counts: SkyGPT m = 100; PV-MM-Diffusion m = 30; SUNSET and the real-future-image oracle m = 10. The standard estimator is biased upward by about E|X − X′| / (2m), so values at different m are not like-for-like.
- Published Winkler scores are raw 5–95% sample intervals (no calibration).

## Our forecasts at matched m (inverse-transform samples from the quantile function; 100 repeats; SD ≤ 0.003)
SkyGPT external set (n = 2,582; all samples routed to the G3 fallback because none has a 60-min history):

| Model | Exact quantile CRPS | m = 10 standard | m = 30 standard | m = 100 standard | Fair (any m) |
|---|---|---|---|---|---|
| G3″-F | 2.455 | 2.597 | 2.494 | 2.454 | 2.437–2.438 |
| E6-S | 2.433 | 2.585 | 2.472 | 2.433 | 2.414–2.415 |
| G4 | 2.469 | 2.605 | 2.503 | 2.467 | 2.450–2.451 |
| E4-B-fit | 2.571 | 2.699 | 2.602 | 2.566 | 2.550–2.551 |
| SUNSET port (10 real members) | – | 3.271 | – | – | 3.229 |

Published (same file, standard estimator): SkyGPT 2.81 (m = 100); PV-MM-Diffusion oracle row 2.52 (m = 10; author confirmation pending); SUNSET 3.31 (m = 10).

Official test set (n = 13,266): G3″-F exact 1.006; m = 10 / 30 / 100 standard 1.104 / 1.038 / 1.015; fair 1.004–1.005.

## Interpretation
- At matched m = 100, G3″-F's sample CRPS (2.454) is below SkyGPT's 2.81; at m = 10 it (2.597) is above the oracle row's 2.52. The exact and fair values (≈ 2.44) are not comparable with published standard-estimator values at small m.
- The fair estimator is m-independent and equals the exact quantile CRPS to within 0.02 kW; it should be the primary number for any future comparison.
- These comparisons are unpaired, on a different test set, with every external forecast from the fallback branch; they are context, not a ranking. No state-of-the-art claim.
