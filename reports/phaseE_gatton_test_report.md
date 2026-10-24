# Phase E: Gatton one-shot sealed test report (second site)

**Official sealed test:** 21 days, 2020-04-16 to 2020-05-06; 13,311 forecasts (0 excluded for missing PV); 4 days sunny by the frozen E-2 rule. **Executed once.**

**Identifiers:**

| Item | Value |
|---|---|
| Scientific freeze | `dd397403ca485c8100bd5f54c3d55e78ad5ce30c` |
| Approved execution commit | `c15d827d9927edfabfb3f3ff065ed5db0197951b` |
| Approval commit (token `GATTON-FINAL-20261024`) | `9f5164e8f946e92f6be578ef2184673329358a4a` (the run's recorded HEAD) |
| Raw outputs commit (unmodified) | `59ccbcd9f13c536bee494d04ab3b899caa8b1fdc` |

**Execution** (`reports/phaseE_gatton_test_execution_log.json`):
- status completed, 1 invocation, 372.6 s, clean tree at start;
- 144 Gatton + 100 Stanford artifacts verified;
- RTX 5000 Ada, torch 2.11.0, CUDA 12.8.

**Output hashes (SHA-256):**
- `reports/phaseE_gatton_test_results.json`: AF97943A…49BDCD8;
- `experiments/E_gatton/test_predictions_v1.npz`: 7D04F615…95283AE1.

**Deviations:** none (`reports/phaseE_gatton_test_deviations.md`).

## Pre-registered hypotheses
All differences are in kW CRPS. Paired test-day bootstrap: 10,000 resamples, stratified sunny/cloudy, seed 20261022.

| Hypothesis | Estimate [95% CI] | Rule | Verdict |
|---|---|---|---|
| **E1 (sole primary):** F1 G3″-F − F1 E4-B-fit | **−46.9 [−57.4, −37.8]** | CI < 0 | **CONFIRMED** |
| E2: F1 G3″-F − smart persistence | −17.0 [−39.2, +2.5] | CI < 0 | **Not confirmed** (CI includes 0) |
| E3 (descriptive): zero-shot Z0 G3″-F − fine-tuned F1 G3″-F | +480.4 [+391.1, +562.3] | – | Zero-shot much worse |
| E3 (descriptive): Z0 G3″-F − smart persistence | +463.4 [+369.9, +550.5] | – | Zero-shot much worse than persistence |
| E4: F1 G3″-F calibrated 90% coverage within 0.90 ± 0.03 | 0.864 | 0.87–0.93 | **Not met** (under-covers by 3.6 pp) |
| E5 (descriptive): lag | see below | – | No anticipation |
| Secondary: F1 G3″-F − F1 E4-B-fit on ramps | +15.7 [+0.8, +30.7] | – | G3″-F **worse** on ramps |
| Secondary: F0 (from scratch) G3″-F − F1 G3″-F | −1.5 [−6.6, +5.9] | – | No difference |

**Primary answer: yes.** At a second site, after fine-tuning, the disagreement-gated forecaster has lower overall CRPS than its single direct expert: 116.9 vs 163.8 kW (−28.6%), with a CI entirely below 0. This replicates the Gate 5 primary result from SKIPP'D.

## CRPS by stratum (kW)
Sample counts: overall 13,311; sunny 2,504; cloudy 10,807; ramp 846; sunrise ramp 115; rising ramp 435; falling ramp 411. Smart persistence's CRPS equals its MAE.

| Model | Overall | Sunny | Cloudy | Ramp | Sunrise ramp | Rising ramp | Falling ramp | RMSE |
|---|---|---|---|---|---|---|---|---|
| **F1 G3″-F** | 116.9 | 106.2 | 119.4 | 496.3 | 151.7 | 510.9 | 480.8 | **270.2** |
| F1 E4-B-fit | 163.8 | 246.6 | 144.6 | **480.6** | 148.6 | 518.0 | 441.0 | 308.7 |
| F1 E6-S | **96.5** | **19.4** | **114.4** | 628.7 | 167.2 | 650.3 | 605.8 | 292.4 |
| F0 G3″-F | 115.5 | 85.1 | 122.5 | 491.7 | 135.2 | 593.8 | 383.7 | 279.0 |
| Z0 G3″-F (zero-shot) | 597.3 | 849.0 | 539.0 | 626.1 | 291.1 | 1,086.3 | 139.1 | 834.6 |
| Z1 G3″-F (zero-shot, flipped) | 590.2 | 819.0 | 537.2 | 629.0 | 290.6 | 1,089.7 | 141.4 | 820.8 |
| Smart persistence | 134.0 | 28.9 | 158.3 | 936.4 | 305.2 | 896.9 | 978.2 | 315.9 |

**Per day:** F1 G3″-F beats F1 E4-B-fit on 21 of 21 days. It beats smart persistence on 14 of 21: it loses on the 4 sunny days (May 1–4) and on 04-27, 04-30 and 05-05.

## Calibration (continuous causal rolling conformal; F1 G3″-F)

| Nominal | Overall coverage | Sunny | Cloudy | Ramp | Sunrise ramp | Width (kW) | Interval score |
|---|---|---|---|---|---|---|---|
| 50% | 0.475 | 0.564 | 0.454 | 0.137 | 0.183 | 237 | 476 |
| 80% | 0.760 | 0.780 | 0.756 | 0.377 | 0.530 | 434 | 689 |
| 90% | 0.864 | 0.907 | 0.854 | 0.649 | 0.852 | 595 | 868 |

- **Misses at 90%, below / above:**

  | Stratum | Below | Above |
  |---|---|---|
  | Overall | 0.059 | 0.077 |
  | Rising ramps | 0.000 | 0.416 |
  | Falling ramps | 0.282 | 0.000 |

  The directional lag pattern is the same as at SKIPP'D.
- **F1 E4-B-fit at 90%:** coverage 0.809, width 857 kW. G3″-F intervals are 31% narrower at higher coverage.

## E5 lag analysis (frozen procedure, `reports/phaseE_gatton_test_lag.json`)
1-min grid, 21 days (17 cloudy):

| Model | Median best lag (all days) | Corr at 0 (all / cloudy) | Corr at 15 (all / cloudy) |
|---|---|---|---|
| F1 G3″-F | 15 min | 0.160 / 0.034 | 0.866 / 0.887 |
| F1 E4-B-fit | 15 min | 0.200 / 0.055 | 0.568 / 0.493 |
| F1 E6-S | 15 min | 0.154 / 0.024 | 0.978 / 0.979 |
| F0 G3″-F | 15 min | 0.182 / 0.035 | 0.905 / 0.898 |
| Z0 G3″-F | 15 min | 0.107 / 0.016 | 0.791 / 0.833 |
| Smart persistence | 15 min | 0.154 / 0.028 | 0.993 / 0.997 |

As at SKIPP'D, **no model anticipates changes**: on cloudy days the lag-0 correlation is ≈ 0, and forecast changes follow the observed changes 15 minutes earlier.

## Exploratory observations (not pre-registered; for interpretation only)
1. **The calm expert alone (F1 E6-S) has the lowest overall CRPS (96.5 kW)**, below G3″-F (116.9) and below persistence. Its advantage comes from calm and sunny conditions (sunny 19.4 vs 106.2). G3″-F is much better on ramps (496 vs 629).
2. **On the 4 sunny days (May 1–4) the gated model is clearly worse than persistence** (about 86–115 vs 26–34 kW).
   - The ramp expert carries a bias in clear conditions, and the gate still assigns it weight on these days.
   - A plausible reason is that the development period contained only 4 sunny days, so the gate had few clear-sky calibration examples. The late-autumn test period is also seasonally outside development.
   - This is the main reason E2 (beating persistence) was not confirmed.
3. **Zero-shot transfer fails.** The flip ablation (Z1) does not help, so the failure comes from appearance or domain shift (camera exposure and colour), not image orientation.
4. **Pre-training on SKIPP'D gives no measurable benefit** over training from scratch (F0 − F1: −1.5 [−6.6, +5.9] kW).

## Conclusion
- **The pre-registered primary hypothesis replicates at a second, very different site:** the disagreement-gated mixture beats its direct expert by 28.6% CRPS.
- **The model does not reliably beat smart persistence there** (E2 not confirmed): it fails on clear days.
- **Calibrated coverage falls short** of the target (0.864).
- **It does not transfer without site-specific training.**
- **The ramp-lag limitation is unchanged.**
