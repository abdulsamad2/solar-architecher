# Revision findings

Synthesis of the v5-review analyses. Details and numbers: `revision/reports/`.

## Timing and calibration integrity
- **C07:** the frozen calibration initialisation drew ~47% of its initial scores from development days after each test day. Strictly past-only rerun: 90% coverage 0.926 (frozen 0.935); ramp coverage unchanged (0.746). CRPS results unaffected.
- **C19:** 83–89% of the conformal window is still development scores at the end of each test day; 3- vs 15-member ensembles do not explain the over-coverage (fold-to-fold variation is larger).
- **C06:** the origin is the latest record (no future images), but PV is forward-averaged (available ≈ t + 60 s; effective lead 14.0 min). With all PV inputs taken ≈ 60 s earlier, H1 is unchanged (−0.110 [−0.148, −0.074]).

## What produces the gain
- **E1:** images carry the ramp information (PV-only ramp CRPS +0.72 kW worse on development, +1.22 on test). In calm development conditions a PV-only model beats the direct image expert overall; on the cloudier test days it is worse overall too. The full framework beats PV-only everywhere.
- **E2:** a fitted gate clearly beats constant blending (50/50 and CRPS-optimal constant). Disagreement carries the gain; the event-probability feature adds little beyond it.
- **E4:** a 60-min history and the 3D encoder each improve ramp CRPS; neither alone reliably improves overall CRPS; the longer history costs a little in calm conditions.
- **C16:** relative to the direct expert the gain is mostly stable-condition distribution quality (65% sunny, 85% non-ramp); relative to E6-S and G4 it is entirely on ramps/cloudy, with a small non-ramp cost.

## Forward-in-time (E3, 83 days Aug–Oct 2019, all fitting on earlier data)
- Confirms the gain over the direct expert (−0.113) and persistence (−0.291).
- Does not show superiority over the calm expert (−0.003, n.s.) or G4 (+0.007, G4 slightly better) in this mostly calm period; better than both on ramps.
- Continuous rolling conformal and ACI reach 0.90 at 90%; daily reset under-covers (0.88). Ramp coverage remains 0.52–0.60 under every method.

## Ramps and events (C18)
- Event probabilities carry advance information (Brier skill 0.25–0.42, ROC AUC ≈ 0.9) although the median lags (median-change slope ≈ 0.3; recall 0.1–0.35; direction accuracy 0.8–0.9).

## Literature (C02, C21)
- Disagreement-based gating has precedent (GateFormer, 2026). The defensible contribution is the configuration. Published SKIPP'D probabilistic results are on a different 5-day test set with m = 10/30/100 standard estimators; at matched m = 100 our sample CRPS is 2.454 vs SkyGPT 2.81 (contextual).

## Latency (C22)
- Batch-1 end to end ≈ 333 ms median (p95 ≈ 393 ms); 5.94 M resident parameters.
