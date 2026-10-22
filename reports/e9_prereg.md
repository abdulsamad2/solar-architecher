# E9 pre-registration (FROZEN): ramp-aware loss and architecture screen for the 60-min ramp expert

Frozen 2026-10-22, before any E9 code was run on data. It was authorised by the user's instruction to follow the recommended improvement round (#1 ramp-aware loss, #3 small architecture screen).
- **Test set:** sealed; the test-evaluation preparation is paused (`reports/test_eval_prereg_DRAFT.md`).
- **Preserved unchanged:** all previous verdicts (Gates 1–4, E7, E7b, Track B).

## 1. Question
Can the 60-min direct-K ramp expert (D2) be improved, by its training loss or its architecture, so that ramp CRPS falls without failing calm-condition margins? And does a better expert improve the gated forecaster G3′?

## 2. Fixed setup (identical to E8 D2 unless stated)
- **Model setup:** direct K quantiles (9 levels, non-crossing head, climatology init); K-history input; **H60 window** (16 frames, 4-min stride); no solar/motion features (zero-filled columns, as D2).
- **Data:** `--calib-holdout`; **common population** (150,618 validation samples; identical across arms, asserted).
- **Training:** AdamW lr 1e−4 with cosine decay over 15 epochs, clip 1.0, EMA 0.999, batch 128, patience 5; checkpoint selection on validation CRPS (kW).
- **Reference arm A0 = E8 D2** (existing seeds 42/123/456; not retrained).

## 3. Arms (screen)
| Arm | Change vs A0 | Notes |
|---|---|---|
| **L1** | Input-weighted pinball loss | weight w = 1 + 2 · min(v / median_fit(v), 5), where v = mean \|K_j − K_{j−1}\| over the 16 window frames (input-only), median over fit-day samples. Weights depend on inputs only, so the conditional-quantile target is unchanged in principle. |
| **A1** | Wider CNN-GRU | channels 32/64/128/128, image features 128, GRU hidden 128 |
| **A2** | 3D-CNN + GRU | Conv3d blocks (temporal kernel 3, no temporal stride, spatial stride 2) → per-time features → GRU(64) |
| **A3** | ConvGRU | per-frame CNN to an 8×8×32 map; ConvGRU over time (32 channels, 3×3); K history as a constant input channel; final state pooled → 64 |
| **A4** | Temporal transformer | per-frame CNN tokens (64) + K + learned positional embedding → 2 encoder layers (d = 64, 4 heads, FF 128, dropout 0.1) → last token |
| **A5** | Frame-difference input | CNN input = [frame, frame − previous window frame] (6 channels; zero difference for the oldest frame); otherwise A0 |

**Capacity:** arms differ in capacity **by design** (this is an architecture screen). Parameter counts and batch-1 latency are reported for every arm.

## 4. Two-stage protocol
**Stage S (screen): seed 42, 5 folds, 6 arms.**
- **Eligible:** an arm with seed-42 overall CRPS ≤ A0 + 0.02 kW **and** sunny CRPS ≤ A0 + 0.015 kW (point estimates, seed 42).
- **Selected:** among eligible arms, the **two with the lowest seed-42 ramp CRPS** that are also below A0's seed-42 ramp CRPS. Ties go to fewer parameters.
- **If no arm qualifies, E9 stops:** "no improvement found".

**Stage C (confirm): seeds 123 and 456 for the ≤ 2 selected arms.**
- **The Track-A rule vs A0 (D2) with the same seeds.** An arm *improves the ramp expert* iff, in ≥ 2 of 3 seeds:
  - ramp CRPS(X) − ramp CRPS(A0) < 0, with the paired day-bootstrap CI entirely below 0;
  - overall CRPS(X) − overall CRPS(A0) ≤ 0.02 kW;
  - sunny CRPS(X) − sunny CRPS(A0) ≤ 0.015 kW.
- **Disclosure:** seed-42 results contribute to the confirmation rule even though they were also used for selection. This selection effect is disclosed, and seeds 123/456 are reported separately as the clean replication.

**Stage G (gating, CPU):** for the best confirmed arm (lowest seed-mean ramp CRPS), **G3″** = the G3′ recipe (features g, d; calibration days only; experts X + E6-S). It is compared with G3′ (experts D2 + E6-S). G3″ is retained iff, in ≥ 2/3 seeds:
- ramp CRPS(G3″) − ramp CRPS(G3′) < 0, with CI < 0; and
- overall CRPS(G3″) − overall CRPS(G3′) ≤ +0.005.

## 5. Reported regardless
- Raw CRPS overall, sunny, cloudy, ramp and non-ramp (Gate-2 ramp stratum, frozen thresholds); event AUPRC and Brier score; per-fold results.
- Rolling-conformal coverage, width and interval score.
- Fallback population: the arm where H60-eligible, E4-B-fit otherwise; sunrise ramps separately.
- Parameters, latency and training time.
- Screen results for every arm, including non-selected ones.

## 6. Stop conditions (asserted)
- Different evaluation samples across arms.
- Loss weights computed from anything other than in-window inputs and fit-day statistics.
- Non-finite outputs or crossing quantiles.
- Frozen E8 hashes (geometry, eligibility, features) not verified.

## 7. Compute estimate
- **Stage S:** 6 runs (≈ 1.3–2.5 h each; larger arms are slower) in 2 lanes, ≈ 5–7 h.
- **Stage C:** ≤ 4 runs, ≈ 3–4 h.
- **Stage G:** CPU, < 0.5 h.
