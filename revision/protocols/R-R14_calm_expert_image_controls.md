# Protocol R-R14: image controls of the calm-condition expert

Committed before any training of this protocol is started. Date: 2026-11-17. Labels: **development out-of-fold**
(category iii); explanatory. Development is closed: **this protocol does not select or change the method.** The calm
expert of the method stays E6-S with real images whatever the result. The official test set, the forward period and
the external days are not touched.

## Origin
R-R8 retrained the direct expert (level target, quantile head) with images shuffled across samples and found that
real images lower the ramp-region part of the CRPS and raise the calm-region part at fixed capacity. The manuscript
generalises this to "images hurt in calm conditions". The calm expert of the method has another target (the change
of the clear-sky index) and another head (event mixture), and for it the question has not been measured: no
shuffled-image calm expert and no PV-only network with the change target exist.

## Known before the run
R-R8 (direct expert: real - shuffled, ramp region -0.022 to -0.027 kW, calm region +0.021 to +0.047 kW, three seeds),
R-R6 (target against head), R-E1 (PV-only network with the level target), R-M1 (mixtures). Nothing about the two
arms below.

## Arms (each: seeds 42, 123, 456 x five folds = 15 trainings; recipe of E6-S unchanged)
Recipe of E6-S (`experiments/E6/v1/E6-S/csi/seed42/config.json`): target csi, input csi, head event (supervised,
two components), frozen thresholds `reports/phase6_frozen_thresholds.json`, calibration hold-out, encoder cnn_gru,
AdamW lr 1e-4 cosine, batch 128, at most 15 epochs, patience 5, clip 1, EMA 0.999, same manifest and folds.

- **C-shuf**: E6-S with the images of another, randomly chosen sample (the fixed permutation of R-R8, seed
  20261116); own clear-sky-index history. Same network, capacity and training budget as the calm expert; no image
  information.
- **C-pv**: E6-S with the PV-only encoder of R-E1 (`pv_gru`, no image branch); same target, head and budget.

At most three trainings run at the same time. Output: `D:/skippd_hires/R_R14/{shuffled,pvonly}`.

## Evaluation (development out-of-fold, per seed, same forecast origins; paired day-bootstrap intervals)
1. Single models: CRPS, ramp-region and calm-region parts, sunny and cloudy days, for E6-S, C-shuf, C-pv (and, for
   reference, the direct expert and its shuffled control from R-R8). Contrasts: E6-S - C-shuf (image information at
   fixed capacity), C-shuf - C-pv (capacity and optimisation), E6-S - C-pv.
2. As calm component of the method: the gated probability mixture M-RC of the 60-min ramp expert with each of the
   three (gate refitted on the calibration days by the rule of R-M1; common 60-min population). Contrasts:
   mixture with E6-S - mixture with C-shuf, - mixture with C-pv. The mixture with E6-S must reproduce R-M1 (STOP
   otherwise).

## Reading rules (fixed now)
- "Images hurt in calm conditions" may be stated for the calm expert only if E6-S - C-shuf has a calm-region
  interval **above** zero in three of three seeds. If the interval is below zero in three of three seeds, the
  manuscript states that the calm expert gains from images in the calm region, and the R-R8 finding is restricted to
  the level-target network. Otherwise: "no effect of images on the calm expert in the calm region is shown".
- "The calm expert uses image information on ramps" is stated only if E6-S - C-shuf has a ramp-region interval below
  zero in three of three seeds.
- "The calm expert needs the camera" (as a component of the method) is stated only if the mixture with E6-S is
  better than the mixture with C-shuf and than the mixture with C-pv, overall, with interval below zero in three of
  three seeds. If neither difference is shown, the manuscript states that a PV-only calm component is not shown to
  be worse, names this as a simplification for future work, and does not change the method.

All results are reported whichever way they fall. Result file: `revision/results/r14_calm_expert_image_controls.json`.
Scripts: `src/revision_r14_calm_controls.py` (training), `src/revision_r14_eval.py` (evaluation).
