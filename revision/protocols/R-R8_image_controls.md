# Protocol R-R8: capacity-matched and single-frame controls for the image information claim

Committed before any of these networks is trained. Label: **development out-of-fold** (category iii). The official
test set is not used.

## Origin
R-E1 compared PV-only models (about 26,000 parameters) with the direct image expert (90,761 parameters) and found the
image model better on ramps. A simulated review (2026-11-16) objected that this does not separate image information
from model capacity, and that replacing the 15-min image sequence by copies of its last frame did not hurt the
frozen direct expert (R-X1), which suggests that the images inform about the sky state and not about motion. A
literature search found no sky-image study with a capacity-matched or permuted-image control.

## Arms (identical to the direct expert E4-B-fit in architecture, parameter count, target, quantile head, loss,
optimiser, epochs, early stopping, folds, calibration hold-out and seeds 42, 123, 456; trained from scratch)
- **Shuffled images:** each forecast receives the 16-frame image sequence of another, randomly chosen development
  sample (one fixed permutation of all samples, seed 20261116); its own clear-sky-index history is unchanged. The
  network has the capacity and the optimisation of the image expert, but the images carry no information about the
  forecast. (Adaptation of control-task methodology; Hewitt and Liang 2019.)
- **Single frame:** the latest frame repeated 16 times, with the unchanged clear-sky-index history; trained that way
  (R-X1 only perturbed the inputs of a network trained on sequences).
References already trained: E4-B-fit (real sequences) and the PV-only model P15.

## Scores
Out-of-fold validation forecasts of all development samples, per seed: overall CRPS, sunny and cloudy days,
ramp-region and calm-region threshold-weighted CRPS (R-Z2, fold thresholds); paired day-block bootstrap.

## Reading rule (fixed now)
- **Image information, capacity held fixed:** E4-B-fit minus shuffled images. The statement "images add information on
  ramps" is kept only if the ramp-region difference has an interval below zero in 3 of 3 seeds.
- **Capacity:** shuffled images minus PV-only model, reported as the part of the R-E1 difference that capacity and
  optimisation explain.
- **Sequence versus state:** E4-B-fit minus single frame. "The 15-min expert uses cloud motion" is stated only if the
  ramp-region difference is below zero in 3 of 3 seeds; otherwise the manuscript says that at a 15-min history the
  images act mainly through the current sky state.
Everything is reported whichever way it falls.
