# Protocol R-H0: pilot - is image resolution the bottleneck of 15-min ramp forecasts?

Committed before any training run of this pilot. Label: **development** (out-of-fold, category iii). The official test
set and the untouched raw days of November-December 2019 are not used.

## Origin
The forecast median lags observed changes by about 15 min ("very smart persistence"), and the manuscript suspects a
perception bottleneck: the benchmark images are 64 x 64 (about 3 degrees of sky per pixel). The raw SKIPP'D release
contains the same camera frames at 2048 x 2048. Feasibility checks made before this protocol (no model trained):
raw frames match the benchmark days; the sun can be located from solar geometry with a fitted fisheye model (median
error 36 px of 2048 on a clear day, within 100 px on cloudy days); raw image archives total about 144 GB.

## Critical prior (stated before the pilot)
Geometry limits what a sun-centred crop can add at a 15-min lead: a cloud that will shade the array in 15 min is now
several kilometres upwind, i.e. far from the sun in the image for low clouds (at 1.5 km height and 7 m/s it is about
75 degrees away), where the fisheye compresses the sky into the outer ring (zenith angles 60-90 degrees occupy about
10 px of radius at 64 x 64, about 330 px at 2048). The pilot therefore tests resolution over the whole sky first and
the sun region second. A null result is plausible and will be reported.

## Data
Benchmark development samples (frozen manifest and blocked folds) of the pilot months 2017-05, 2017-09, 2017-10,
2017-11, 2017-12, 2018-01, 2018-02, 2018-03, 2018-11, 2019-01, 2019-02, 2019-05 (chosen for ramp counts and download
size before any training). Raw frames are matched to benchmark records by timestamp (within 30 s); samples without a
complete raw history are dropped for every arm. Inputs derived from raw frames:
- **F256:** the full frame resampled to 256 x 256 (area averaging).
- **SUN:** a 512 x 512 raw window centred on the sun position computed from solar geometry (no image search),
  resampled to 64 x 64 (about 0.7 degrees per pixel).

## Arms (direct 15-min expert; same clear-sky-index target, quantile head, PV inputs, optimiser, epochs, early stopping
and folds as E4-B-fit; all arms retrained on the pilot subset; seed 42, five folds, out-of-fold predictions)
- **B64:** benchmark 64 x 64 frames (reference).
- **C256 (control):** the benchmark 64 x 64 frames upsampled bilinearly to 256 x 256, in the network of H256.
- **H256:** F256 frames. Same network as C256 (the E4-B-fit encoder with stride/pooling adapted to 256 x 256 input so
  that the feature map entering the head has the same size; fixed before training).
- **S64 (secondary):** B64 plus a second encoder branch of the same design on the SUN window.
- **CS64 (control for S64):** as S64, but the window is cut from the 64 x 64 benchmark frame and upsampled.
Each control has exactly the architecture and parameter count of its treatment and differs only in the information
content of the input.

## Metrics (development out-of-fold, common samples)
Primary: ramp-region threshold-weighted CRPS (R-Z2 definition, fold thresholds). Secondary: CRPS, calm-region part,
median lag (lag of maximum cross-correlation between median and observed changes) and ramp-magnitude ratio.
Paired day-block bootstrap (10,000 resamples) of treatment minus control.

## Decision rule (fixed now)
**GO** for a resolution arm if treatment minus control (H256 - C256, or S64 - CS64) has a ramp-region twCRPS
difference whose 95% CI lies below 0, the relative improvement is at least 3% of the control's ramp-region score, and
the overall CRPS is not worse by more than 1%. Otherwise **NO-GO** for that arm.
- On GO: a full protocol (R-H1) is written and committed before further training: all months, three seeds, both expert
  types, gate refitted, and a frozen confirmatory evaluation on raw days of November-December 2019 that are not in
  the benchmark and were not used in the external comparison of the manuscript.
- On NO-GO for both arms: the pilot is reported as a negative finding (resolution was not shown to be the bottleneck
  at a 15-min lead) and no further resolution training is done.
Pilot numbers are never reported as confirmatory.
