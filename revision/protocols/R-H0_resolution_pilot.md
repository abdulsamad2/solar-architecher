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

---

# Amendment 1 (2026-11-15, before any training run; no model output has been seen)

Reason: a literature review made after the original text (three independent searches) changed the design. It replaces
the sections "Data" (derived inputs), "Arms", and the hypothesis; metrics, bootstrap and the GO threshold are unchanged.

## What the review established
- No study was found that crosses image information with lead time under matched controls and a proper ramp score.
  Closest: Feng et al. 2022 (NREL irradiance, 64/128/256 x 10-60 min, average error only, capacity not matched;
  128 best, 256 "redundant"); Paletta et al. (SPIN, arXiv:2111.14507: a sun close-up helps at 2 min but loses to a wide
  sun-centred view by 10 min); Wang & You 2026 (arXiv:2605.03165: SKIPP'D ramp detection by lead bin with a
  real-future-image oracle, raw frames downscaled to 128, own ramp-filtered split); SolarBench 2026
  (arXiv:2609.06187: names the attribution of the ramp deficit as open).
- The default explanation of the lag in the literature is a model limit (loss smoothing, reactive mapping: Paletta
  et al. 2021, 2022; SkyGPT locates the error in the image-to-power mapping). The information-limit argument is made
  physically by Julian et al. (arXiv:2409.12162: fisheye compression near the horizon) and by the field-of-view
  geometry of Schmidt et al. 2015 and Wang, Kurtz and Kleissl 2016, without a matched test.

## Hypothesis (fixed now)
If the ramp deficit is an **information limit set by viewing geometry**, added image information should help where the
relevant cloud is: near the sun for short leads and near the horizon for long leads. Prediction (double dissociation):
the gain of sun-region detail is larger at 5 min than at 15 min, and the gain of horizon detail is larger at 15 min
than at 5 min. If the deficit is a **model limit**, added information should help similarly at both leads or not at
all. Both outcomes are reported.

## Lead times
15 min (the benchmark task) and 5 min. The 5-min samples use the same forecast origins, histories and folds, with the
target record at origin + 5 min (within 30 s); the 5-min ramp threshold is derived by the frozen rule of the 15-min
thresholds applied to 5-min changes on the fit days of each fold. Models are trained separately per lead.

## Inputs derived from raw frames (information levels; tensor sizes fixed so that each network sees one size)
- **FULL:** the full frame as a 256 x 256 tensor at three information levels: I64 (benchmark frame, bilinear
  upsampling; control), I128 (raw frame area-averaged to 128, then upsampled), I256 (raw frame area-averaged to 256).
- **SUN:** a 512 x 512 raw window centred on the sun position from solar geometry, as a 64 x 64 tensor: high (from
  the raw frame) or control (the same window cut from the benchmark frame and upsampled).
- **RING:** the outer sky ring, zenith angles 55-88 degrees, unwrapped to azimuth x zenith (256 x 32 tensor, azimuth
  measured from the sun's azimuth): high (from the raw frame) or control (from the benchmark frame). The fisheye model
  is fitted once on clear-sky frames of several seasons before any training.

## Arms (direct expert; same target, quantile head and pinball loss, PV inputs, optimiser, epochs, early stopping, folds;
seed 42; five folds; pilot months only)
- **B64:** benchmark frames, the E4-B-fit network (reference, not a control).
- **N-FULL** x {I64, I128, I256}: one network for 256 x 256 input (E4-B-fit encoder with extra stride so that the
  feature map entering the head is unchanged).
- **N-SUN** x {control, high}: B64 plus a second encoder branch on the SUN tensor.
- **N-RING** x {control, high}: B64 plus a second encoder branch on the RING tensor.
Within each network family the arms differ only in the information content of the input. The quantile (pinball)
objective is probabilistic, so the loss-smoothing explanation of the literature applies less than to MSE training;
ramp-aware objectives and a real-future-frame oracle are left to the full study.

## Contrasts and decision rule
Treatment minus control at each lead: I128 - I64, I256 - I64, SUN high - control, RING high - control (eight
contrasts). **GO** for an arm if at either lead the ramp-region twCRPS difference has a 95% CI below 0, the relative
improvement is at least 3% of the control, and the overall CRPS is not worse by more than 1%. The dissociation is
assessed by the paired bootstrap of the difference in relative gains between the two leads (SUN and RING separately).
Eight contrasts are tested without multiplicity correction because the pilot only selects what enters the
confirmatory study (R-H1, frozen holdout of November-December 2019). On NO-GO for every arm the pilot is reported as a
negative finding and no further resolution training is done.

## Additional secondary metrics
Skill against smart persistence, ramp recall of the median by ramp-onset lead bin, and results for patchy-cloud
samples (high within-hour variability of the clear-sky index), as expected by recent benchmarks.
