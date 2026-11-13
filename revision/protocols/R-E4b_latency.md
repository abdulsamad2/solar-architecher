# R-E4b: inference cost of the four history × encoder arms (completes review item E4)

Status: post hoc addition to R-E4 (the review asks E4 to report "inference cost"); written before any timing was run.

## Question
How much inference time does each E4 arm need per forecast: 2D CNN-GRU at 15 min (D0) and 60 min (D2), 3D CNN-GRU at
15 min (A2-H15) and 60 min (A2)?

## Method (identical to the deployed-system latency of R-C22)
- Hardware and software as in R-C22 (NVIDIA RTX 5000 Ada; PyTorch from `requirements-lock.txt`).
- All 15 frozen members (5 folds × 3 seeds) of each arm resident on the GPU, loaded from their run folders
  (`src/revision_posthoc_dump.EXPERTS`), with the arm's own input window (15 min or 60 min, 16 frames) and extra-input
  shape; batch size 1; members run sequentially, no batching or compilation.
- Timed per forecast and arm: image gather and normalisation plus the 15 member forward passes (the "experts" step of
  R-C22); preprocessing of the clear-sky index is shared by all arms and timed once.
- Samples: the same random draw as R-C22 (`numpy.random.default_rng(0)`), restricted to official-test samples with a
  valid 60-min history so that all four arms run on identical samples; 50 warm-up forecasts, 500 timed.
- Report median and 95th percentile (ms) per arm, and parameters per member.

## Output
`revision/results/e4_latency.json`; code `src/revision_e4_latency.py`. No model is trained and no test outcome is used.
