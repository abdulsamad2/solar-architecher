# R-H0 report: resolution pilot (sun-region and horizon-ring detail), development out-of-fold

Protocol: `revision/protocols/R-H0_resolution_pilot.md` (amendments 1 and 2, note of 2026-11-16). Code:
`src/revision_h0_prepare.py`, `src/revision_h0_train.py`, `src/revision_h0_eval.py`. Raw output:
`revision/results/h0_resolution_pilot.json`. Exploratory (category iii); seed 42; 12 pilot months, 147 days,
40,922 forecasts; folds 0, 1 and 3 (they contain every pilot day once as a validation day).

## Result: NO-GO for both arms that were run
Each arm is the frozen base network (E4-B-fit at 15 min; the same network trained on all development samples at
5 min) plus one extra image branch, started as exactly the base forecaster. Treatment and control differ only in the
information content of the extra input (raw 2048-px detail against the same region area-averaged to the 64-px grid).

| Contrast (high detail - control) | Ramp-region twCRPS (kW) | Relative gain | CRPS (kW) | Rule (>= 3%, CI < 0) |
|---|---|---|---|---|
| Sun window, 15 min | -0.0002 [-0.0005, +0.0000] | 0.1% | -0.0040 [-0.0055, -0.0025] | no-go |
| Sun window, 5 min | -0.0002 [-0.0004, -0.0001] | 0.1% | -0.0008 [-0.0016, -0.0001] | no-go |
| Horizon ring, 15 min | -0.0000 [-0.0011, +0.0009] | 0.0% | -0.0022 [-0.0073, +0.0028] | no-go |
| Horizon ring, 5 min | +0.0001 [-0.0002, +0.0004] | 0.0% | +0.0000 [-0.0019, +0.0018] | no-go |

Dissociation (relative ramp gain at 5 min minus at 15 min): sun +0.0000 [-0.0008, +0.0006], ring -0.0002
[-0.0031, +0.0021]. The predicted pattern (sun detail helps more at 5 min, horizon detail more at 15 min) is not
observed. The median lag is 16 min (15-min lead) and 6 min (5-min lead) in every arm.

## Limits (they make this a weak null)
- Every arm stopped at its first or second epoch and is 1.1-2.0% worse in CRPS than the base network it started from:
  fine-tuning on the 12 mostly-winter pilot months degrades the base faster than the new branch learns. Control and
  treatment differ by 0.0002 kW, so the branch contributed almost nothing in either. The pilot does not separate "no
  usable information in the added detail" from "the added branch was not trained enough".
- The whole-sky arms (three information levels) were not run: one run filled the 32 GB GPU and they were stopped to
  free the GPU for the review-driven analyses. They remain untested.
- Cold-start arms (amendment 1) lost to persistence and are described in the protocol; their one computed contrast was
  also null.
- No untouched raw day with PV exists (protocol note), so a GO could not have been confirmed on new data.

## What may be claimed
Within this design, adding raw-resolution detail of the sun region or of the horizon ring to the 64-px direct expert
did not improve the ramp-region score at a 5- or 15-min lead. It is not shown that image resolution is irrelevant.
