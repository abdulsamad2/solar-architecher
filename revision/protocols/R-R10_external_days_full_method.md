# Protocol R-R10: the full method on SkyGPT's external test days (60-min histories rebuilt from the file itself)

Committed before the analysis is run. Label: **post hoc external** (contextual, unpaired with published values).
The five external days (2,582 forecasts, November-December 2019, later than all development data) were scored once in
Phase A2 with the frozen models; there every forecast was routed to the 15-min fallback because the file gives only a
15-min history per sample. Those results are known (frozen forecaster 2.455 kW exact CRPS; calibrated 90% coverage
0.77 with the daily-reset initialisation). No model is trained here.

## What is new
The samples of the external file are consecutive in time, so the 16 frames and PV values of each sample (t-15 ... t)
form a per-minute record table of each day. From this table the 60-min window of the 60-min ramp expert (16 frames
at a 4-min stride, t-60 ... t, with the PV history at the same times) is built for every forecast whose window lies
inside the table; the others keep the 15-min fallback, exactly as the routing rule of the method prescribes. Only
frames and PV values contained in the external file are used (no raw archive).

## Checks before scoring (STOP conditions)
- Frames and PV values of a minute that appears in several samples are identical across samples.
- With every forecast forced to the fallback, the frozen forecaster reproduces the stored Phase A2 forecasts.

## Forecasters
Frozen experts (15 members). (a) frozen rule G3''-F with its routing; (b) method v2 (R-M1: gated probability mixtures,
gates fitted on each member's calibration days as in R-R9); (c) G4, calm expert, direct expert (as Phase A2).

## Scores
Exact quantile CRPS; standard empirical CRPS at m = 10, 30, 100 samples and the fair estimator (R-C21 procedure), next
to the published values (SkyGPT 2.81 at m = 100; PV-MM-Diffusion 2.63 at m = 30; SUNSET 3.31 at m = 10); paired
day-block bootstrap between our forecasters (five days: descriptive only). Coverage and width of the 90% interval:
raw, continuous rolling conformal prediction and adaptive conformal inference, both started from development scores
of the frozen forecaster (all earlier than the external days).

## Reading rule (fixed now)
The comparison with published values stays contextual and unpaired, at matched sample counts; no state-of-the-art
claim is made from five days. Whether the 60-min branch helps on these days is reported as the paired difference
between the routed forecaster and its fallback-only version, whichever sign it has.
