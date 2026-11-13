# Protocol R-X4: forward evaluation with separate early-stopping and gate-fitting days

Committed before any training. Label: **forward** (same evaluation period as R-E3; new refit).
Origin: R-E3 disclosure. In E3 the 29 selection days (2019-07-02..07-31) served both for early stopping of the forward experts and for fitting the gates and conformal initial scores; the ranking against G4 (+0.007 [+0.000, +0.014] kW overall) was reported as not robust.

## Question
Does the forward ranking of the proposed forecaster against the calm-condition expert (E6-S) and G4 change when the gates are fitted on days that were not used for early stopping?

## Split (`revision/manifests/folds_forward_v2.json`, built by `src/forward_split.py --v2`)
Fit days unchanged (<= 2019-06-30). Selection days split chronologically: the first half (days 1-15 of the 29 selection days) for early stopping (`val_days`), the second half (days 16-29) for gate fitting, lambda selection and conformal initial scores (`gate_days`). Evaluation days unchanged (83 days, 2019-08-02..10-26).

## Retraining
E4-B-fit, E6-S, A2 and P15, seeds 42/123/456/789/1011, identical commands to `scripts/revision_train_queue.ps1` stream B except `--folds folds_forward_v2.json` and output `experiments/R_X4/<model>`. Gate-day predictions are produced with the trained members (`revision_e3.predict`, fit-period anchor).

## Analysis
As `src/revision_e3.py` (fit-period anchor, gates G3, G3'', G4 by `select_and_fit` on gate days only, 5-seed ensembles, raw CRPS overall / sunny / cloudy / ramp / non-ramp, paired day bootstrap of G3''-F - each comparator, 90% intervals with no calibration, daily-reset rolling, continuous rolling and ACI initialised with the latest gate-day scores, all matured before the first evaluation origin). Event threshold tau from the fit days (E6-S training rule, unchanged).

## Cost rule
If the retraining is estimated above ~6 GPU hours (sequential), it runs in a second stream parallel to R-X3; if the GPU cannot hold both, R-X3 I15 has priority and R-X4 is reported as not run.

## Interpretation rule
Report G3''-F - E6-S and G3''-F - G4 (overall, ramp) and state whether each CI sign changes relative to E3. The E3 result stays in the paper; R-X4 is reported as a sensitivity check.
