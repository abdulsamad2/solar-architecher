# R-E3b: forward models on the official test days inside the forward window (completes the R-E3 subset)

Status: R-E3 listed "the 3 official test days in the evaluation window (2019-08-10, 2019-10-14, 2019-10-19) ... as a
separate subset", but the subset was not scored (deviation recorded in `revision/reports/E3_forward.md`). This
protocol scores it, post hoc; written before any of these forecasts was computed.

## Method
- Models, gates and conformal initial scores exactly as in R-E3 (`src/revision_e3.py`): the five-seed forward
  experts trained on the fit period, the clear-sky anchor fitted on the fit period, gates fitted on the selection
  period, timestamp routing (60-min blend when a full 60-min history exists, else the 15-min blend). Nothing is refitted.
- Samples: all official-test samples (1-min cadence) whose origin day is 2019-08-10, 2019-10-14 or 2019-10-19.
- Scores: CRPS (raw quantiles) of G3''-F, E4-B-fit, E6-S, G4 and P15, and smart persistence (MAE), overall, per day and
  on endpoint ramps (forward threshold τ of R-E3); RMSE of the median; 90% interval coverage with no calibration,
  daily-reset and continuous rolling conformal prediction and ACI, initialised from the latest selection-period scores.
- With three days a day bootstrap is not informative: report point estimates and per-day differences only.
- For context, the frozen 15-member models' CRPS on the same samples (`experiments/final/test_predictions_v1.npz`).

## Output
`revision/results/e3_testdays.json`; code `src/revision_e3_testdays.py`. Labelled POST HOC; the frozen evaluation and
the R-E3 results are unchanged.
