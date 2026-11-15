# Protocol R-R5: upward reserve compared at equal reliability

Committed before the analysis is run. Label: **post hoc test** (category ii), stylised. No model is trained. R-Y3 is
known (mean reserve 2.88 kW at reliability 0.959 for the frozen forecaster; a constant persistence-based reserve
reached 0.889), but methods were compared there at different reliabilities.

## Origin
A simulated review (2026-11-16) objected that the reserve comparison is not made at equal reliability, gives no
reliability on falling ramps, and is in kW of one array.

## Set-up
PV is scheduled at a point forecast f; the upward reserve r >= 0 covers shortfalls; the schedule is reliable when
y >= f - r. For each method the reserve is r = f - q05 + delta, where q05 is the method's 5% quantile and the
constant delta is set so that the reliability on the 13,266 official test forecasts is exactly 0.95 (equalisation on
the evaluated data, a comparison device, not a deployable calibration; the past-only calibrated values are in R-Y3).
- Probabilistic methods (f = median): frozen forecaster G3''-F, direct expert, calm expert, G4, PV-only model P15,
  smart persistence dressed with the last 120 min of its errors (R-R3).
- Constant reserves (r = delta): around official smart persistence and around the median of the frozen forecaster.

## Reported
Mean reserve in kW and in % of the 30.1 kW capacity, overall and on cloudy days; reliability, mean and 95th-percentile
shortfall on falling ramps, rising ramps and non-ramp forecasts; paired day-block bootstrap of the reserve difference,
frozen forecaster minus each alternative (delta fixed), with the day-level sign-flip p-value.

## Reading rule (fixed now)
"Less reserve at equal reliability than X" is stated only if the difference has a bootstrap interval below zero and
sign-flip p < 0.05. The reliability on falling ramps is reported next to every reserve figure. Reported whichever way
it falls.
