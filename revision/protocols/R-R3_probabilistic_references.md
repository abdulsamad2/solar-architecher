# Protocol R-R3: probabilistic reference forecasts for the CRPS skill on the official test set

Committed before the analysis is run. Label: **post hoc test** (category ii). No network is trained; the frozen
evaluation is unchanged. The CRPS of the frozen models is known; the CRPS of the references below has not been computed.

## Origin
A simulated review of manuscript v8 (2026-11-16) objected that the headline CRPS skill (41.1%) is measured against
deterministic smart persistence, whose CRPS equals its MAE, and that the result table has no probabilistic reference.
The verification literature treats the reference of a CRPS skill score as a probabilistic forecast and recommends at
least two benchmarks, a reliable naive one and a sharper one (Doubleday, Van Scyoc Hernandez and Hodge 2020; Yang
2019; Lauret, David and Pinson 2019).

## Reference forecasts (clear-sky index K from the frozen test clear-sky anchor; target = K x P_clear(t + 15 min);
quantiles at the nine levels of the models, empirical with linear interpolation; values below 0 set to 0)
- **PeEn-120 (primary sharp reference):** persistence ensemble; members = K of all records of the same day in the
  120 min up to the issue time (Doubleday et al. use a 2-h window). Sensitivity: 30 and 60 min. With fewer than five
  members the forecast is the point value K(t) x P_clear(t + 15).
- **SP-dressed, static:** smart persistence K(t) plus the empirical distribution of the 15-min change of K over all
  development samples (development data only; the same data that trained the models).
- **SP-dressed, adaptive:** as above, but with the 15-min changes of K that have matured within the last 120 min of
  the same day (at least ten, else the static distribution), following the 2-h error window of Doubleday et al.
- **CH-PeEn (naive yardstick):** complete-history persistence ensemble; members = K of all development records with the
  same hour of day (Yang 2019).
- Point references (CRPS = MAE): the official smart persistence and persistence of K with this paper's clear-sky model.
- Learned reference, reported separately: the PV-only quantile model P15 (15 members, R-E1).

## Scores
CRPS with the estimator of the manuscript on the 13,266 official test forecasts, overall and for sunny and cloudy
days; skill of the proposed forecaster against each reference; paired day-block bootstrap of the difference; exact
day-level sign-flip p-value (R-R2). STOP check: the CRPS of the proposed forecaster and of official smart persistence
reproduce Table 2 (1.006 and 1.707 kW).

## Reading rule (fixed now)
The manuscript reports the CRPS skill against **the naive probabilistic reference with the lowest overall CRPS** among
PeEn-120, SP-dressed (static), SP-dressed (adaptive) and CH-PeEn as its headline probabilistic skill, gives the other
references in the table, and keeps the 41.1% against deterministic smart persistence only as a labelled secondary
figure. The 30- and 60-min windows are sensitivity results and cannot become the headline reference. The result is
reported whichever way it falls.
