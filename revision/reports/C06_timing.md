# C06 report: issuance-time input availability

Protocol: `revision/protocols/R-C06_input_timing.md`. Code: `src/revision_c06.py`. Raw outputs: `revision/results/c06_timing_audit.json`, `revision/results/c06_shifted_inputs.json`. Label: audit + post hoc (test).

## Timestamp audit (manifest `forecast_tol30_v2`)
- **The forecast origin t is the latest matched record itself** (`history[:, 0] == origin` for all 164,362 development and 13,266 test samples). No history image is later than t (0 samples).
- Earlier frames are matched to the record nearest t − j min within ±30 s; offsets are −30 … +20 s (median 0), so every earlier frame precedes t.
- Target record: within −30 … +20 s of t + 15 min (median 0).
- **PV:** SKIPP'D PV values are averages over the following minute, so PV(t) is available only at ≈ t + 60 s. The persistence baseline, the K(t) anchor of the residual expert and the gate features use PV(t).
- **Effective lead time** from PV(t) availability to the target record: median 14.0 min (13.5–14.3).

## Past-only sensitivity (post hoc, 15 frozen members, frozen gates)
Every PV/K input (history, K(t), gate features, persistence) taken from the record ≈ 60 s earlier, so all inputs are available at t; images unchanged. 53 of 13,266 samples have no earlier record and are excluded from both arms.

| Model | Shifted − original CRPS (kW) [95% CI], overall | ramp | non-ramp |
|---|---|---|---|
| G3″-F | +0.009 [+0.005, +0.013] | −0.081 [−0.108, −0.058] | +0.024 |
| E4-B-fit | +0.008 [+0.005, +0.011] | −0.102 | +0.026 |
| E6-S | +0.016 [+0.008, +0.023] | −0.214 | +0.054 |
| G4 | +0.011 [+0.007, +0.016] | −0.130 | +0.035 |
| Smart persistence | +0.062 [+0.047, +0.078] | −0.491 | +0.154 |

**H1 under past-only inputs:** G3″-F − E4-B-fit = −0.110 [−0.148, −0.074] (original −0.111 [−0.150, −0.076]). G3″-F − persistence = −0.754 [−0.886, −0.626].

**Decision rule met:** H1's CI stays below 0, so the primary conclusion is robust to input availability.

Note (descriptive): ramp-stratum CRPS improves under the shift for every model including persistence. The ramp stratum is defined from K at the origin record; a forward-averaged PV(t) already contains part of a ramp beginning in the first minute, so the original inputs make some ramps look smaller than they are for persistence-like forecasts. This is consistent with PV(t) carrying information from the minute after t.

## Manuscript implications
- §4.1: document nominal origin = latest record, matching tolerance and offsets, PV averaging interval, availability time (t + 60 s) and the 14-min effective lead.
- Report the shifted-input sensitivity (one table row or sentence; supplementary table).
