# Protocol R-C06: issuance-time input availability (C06)

Committed before any run. Label: **audit** + **post hoc** sensitivity (test), development audit.

## Question
Is every model input available at the nominal issue time t? If not, how much does the result depend on it?

## Facts to document (from code and data)
- Nominal origin t, the matched record times of all 16 frames and of the target (±30 s nearest-record matching, `src/skippd.py`).
- PV values are forward-averaged over the following minute (SKIPP'D documentation): a PV value stamped r is available at about r + 60 s.

## Procedure
1. **Timestamp audit (dev and test):** distribution of (latest history record time − t), (target record time − (t + 15 min)); share of samples with the latest record after t; effective availability time = latest image record time and latest PV record time + 60 s; effective lead time = target record time − availability time.
2. **Past-only sensitivity (test, post hoc; dev OOF for E4-B-fit, E6-S, A2):** rebuild every history window with each frame matched to the record nearest (nominal frame time − 90 s) within ±30 s, i.e. strictly ≤ t − 60 s, so images and forward-averaged PV are available by t. Target unchanged (t + 15 min). Run the frozen 15 members of E4-B-fit, E6-S, A2 with frozen gates → G3″-F, G4, E4-B-fit, E6-S. Score raw CRPS by stratum; paired day bootstrap (10,000, seed 20261022) of shifted − original for each model, and H1 (G3″-F − E4-B-fit) under the shifted inputs.
3. Samples whose shifted window is incomplete keep their original routing rule applied to the shifted availability (G3 fallback if the 60-min window is incomplete); counts reported.

## Decision rule
If H1's CI stays below 0 under the shifted inputs, the primary conclusion is robust to input availability; the manuscript documents the timing and reports the shifted result. Otherwise the manuscript reports the shifted result as primary for deployment-relevant statements.

## Outputs
`revision/results/c06_timing_audit.json`, `revision/results/c06_shifted_inputs.json`; report `revision/reports/C06_timing.md`.
