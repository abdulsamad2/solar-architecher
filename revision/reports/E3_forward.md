# E3 report: forward-in-time evaluation and calibration (SKIPP'D)

Protocol: `revision/protocols/R-E3_forward.md` (committed before training). Split: `revision/manifests/folds_forward_v1.json` (fit 2017-03-09…2019-06-30, 383 days; selection 2019-07-02…07-31, 29 days; evaluation 2019-08-02…10-26, 83 days). Code: `src/revision_e3.py`. Raw output: `revision/results/e3_forward.json`.

Every model (E4-B-fit, E6-S, A2, P15; 5 seeds) was retrained on fit days with early stopping on the selection period; gates (G3, G3″, G4) and conformal initial scores come from the selection period; the evaluation period was scored once.

**Disclosure:** architectures and hyperparameters were chosen during development with blocked CV whose fold 4 contained the evaluation dates. E3 refits everything on earlier data but is not a never-seen period in the design sense. The evaluation period (late summer–autumn 2019, after the 2019-05-22 camera change) is mostly calm: 48 of 83 days sunny; 2,046 of 27,937 samples are ramps (7.3%).

## Raw CRPS (kW; 5-member ensembles)
| Model | Overall | Sunny | Cloudy | Ramp | RMSE (median) |
|---|---|---|---|---|---|
| **G3″-F** | 0.374 | 0.131 | **0.713** | **1.813** | **1.344** |
| E4-B-fit | 0.487 | 0.258 | 0.808 | 1.823 | 1.400 |
| E6-S | 0.377 | **0.108** | 0.753 | 2.176 | 1.473 |
| G4 | **0.368** | 0.118 | 0.716 | 1.884 | 1.354 |
| P15 (PV only) | 0.454 | 0.183 | 0.831 | 2.398 | 1.510 |
| Smart persistence | 0.665 | 0.285 | 1.195 | 3.779 | 1.683 |

Paired day bootstrap, G3″-F − comparator [95% CI]:

| Comparator | Overall | Sunny | Cloudy | Ramp |
|---|---|---|---|---|
| E4-B-fit | **−0.113 [−0.127, −0.100]** | −0.126 | −0.095 | −0.010 [−0.068, +0.041] |
| E6-S | −0.003 [−0.016, +0.010] | +0.023 | −0.040 | −0.363 [−0.457, −0.272] |
| G4 | **+0.007 [+0.000, +0.014]** | +0.013 | −0.003 | −0.071 [−0.118, −0.028] |
| P15 | −0.080 [−0.098, −0.063] | −0.052 | −0.118 | −0.585 |
| Persistence | −0.291 [−0.334, −0.251] | −0.153 | −0.482 | −1.966 |

**Pre-specified questions:** F1 (G3″-F better than E4-B-fit overall) **confirmed**; F2 (better than persistence) **confirmed**.
**Additional:** in this forward period G3″-F is **not** better than the calm expert overall (difference ≈ 0) and is slightly **worse than G4 overall** (+0.007, CI lower bound 0.000), while being better than both on ramps. The ranking against G4 seen on the official test (−0.018, CI including 0) does not hold forward.

## Calibration (G3″-F; initial scores = latest selection-period scores)
| Nominal | Method | Coverage overall | Ramp | Width (kW) | Winkler |
|---|---|---|---|---|---|
| 90% | none | 0.930 | 0.596 | 2.42 | 3.17 |
| 90% | daily-reset rolling | 0.882 | 0.517 | 2.26 | 3.16 |
| 90% | **continuous rolling** | **0.901** | 0.530 | 2.32 | **3.15** |
| 90% | ACI (γ = 0.005) | 0.901 | 0.557 | 2.45 | 3.26 |
| 80% | none / daily / continuous / ACI | 0.812 / 0.744 / 0.790 / 0.799 | 0.37 / 0.33 / 0.34 / 0.36 | | |
| 50% | none / daily / continuous / ACI | 0.442 / 0.390 / 0.476 / 0.498 | 0.10 / 0.09 / 0.11 / 0.11 | | |

F3 (90% within 0.90 ± 0.03): daily-reset, continuous and ACI met; raw (0.930) at the boundary, not met. Continuous rolling and ACI are closest to nominal at all three levels; daily reset under-covers. **Ramp-conditioned coverage remains far below nominal under every method (0.52–0.60 at 90%).**

## Deviation
The protocol listed the 3 official test days inside the evaluation window as a separate subset. They were not scored in E3 (they are part of the official test set already reported); this subset is omitted.

## Manuscript implications (C08)
- Add E3 as a forward-in-time evaluation, separate from the frozen benchmark, with the disclosure above.
- Deployment statements: supported relative to the direct expert and persistence; not supported as superiority over the calm expert or G4 in calm periods.
- Recommend continuous rolling conformal or ACI over daily reset for deployment; ramp-conditioned under-coverage is a stated limitation.
