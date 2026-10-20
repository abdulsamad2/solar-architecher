# Commit hash mapping after author rewrite (2026-10-21)

All commits were re-authored to the repository owner's identity; content, messages and order are unchanged. Hashes cited in reports before this date refer to the **old** column.

| Old | New | Message |
|---|---|---|
| f18cf20 | 930d6fd | Phase 0-3: scaffold, SKIPP'D manifests, E0 persistence, E1 SUNSET port |
| 9c97ed7 | be82eb4 | Clear-sky PV anchor, blocked dev folds, E2/E3 CNN-GRU trainer |
| 0781e4c | da7a35c | Record git commit in run config |
| c4bfc23 | 5427c41 | Real-data preprocessing audit (Phase 2.6), opt-in via pytest -m realdata |
| 9a0681d | 44efcba | Trainer v2: cosine lr, grad clip, fused EMA with warm-up, CNN-only bf16, spatial pooling option |
| 9b8dd69 | 2f6738d | E3b: K-history input option and pre-training audit |
| 35a2dfb | 31158c7 | Paired comparison tool (day-block bootstrap); E3b protocol notes |
| 4249d7a | f29ea3c | E2/E3/E3b paired comparison and Gate 1 recommendation |
| 8263d33 | 136ebe8 | Gate 1: fix replication protocol and closing rule before seed results |
| 75ecd81 | 1646b5e | E4 prep: quantile head option, interpolated-quantile CRPS, probabilistic scores, CRPS comparison, Gate 1 seed summary |
| 7c58010 | aac07b5 | E4 pre-registration |
| bc87a1d | 008ff57 | Gate 1 replication (seeds 42/123/456) results and closing-rule verdict |
| 51c46ee | ca5d058 | Gate 1 closed; fix quantile-head init (start at training climatology) before E4 |
| 4db1938 | db03bb4 | E4-A/E4-B results, calibration diagnostics, Gate 2 assessment (not closed) |
| 7169375 | 1640db9 | Gate 2 pre-registration: E4 replication + calibration protocol; calib-holdout mode with stop-condition asserts |
| 308ecee | 41772e2 | Calibration module (CQR, monotone recalibration) + tests; prereg amendment 1 (z-space tail extrapolation) before results |
| f1fc37a | dcc5993 | Gate 2 report generator, seed-42 context backfill, inference benchmark |
| af450b2 | 71dde65 | Gate 2 report: E4 replication, calibration (CQR / recalibration), cost; closing rule not met |
| df2b0a8 | 2391c17 | E4-CAL2 pre-registration (adaptive conformal), before any validation outcome |
| 614edd5 | ba8555c | E4-CAL2 core (causal replay, ACI, rolling conformal, selection) + tests; prereg amendment 1 (ACI anti-windup) before results |
| 6506a07 | 2e92bc6 | E4-CAL2 runner |
| 10b70b7 | 1eb9fb1 | E4-CAL2 results and report: adaptive conformal retained (rolling recommended); Gate 2 verdict preserved |
| ee0d2d6 | 4fcaf5c | Phase 6 prep: event labels, EventMixtureHead (+init), capacity-matched/residual quantile heads, trainer integration, design DRAFT |
| 4e2a2d0 | 1944fc2 | Phase 6 pre-registration FROZEN (thresholds, margin 0.02 kW, endpoints, Gate 3 rule); frozen-threshold STOP check |
| 982bfaf | 987cb4d | Sep-Oct 2017 audit: one-sided under-prediction from inter-annual PV/clear-sky decline; smoke not supported; 2019 camera colour shift evidenced |
| 978e87c | 098b593 | Gate 3 analysis script |
| 229a8be | 42811d7 | Gate 3 report: event decomposition NOT retained (0/3 seeds); residual-target gain with ramp trade-off; severity diagnostic pass |
| 62c8ff5 | bbc0ef1 | E7-BLEND pre-registration (post-hoc gated mixture of experts), before evaluation |
| 1fd44a7 | 1d65087 | E7-BLEND implementation + tests |
| 3f722fe | 9daa1b4 | E7-BLEND report: B1 not retained (ramp non-inferiority 0/3); gating beats ensembling 3/3; B1 best overall (0.592) |
| 518d417 | 53cac39 | E7b pre-registration (enhanced gate features), gate module and synthetic scoring tests, before any E7b result |
| 9adb8ee | 54bef45 | E7b runner |
