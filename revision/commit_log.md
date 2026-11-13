# Commit log per revision item

Revert one item: `git revert <sha>` for its commits, newest first. Revert everything: `git reset --hard pre-v5-revision` (or switch back to `paper/manuscript-audit`).

| Commit | Kind | Message |
|---|---|---|
| 6e57387 | plan | revision(plan): v5 review revision plan, commit/revert procedure, research state |
| 5384f13 | chore | chore: scope Git LFS to paper/, manuscript/ and revision/ binaries (leave earlier tracked files as normal blobs) |
| d537511 | protocol | revision(protocol): R-C07, R-C06, R-E1, R-E2, R-E3, R-E4, R-C18, R-C16/C17/C20, R-C22/C21 protocols (before any run) |
| 123da81 | code | revision(code): E1 PV-only encoder (pv_gru), E3 chronological split manifest, revision GPU training queue |
| a1b4f7b | code | revision(code): C07 past-only conformal initialisation analysis + shared CPU test context |
| 96f3fab | results | revision(results): C07 raw output (frozen H5 reproduced to 1e-9; past-only init rerun; window composition) |
| be0d7be | code | revision(code): E2 gate/constant-weight ablation (development) |
| 787fe2b | results | revision(results): E2 development raw output (G3'' reproduced to 1e-9) |
| fa4783f | literature | revision(literature): SKIPP'D probabilistic baselines and conformal references (deep-research notes) |
| ffe7cd5 | code | revision(code): C16/C20 contribution decomposition and day bootstrap vs E6-S/G4 |
| c2328aa | results | revision(results): C16/C20 raw output; forecast-combination literature notes |
| 7a3ccc6 | code | revision(code): post hoc per-member test dump (frozen ensembles reproduced exactly) and C15/C18 event scoring |
| e3bbcc5 | results | revision(results): member-dump manifest (hashes; max /diff/ 0.0 vs frozen), C18 development output, prior-art notes |
| 53e3296 | code | revision(code): post hoc test analyses (C19 ensemble size, E2 test, C18 test, E1/E4 test) |
| 558b96e | results | revision(results): post hoc test raw outputs for C19, E2, C18 |
| e252459 | report | revision(report): C02/C04/C21 literature report (deep-research skill): prior art, terminology, SKIPP'D comparison framing |
| 9f0381d | code | revision(code): C06 timestamp audit and PV-availability (t-60 s) sensitivity rerun |
| b686bd2 | results | revision(results): C06 raw outputs (H1 shifted -0.110 [-0.148,-0.074]); E1/E4 test member hashes |
| b740c52 | results | revision(results): training records for E1 (P15, P60), E4 (3D@H15) and E3 forward refits (configs, metrics, histories, logs) |
| ab8211c | code | revision(code): E1/E4 development comparisons (paired day bootstrap, ramp metrics, parameter counts) |
| 0e29e33 | results | revision(results): E1/E4 development and post hoc test raw outputs |
| 675fd49 | code | revision(code): E3 forward evaluation (selection-period gates and conformal init; 4 calibration methods) |
| a8fffe6 | results | revision(results): E3 forward raw output (83 days; F1, F2 confirmed; G4 slightly better than G3''-F) |
| e035fa1 | code | revision(code): C22 batch-1 end-to-end latency benchmark |
| 8798d68 | results | revision(results): C22 latency raw output (~330 ms median batch-1; 5.94 M resident parameters) |
| 90df191 | code | revision(code): C21 matched-m sample CRPS (standard and fair estimators) |
| d814333 | results | revision(results): C21 raw output |
| b3fee58 | report | revision(report): interpretation reports for C06, C07/C19, E1/E4, E2, E3, C15/C18, C16/C17/C20/C22, C21 |
| b9e837c | review | revision(review): add the v5 review document (.docx via LFS) and its extracted text |
| d4c9d07 | code | revision(code): C24/D7 pinned environment (requirements-lock.txt, matches the test execution manifest) and revision hash inventory script |
| 70c6ec5 | results | revision(results): SHA-256 inventory of revision checkpoints, member dumps, manifests and results (88 files) |
