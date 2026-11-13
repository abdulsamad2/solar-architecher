# v5 review revision plan

Source: `SHADOW_MoE_v5_Review_and_Revision_Plan.docx` (12 Nov 2026). Recommendation: major revision.
Branch: `revision/v5-review`. Baseline tag: `pre-v5-revision` (commit 6a1ced3).

## Commit and revert procedure

Every item is committed separately, in this order:

1. `revision(protocol): <ID> ...`: the protocol (question, design, prediction, decision rule), committed **before** any run. This is the temporal proof that the analysis was planned before its results existed.
2. `revision(code): <ID> ...`: analysis or training code (+ tests where behaviour is new).
3. `revision(results): <ID> ...`: raw outputs, committed unchanged before interpretation.
4. `revision(report): <ID> ...`: the interpretation.
5. `revision(manuscript): <IDs> ...`: manuscript edits that use those results.

Reverting:
- one item: `git revert <sha>` for its commits (newest first), or `git revert <first-sha>^..<last-sha>`;
- the whole revision: `git switch paper/manuscript-audit` (untouched), or `git reset --hard pre-v5-revision` on this branch;
- the commit list per item is kept in `revision/commit_log.md`.

Large binaries (docx, pdf, png, pptx) are stored with Git LFS.

## Evidence labels

- **Frozen**: the original pre-registered Gate 5 SKIPP'D evaluation. Never rescored or replaced.
- **Development (CV)**: 5 blocked folds × 3 seeds on the 497 development days.
- **Post hoc (test)**: new analyses on the 20 official test days, run after the frozen evaluation. Exploratory; reported separately; never relabelled pre-registered (C09, C10).
- **Forward (E3)**: a new chronological SKIPP'D split; protocol committed before training.

## Items

| ID | Review item | Type | Plan | Evidence |
|---|---|---|---|---|
| C07, C19 | Conformal initialisation strictly past-only; init-score mismatch | Essential check | Audit (done: ~47% future-dated); rerun calibration from frozen predictions with past-only init; survival of init scores per day; matched-ensemble sensitivity | post hoc (test), dev |
| C06 | Issuance-time input availability | Essential check | Timestamp audit of matched records vs nominal time; past-only (≤ t) matching rerun of inference with frozen checkpoints | post hoc (test), dev |
| E2 | Gate feature and constant weight ablation | Essential | Frozen expert predictions; 50/50, CRPS-optimal constant, event-only, disagreement-only, proposed gate, G4 | dev, post hoc (test) |
| E1 | PV-only probabilistic baseline | Essential | Train PV/CSI-history model, matched folds/seeds/budget/ensemble | dev, post hoc (test) |
| E4 | History × encoder | Strongly recommended | Full 2×2: 2D/3D × H15/H60, 16 frames, 3 seeds | dev, post hoc (test) |
| E3 | Forward evaluation and calibration | Essential for deployment claims | Chronological SKIPP'D split; retrain experts and gates on earlier data; compare 4 calibration methods | forward |
| C15, C18 | Event definitions; event probabilities and ramp metrics | Essential / strongly recommended | Brier, reliability, PR vs climatology; confusion counts and denominators; change slope | dev, post hoc (test) |
| C16, C17, C20 | Where the gain occurs; conditional coverage; strong comparators | Essential / strongly recommended | Contribution decomposition; raw + calibrated coverage by condition and ramp direction; day bootstrap vs E6-S and G4 | frozen predictions, post hoc |
| C22 | Latency | Strongly recommended | Batch-1 end-to-end latency, both branches; parameter counts | measurement |
| C21 | External comparison | Strongly recommended | Fair sample-based CRPS, matched sample counts, repeats | contextual |
| C02 | Specific contribution, related work | Essential | Literature search (deep-research) | literature |
| C01, C03–C05, C08–C14, C23 | Text corrections | Essential editorial | Manuscript v6 edits | – |
| C12 | Representation vs loss weighting | Conditional | Narrow the claim (no 2×2 rerun) unless time allows | – |
| C13 | Physics loss naming | Essential editorial | Rename; limit conclusion | – |
| C24 | Declarations and release | Essential before submission | Pin environment, manifests, hashes; author fields stay as marked placeholders | – |
| Tables | Rearrangement | – | Name-mapping table; new result tables; reorder | – |

Optional extensions (second site, optical flow, other horizons) are out of scope, as the review allows.
