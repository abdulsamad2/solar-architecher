"""Development rehearsal of the complete one-shot evaluation body (src.final_eval.evaluate), no test data.

Forecasts: 3-member out-of-fold ensembles (genuinely held out), all development validation samples.
Day labels: development clear-day labels. SUNSET: not available on development (None).
Calibration initialisation: frozen development out-of-fold scores from OTHER days only.
Output: reports/final_rehearsal_dev.json. Bootstrap CIs use development days (not the 20 test days).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.e8_features import FEAT, verify_frozen
from src.final_eval import FOLDS, SEEDS, Split, assemble, evaluate


def main():
    verify_frozen()
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    tau = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    rec = np.load(FEAT / "e8_record_features_v1.npz")
    split = Split("dev", torch.device("cuda"), np.concatenate([rec["P"], rec["M"]], 1).astype(np.float32))
    va = {k: np.flatnonzero(np.isin(split.ds.day.astype(str), folds_doc["folds"][k]["val_days"])) for k in FOLDS}
    fc = assemble(split, {(s, k): va[k] for s in SEEDS for k in FOLDS}, folds_doc, tau)
    idx = np.concatenate([va[k] for k in FOLDS])
    sunny = np.array([folds_doc["day_labels"][d] for d in split.ds.day.astype(str)[idx]])
    res = evaluate(split, idx, fc, sunny, None, exclude_same_day=True)
    res["note"] = "DEVELOPMENT REHEARSAL (out-of-fold 3-member ensembles; dev clear-day labels; no SUNSET). Not a test result."
    Path("reports/final_rehearsal_dev.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"n": res["n"], "n_days": res["n_days"], "H1": res["hypotheses"]["H1 overall CRPS G3''-F - E4-B-fit"],
                      "H3": res["hypotheses"]["H3 ramp CRPS G3''-F - E4-B-fit (non-inferiority margin 0.02)"],
                      "H5": res["hypotheses"]["H5 G3''-F calibrated 90% coverage"]}, indent=1))


if __name__ == "__main__":
    main()
