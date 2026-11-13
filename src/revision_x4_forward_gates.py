"""Revision R-X4: forward evaluation with separate early-stopping and gate-fitting days
(protocol revision/protocols/R-X4_forward_gate_split.md).

Runs the unchanged src/revision_e3.py analysis with (i) the R-X4 forward refits (experiments/R_X4; early stopping on
the first 15 selection days), (ii) gate-day predictions (last 14 selection days) in place of the selection-period
validation predictions: gates, lambda and conformal initial scores come from the gate days only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import src.revision_e3 as e3
from src.e8_features import FEAT
from src.final_eval import Split
from src.revision_common import save

FOLDS = json.loads(Path("revision/manifests/folds_forward_v2.json").read_text())
STATE = {}


def gate_preds(name, seed):
    """Predictions of one forward member on the gate days (A2: common 60-min population, as its validation set)."""
    key = (name, seed)
    if key not in STATE:
        split, anchor = STATE["split"], STATE["anchor"]
        day = split.ds.day.astype(str)
        idx = np.flatnonzero(np.isin(day, FOLDS["gate_days"]))
        if name == "A2":
            idx = idx[STATE["common"][idx]]
        p = e3.predict(split, name, seed, idx, anchor)
        p |= {"target": split.pv[split.ds.target_rows[idx]], "sample_ids": split.ds.sample_ids[idx],
              "origin_time_us": split.ds.origin_time.astype("datetime64[us]").astype(np.int64)[idx]}
        if name == "E6-S":  # event threshold from the member's own training (fit) days, as in E3
            p["tau"] = float(np.load(e3.ROOT / "E6-S" / "csi" / f"seed{seed}" / "fold_0" / "preds_val.npz")["tau"])
        STATE[key] = p
    return STATE[key]


def main():
    assert FOLDS["folds"][0]["train_days"] == e3.FOLDS["folds"][0]["train_days"], "STOP: fit days differ from E3"
    assert FOLDS["eval_days"] == e3.FOLDS["eval_days"], "STOP: evaluation days differ from E3"
    assert not set(FOLDS["gate_days"]) & set(FOLDS["folds"][0]["val_days"]), "STOP: gate days overlap early-stopping days"
    split = Split("dev", torch.device("cuda"))
    win = np.load(FEAT / "e8_windows_v1.npz")
    assert np.array_equal(win["sample_ids"], split.ds.sample_ids)
    STATE.update(split=split, anchor=e3.fit_anchor(split), common=win["eligible_common"])
    e3.ROOT = Path("experiments/R_X4")
    e3.FOLDS = FOLDS
    e3.sel_preds = gate_preds
    e3.Split = lambda name, device: split

    def save_x4(name, obj):
        obj = {**obj, "protocol": "revision/protocols/R-X4_forward_gate_split.md",
               "label": "FORWARD sensitivity (R-X4): early stopping on selection days 1-15, gates and conformal initial scores on days 16-29",
               "early_stopping_days": FOLDS["folds"][0]["val_range"], "gate_days": [FOLDS["gate_days"][0], FOLDS["gate_days"][-1]],
               "n_gate_days": len(FOLDS["gate_days"])}
        return save("x4_forward_gate_split.json", obj)

    e3.save = save_x4
    e3.main()


if __name__ == "__main__":
    main()
