"""Revision E3: chronological SKIPP'D split (protocol revision/protocols/R-E3_forward.md).

One fold: train_days = fit period, val_days = selection/calibration period (early stopping, gate fitting,
conformal initial scores). The evaluation period is stored separately and is never passed to training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SRC = Path("data/manifests/folds_blocked5_v1.json")
OUT = Path("revision/manifests/folds_forward_v1.json")
OUT_V2 = Path("revision/manifests/folds_forward_v2.json")  # R-X4: selection days split into early-stopping / gate days
FIT_END, SEL_START, SEL_END, EVAL_START = "2019-06-30", "2019-07-02", "2019-07-31", "2019-08-02"


def build(v2: bool = False) -> dict:
    src = json.loads(SRC.read_text())
    days = sorted(set(src["folds"][0]["train_days"]) | set(src["folds"][0]["val_days"]) | set(src["folds"][0]["purged_days"]))
    fit = [d for d in days if d <= FIT_END]
    sel = [d for d in days if SEL_START <= d <= SEL_END]
    ev = [d for d in days if d >= EVAL_START]
    purged = [d for d in days if d not in set(fit) | set(sel) | set(ev)]
    assert max(fit) < min(sel) and max(sel) < min(ev), "STOP: periods not chronological"
    gate = []
    if v2:  # R-X4: first half of the selection days for early stopping, second half for gates and conformal scores
        sel, gate = sel[:(len(sel) + 1) // 2], sel[(len(sel) + 1) // 2:]
    doc = {"n_folds": 1, "purge_days": 1, "day_labels": src["day_labels"],
           "folds": [{"fold": 0, "train_days": fit, "val_days": sel, "purged_days": purged,
                      "val_range": [sel[0], sel[-1]], "n_train_days": len(fit), "n_val_days": len(sel)}],
           "eval_days": ev, **({"gate_days": gate} if v2 else {}),
           "periods": {"fit": [fit[0], fit[-1]], "selection": [sel[0], sel[-1]], "evaluation": [ev[0], ev[-1]]}
           | ({"gate": [gate[0], gate[-1]]} if v2 else {}),
           "source_folds_sha256": src["content_sha256"]}
    body = json.dumps({k: v for k, v in doc.items()}, sort_keys=True).encode()
    doc["content_sha256"] = hashlib.sha256(body).hexdigest()
    out = OUT_V2 if v2 else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    return doc


if __name__ == "__main__":
    import sys

    d = build("--v2" in sys.argv)
    f = d["folds"][0]
    print({"fit_days": len(f["train_days"]), "selection_days": len(f["val_days"]), "eval_days": len(d["eval_days"]),
           "gate_days": len(d.get("gate_days", [])), "purged": f["purged_days"], "periods": d["periods"]})
