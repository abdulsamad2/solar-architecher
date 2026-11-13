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
FIT_END, SEL_START, SEL_END, EVAL_START = "2019-06-30", "2019-07-02", "2019-07-31", "2019-08-02"


def build() -> dict:
    src = json.loads(SRC.read_text())
    days = sorted(set(src["folds"][0]["train_days"]) | set(src["folds"][0]["val_days"]) | set(src["folds"][0]["purged_days"]))
    fit = [d for d in days if d <= FIT_END]
    sel = [d for d in days if SEL_START <= d <= SEL_END]
    ev = [d for d in days if d >= EVAL_START]
    purged = [d for d in days if d not in set(fit) | set(sel) | set(ev)]
    assert max(fit) < min(sel) and max(sel) < min(ev), "STOP: periods not chronological"
    doc = {"n_folds": 1, "purge_days": 1, "day_labels": src["day_labels"],
           "folds": [{"fold": 0, "train_days": fit, "val_days": sel, "purged_days": purged,
                      "val_range": [sel[0], sel[-1]], "n_train_days": len(fit), "n_val_days": len(sel)}],
           "eval_days": ev,
           "periods": {"fit": [fit[0], fit[-1]], "selection": [sel[0], sel[-1]], "evaluation": [ev[0], ev[-1]]},
           "source_folds_sha256": src["content_sha256"]}
    body = json.dumps({k: v for k, v in doc.items()}, sort_keys=True).encode()
    doc["content_sha256"] = hashlib.sha256(body).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1))
    return doc


if __name__ == "__main__":
    d = build()
    f = d["folds"][0]
    print({"fit_days": len(f["train_days"]), "selection_days": len(f["val_days"]), "eval_days": len(d["eval_days"]),
           "purged": f["purged_days"], "periods": d["periods"]})
