"""Revision C24: SHA-256 inventory of the revision artefacts (checkpoints and member dumps are not versioned).

Writes revision/hash_inventory.json: every best.pt of the revision arms (R_E1, R_E3, R_E4), the post hoc member
dumps, the forward split manifest and every raw result JSON.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

GLOBS = ["experiments/R_E1/**/best.pt", "experiments/R_E3/**/best.pt", "experiments/R_E4/**/best.pt",
         "experiments/R_posthoc/*.npz", "revision/manifests/*.json", "revision/results/*.json"]


def main():
    files = sorted({p for g in GLOBS for p in Path(".").glob(g)})
    inv = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest().upper() for p in files}
    Path("revision/hash_inventory.json").write_text(json.dumps({"n_files": len(inv), "files": inv}, indent=1))
    print(len(inv), "files hashed")


if __name__ == "__main__":
    main()
