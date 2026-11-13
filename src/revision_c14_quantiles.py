"""C14 (review): how often issued power quantiles are physically implausible on the official test set.

Issued quantiles are not truncated (Sec. 3.2.3); this counts, per model, the forecasts with at least one of the nine
quantiles below zero or above the 30.1 kW nameplate capacity, and the forecasts whose median is negative.
Read-only: uses the saved frozen test predictions (experiments/final/test_predictions_v1.npz). POST HOC.
Run from the repository root: python -m src.revision_c14_quantiles
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CAP_KW = 30.1
MODELS = {"G3''-F": "q|G3''-F", "E4-B-fit": "q|E4-B-fit", "E6-S": "q|E6-S", "G4": "q|G4"}


def main():
    p = np.load("experiments/final/test_predictions_v1.npz")
    res = {"label": "POST HOC (test); frozen evaluation unchanged", "capacity_kw": CAP_KW, "n": int(len(p["sample_ids"]))}
    for name, key in MODELS.items():
        q = p[key]
        res[name] = {
            "share_any_quantile_negative": float((q < 0).any(1).mean()),
            "share_median_negative": float((q[:, 4] < 0).mean()),
            "share_any_quantile_above_capacity": float((q > CAP_KW).any(1).mean()),
            "share_median_above_capacity": float((q[:, 4] > CAP_KW).mean()),
            "min_quantile_kw": float(q.min()),
            "max_quantile_kw": float(q.max()),
        }
    out = Path("revision/results/c14_quantile_plausibility.json")
    out.write_text(json.dumps(res, indent=2))
    print("saved", out)
    for name in MODELS:
        r = res[name]
        print(f"{name:9s} any<0 {r['share_any_quantile_negative']:.2%}  median<0 {r['share_median_negative']:.3%}  "
              f"any>cap {r['share_any_quantile_above_capacity']:.2%}  median>cap {r['share_median_above_capacity']:.3%}")


if __name__ == "__main__":
    main()
