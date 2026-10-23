"""Post-hoc (descriptive) check of CRPS-estimator comparability for Phase A2, on committed predictions only.

SkyGPT reports the empirical-ensemble CRPS; our models are scored by the piecewise-linear quantile CRPS over 9 levels
(5-95%). On the SUNSET port's 10 committed members we compute both estimators: the empirical CRPS, and the quantile
CRPS from the members' 9 empirical quantiles. Their difference indicates how much the estimator alone moves CRPS.
Output: reports/phaseA_a2_estimator_check.json
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from src.metrics import crps_quantile_interp
from src.phaseA_a2 import H5X, LEVELS, ensemble_crps


def main():
    p = np.load("experiments/phaseA/a2_predictions.npz")
    with h5py.File(H5X, "r") as f:
        y = f["test/pv_pred"][:, 14].astype(float)
    sun = p["sunset_members"]
    emp = ensemble_crps(sun, y).mean()
    qnt = crps_quantile_interp(np.quantile(sun, LEVELS, axis=0).T, y, LEVELS).mean()
    out = {"SUNSET_port_empirical_crps": float(emp), "SUNSET_port_quantile_crps_9_levels": float(qnt),
           "quantile_minus_empirical": float(qnt - emp), "relative": float(qnt / emp - 1)}
    Path("reports/phaseA_a2_estimator_check.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
