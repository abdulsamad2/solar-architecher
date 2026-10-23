"""Physical-plausibility diagnostics for K-quantile forecasts (evaluation only; numpy).

  power_consistency_mae / median_abs : |P_true - K50 * max(P_clear, c_min)| in kW
  frac_k_neg / frac_k_over            : share of predicted quantile values with K < 0 / K > K_max
  ramp_direction_accuracy             : on ramp samples, share with sign(K50 - K_now) == sign(K_true - K_now)
  ramp_magnitude_error                : on ramp samples, mean | |K_true - K_now| - |K50 - K_now| |
"""
from __future__ import annotations

import numpy as np


def physics_diagnostics(k_q, k_true, k_now, p_true_kw, p_clear_kw, is_ramp, k_max: float = 1.5,
                        c_min_kw: float = 1.0, median_index: int = 4) -> dict:
    k_q, k_true, k_now = np.asarray(k_q, float), np.asarray(k_true, float), np.asarray(k_now, float)
    p_true, pc = np.asarray(p_true_kw, float), np.maximum(np.asarray(p_clear_kw, float), c_min_kw)
    is_ramp = np.asarray(is_ramp, bool)
    k50 = k_q[:, median_index]
    err = np.abs(p_true - k50 * pc)
    d_true, d_pred = k_true - k_now, k50 - k_now
    out = {"power_consistency_mae_kw": float(err.mean()), "power_consistency_median_abs_kw": float(np.median(err)),
           "frac_k_neg": float((k_q < 0).mean()), "frac_k_over_kmax": float((k_q > k_max).mean()), "k_max": k_max}
    if is_ramp.any():
        out["ramp_direction_accuracy"] = float(np.mean(np.sign(d_pred[is_ramp]) == np.sign(d_true[is_ramp])))
        out["ramp_magnitude_error"] = float(np.mean(np.abs(np.abs(d_true[is_ramp]) - np.abs(d_pred[is_ramp]))))
        out["n_ramp"] = int(is_ramp.sum())
    return out
