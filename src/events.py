"""Phase 6 event labels on real data: K trajectory over the next 15 minutes, first-crossing arrival, severity.

Labels only; never model inputs. Per fold, K uses the fold's training-day clear-sky anchor and the event
threshold tau is the 90th percentile of |K(t+15) - K(t)| over the fold's TRAINING samples (the Gate 2 /
E4-CAL2 ramp definition), so ramp strata and event labels share one threshold.

Arrival classes: 0 = (0, 5] min, 1 = (5, 10] min, 2 = (10, 15] min, 3 = no event.
Severity: dK15 = K(t+15) - K(t) (the forecast target in K-space), available for every sample.
"""
from __future__ import annotations

import numpy as np

from src.ramp_labels import first_crossing_labels
from src.skippd import _nearest

MIN_US = 60_000_000
TOL_US = 30_000_000
BIN_EDGES = (5, 10, 15)


def future_rows(rec_us: np.ndarray, origin_us: np.ndarray, horizon: int = 15) -> np.ndarray:
    """[N, horizon] record rows nearest to origin + h minutes (h = 1..horizon) within +-30 s; -1 if missing."""
    return np.stack([_nearest(rec_us, origin_us + h * MIN_US, TOL_US) for h in range(1, horizon + 1)], axis=1)


def event_labels(k_now: np.ndarray, k_future: np.ndarray, k_target: np.ndarray, tau: float) -> dict:
    """k_future [N, 15] (NaN where a minute is missing). An event label is valid only if every minute up to
    the first crossing (or all 15 if none) is present; severity is always valid."""
    filled = np.where(np.isnan(k_future), k_now[:, None], k_future)
    lab = first_crossing_labels(k_now, filled, tau, BIN_EDGES)
    missing = np.isnan(k_future)
    upto = np.where(lab["arrival_min"] > 0, lab["arrival_min"], k_future.shape[1])
    valid = ~np.array([missing[i, :upto[i]].any() for i in range(len(k_now))])
    return {"time_class": lab["time_class"], "arrival_min": lab["arrival_min"],
            "severity": k_target - k_now, "event_valid": valid, "tau": tau}


def implied_event_probability(pred_q_k: np.ndarray, levels, k_now: np.ndarray, tau: float) -> np.ndarray:
    """P(|K(t+15) - K(t)| >= tau) implied by a quantile forecast of K(t+15) (the direct-baseline comparator)."""
    from src.calibrate import extended_quantile_function

    taus = np.linspace(0.001, 0.999, 999)
    qf = extended_quantile_function(pred_q_k, levels, taus, floor=-np.inf)
    below = (qf <= (k_now - tau)[:, None]).mean(axis=1)
    above = (qf >= (k_now + tau)[:, None]).mean(axis=1)
    return np.clip(below + above, 0.0, 1.0)
