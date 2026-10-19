"""Causal first-crossing ramp labels on the clear-sky index.

Thresholds must be chosen on training data only.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def first_crossing_labels(
    k_now, k_future, threshold: float, bin_edges_min: Sequence[int] = (5, 10, 15)
) -> dict[str, np.ndarray]:
    """k_now [N], k_future [N, H] at minutes 1..H.

    Returns per sample:
      arrival_min: first minute with |K_{t+h} - K_t| >= threshold, or 0 if none
      time_class:  index of the arrival bin (bins (0,5], (5,10], (10,15]); len(bins) == no event
      severity:    Delta K at the arrival minute (0 if no event)
    """
    k_now = np.asarray(k_now, float)
    k_future = np.asarray(k_future, float)
    delta = k_future - k_now[:, None]
    crossed = np.abs(delta) >= threshold
    has_event = crossed.any(axis=1)
    first = crossed.argmax(axis=1)  # 0-based minute index

    arrival = np.where(has_event, first + 1, 0)
    edges = np.asarray(bin_edges_min)
    time_class = np.where(has_event, np.searchsorted(edges, arrival, side="left"), len(edges))
    severity = np.where(has_event, delta[np.arange(len(delta)), first], 0.0)
    return {"arrival_min": arrival, "time_class": time_class, "severity": severity}
