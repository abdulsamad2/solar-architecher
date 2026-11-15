"""Shared helpers for R-Z1..R-Z3: the issue-time ramp region of the outcome space and the past-only calibrated intervals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.cal2 import ALPHA
from src.calibrate import base_interval
from src.clear_sky import ClearSkyPV
from src.final_eval import rolling_conformal_daily
from src.revision_c07 import past_only_init
from src.revision_common import LEVELS


def ramp_region(c):
    """Clear-sky power at the target (kW, floored at 1 as in the frozen ramp rule) and the non-ramp band (L, U) in kW:
    an outcome z is a ramp iff |z / pc - k_now| >= tau, i.e. z <= L or z >= U. Known at issue time."""
    rd = json.loads(Path("reports/test_ramp_definition_frozen.json").read_text())
    pc = np.maximum(ClearSkyPV(rd["anchor_coef"])(c["rec_t"]), 1.0)[c["ds"].target_rows]
    lo, hi = (c["k_now"] - c["tau"]) * pc, (c["k_now"] + c["tau"]) * pc
    dk = c["y"] / pc - c["k_now"]
    assert np.array_equal(np.abs(dk) >= c["tau"], c["ramp"]), "STOP: ramp region does not reproduce the frozen ramp labels"
    return pc, lo, hi


def calibrated_intervals(c, q, model):
    """C07 past-only rolling conformal intervals at 50/80/90% (reproduced against the stored C07 result)."""
    cal = np.load("experiments/final/dev_oof_calibration_v1.npz")
    W = json.loads(Path("experiments/final/dev_oof_calibration_v1.json").read_text())["W"][model]
    c07 = json.loads(Path("revision/results/c07_conformal_pastonly.json").read_text())["past_only"][model]
    cal_org = cal["origin_us"].astype(np.int64)
    days = np.unique(c["day"])
    start = {d: c["org"][c["day"] == d].min() for d in days}
    out = {}
    for nm, a in ALPHA.items():
        lo, hi = base_interval(q, LEVELS, nm)
        init = {d: past_only_init(cal_org, cal[f"{model}|{nm}"], start[d], W)[0] for d in days}
        clo, chi = rolling_conformal_daily(c["org"], c["tgt"], c["day"], lo, hi, c["y"], a, W, init)
        cov = float(np.mean((c["y"] >= clo) & (c["y"] <= chi)))
        assert abs(cov - c07[nm]["overall"]["coverage"]) < 1e-9, f"STOP: C07 past-only not reproduced ({model} {nm})"
        out[nm] = (clo, chi)
    return out
