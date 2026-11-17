"""Revision R-R12: ramp warnings (as R-R4) and reserve at equal reliability (as R-R5) of method v2 on the official
test set (protocol revision/protocols/R-R12_warnings_reserve_method_v2.md). Post hoc test, no training, CPU only.
Run: python -m src.revision_r12_warnings_reserve_v2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import revision_r4_ramp_warnings as r4
from src import revision_r5_reserve as r5
from src.metrics import crps_quantile_interp
from src.revision_common import LEVELS, test_context

PROTOCOL = "revision/protocols/R-R12_warnings_reserve_method_v2.md"
RES = Path("revision/results")


def close(a, b, path=""):
    """Two result trees agree to 1e-9 (numbers) and exactly (everything else); 'forecaster' is a key added in R-R12."""
    if isinstance(a, dict):
        assert set(a) - {"forecaster"} == set(b) - {"forecaster"}, f"STOP: keys differ at {path}"
        for k in b:
            if k != "forecaster":
                close(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"STOP: length differs at {path}"
        for i, (x, z) in enumerate(zip(a, b)):
            close(x, z, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert abs(float(a) - float(b)) < 1e-9, f"STOP: {path}: {a} != {b}"
    else:
        assert a == b, f"STOP: {path}: {a} != {b}"


def main():
    c = test_context()
    z = np.load("D:/skippd_hires/R_M3/eval/test_quantiles.npz")
    assert np.array_equal(z["sample_ids"], c["sample_ids"])
    v2 = z["method_v2"]
    r9 = json.loads((RES / "r9_method_v2_official_test.json").read_text())
    assert abs(crps_quantile_interp(v2, c["y"], LEVELS).mean() - r9["scores"]["method v2"]["crps"]) < 1e-9, "STOP: method v2 (R-R9) not reproduced"

    print("=== check: the scripts with their defaults reproduce R-R4 and R-R5", flush=True)
    close(json.loads(json.dumps(r4.main(out=None))), json.loads((RES / "r4_ramp_warnings.json").read_text()))
    close(json.loads(json.dumps(r5.main(out=None))), json.loads((RES / "r5_reserve_equal_reliability.json").read_text()))
    print("=== reproduced", flush=True)

    print("\n=== R-R12 warnings, method v2 as the forecaster", flush=True)
    r4.main(forecaster=v2, out="r12_ramp_warnings_method_v2.json", protocol=PROTOCOL)
    print("\n=== R-R12 reserve, method v2 as the forecaster", flush=True)
    r5.main(forecaster=v2, out="r12_reserve_method_v2.json", protocol=PROTOCOL)


if __name__ == "__main__":
    main()
