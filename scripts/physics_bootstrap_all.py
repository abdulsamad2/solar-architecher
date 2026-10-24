"""Phase F descriptive supplement: paired day bootstrap of EVERY physics arm vs A0 (development out-of-fold).

The pre-registered rule selected no best arm (none had ramp CRPS <= A0), so these comparisons are descriptive.
Output: reports/phaseF_bootstrap_all.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_physics_ablation import arm_oof, context, evaluate  # noqa: E402
from src.final_eval import day_bootstrap  # noqa: E402


def main():
    z = np.load(ROOT / "experiments/phaseC/oof_streams.npz")
    ctx = context(z)
    day, sunny, ramp = z["day"], z["sunny"].astype(bool), z["ramp"].astype(bool)
    res = {a: evaluate(arm_oof(a, z), z, ctx) for a in ("A0", "A1", "R", "A3", "A4", "A5")}
    out = {}
    for a in ("A1", "R", "A3", "A4", "A5"):
        d = res[a]["_crps_per_sample"] - res["A0"]["_crps_per_sample"]
        out[a] = {"dCRPS_overall": day_bootstrap(d, day, sunny),
                  "dCRPS_cloudy": day_bootstrap(d[~sunny], day[~sunny], sunny[~sunny]),
                  "dCRPS_ramp": day_bootstrap(d[ramp], day[ramp], sunny[ramp]),
                  "dCov90_ramp": day_bootstrap((res[a]["_cov90"] - res["A0"]["_cov90"])[ramp], day[ramp], sunny[ramp])}
        print(a, {k: [round(x, 4) for x in v] for k, v in out[a].items()}, flush=True)
    out["physics_diagnostics"] = {a: r["physics"] for a, r in res.items()}
    (ROOT / "reports/phaseF_bootstrap_all.json").write_text(json.dumps(out, indent=1, default=float))
    for a, p in out["physics_diagnostics"].items():
        print(a, {k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()})


if __name__ == "__main__":
    main()
