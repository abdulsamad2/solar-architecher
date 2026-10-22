"""Freeze G3'' gate parameters (E9 Stage G recipe; deterministic L-BFGS refit on calibration days only).

Writes experiments/E9/v1/G3pp_params/seed{s}/fold_{k}/params.json and verifies the refit reproduces the
recorded E9 Stage-G validation CRPS (reports/e9_gate.json) to 1e-9.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.e8_features import FEAT, verify_frozen
from src.gate2_report import ramp_labels, strata
from src.gate7b import apply_gate, raw_features, select_and_fit
from src.metrics import crps_quantile_interp

OUT = Path("experiments/E9/v1/G3pp_params")
A2 = Path("experiments/E9/v1/A2/csi")
RESID = Path("experiments/E6/v1/E6-S/csi")


def main():
    verify_frozen()
    frozen = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    win = np.load(FEAT / "e8_windows_v1.npz")
    common = set(win["sample_ids"][win["eligible_common"]])
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    rl = ramp_labels("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz", folds_doc)
    recorded = json.loads(Path("reports/e9_gate.json").read_text())
    for s in (42, 123, 456):
        qs, metas = [], []
        for k in range(5):
            tau = frozen[str(k)]["training_labels_fit_days"]
            sp = {}
            for split in ("cal", "val"):
                r = dict(np.load(RESID / f"seed{s}" / f"fold_{k}" / f"preds_{split}.npz"))
                keep = np.array([i in common for i in r["sample_ids"]])
                r = {kk: (v[keep] if np.ndim(v) and len(v) == len(keep) else v) for kk, v in r.items()}
                d = dict(np.load(A2 / f"seed{s}" / f"fold_{k}" / f"preds_{split}.npz"))
                assert np.array_equal(d["sample_ids"], r["sample_ids"])
                sp[split] = (d, r)
            (dc, rc), (dv, rv) = sp["cal"], sp["val"]
            order = np.argsort(rc["origin_time_us"], kind="stable")
            p = select_and_fit(raw_features(dc, rc, tau), ("g", "d"), dc["pred_q"], rc["pred_q"], rc["target"],
                               rc["quantile_levels"], order)
            p["tau"] = tau
            (OUT / f"seed{s}" / f"fold_{k}").mkdir(parents=True, exist_ok=True)
            (OUT / f"seed{s}" / f"fold_{k}" / "params.json").write_text(json.dumps(p, indent=1))
            w = apply_gate(p, raw_features(dv, rv, tau))
            qs.append(w[:, None] * dv["pred_q"] + (1 - w[:, None]) * rv["pred_q"])
            metas.append(rv)
        cat = lambda key: np.concatenate([v[key] for v in metas])
        m = {key: cat(key) for key in ("sample_ids", "target", "clear_day", "fold", "day", "origin_time_us")}
        st = strata(m, rl["is_ramp"])
        crps = crps_quantile_interp(np.concatenate(qs), m["target"], metas[0]["quantile_levels"])
        for g in ("overall", "ramp", "sunny", "cloudy"):
            got, want = float(crps[st[g]].mean()), recorded["per_seed"][str(s)]["raw_crps"]["G3''"][g]
            assert abs(got - want) < 1e-9, f"STOP: G3'' refit does not reproduce E9 ({s}, {g}): {got} vs {want}"
        print(f"seed {s}: G3'' params frozen; E9 CRPS reproduced exactly")


if __name__ == "__main__":
    main()
