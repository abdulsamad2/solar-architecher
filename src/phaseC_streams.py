"""Phase C data: out-of-fold development streams for G3''-F and E4-B-fit plus forecast-time regime signals.

Member (seed s, fold k) predicts only fold k's validation samples (3-member out-of-fold ensembles). G3''-F follows
its frozen routing (G3'' with valid 60-min history, else G3). g = E6-S implied event probability (frozen gate code).
Output: experiments/phaseC/oof_streams.npz (development data only).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.e8_features import FEAT, verify_frozen
from src.final_eval import FOLDS, SEEDS, Split, blend, gate_params, member_predict
from src.gate2_report import ramp_labels
from src.gate7b import raw_features

OUT = Path("experiments/phaseC/oof_streams.npz")


def sub(d: dict, m: np.ndarray) -> dict:
    return {k: (v if k == "quantile_levels" else v[m]) for k, v in d.items()}


def main():
    verify_frozen()
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    tau = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    rec = np.load(FEAT / "e8_record_features_v1.npz")
    split = Split("dev", torch.device("cuda"), np.concatenate([rec["P"], rec["M"]], 1).astype(np.float32))
    n = len(split.ds.sample_ids)
    acc = {"G3''-F": np.zeros((n, 9)), "E4-B-fit": np.zeros((n, 9))}
    g_acc, cnt = np.zeros(n), np.zeros(n)
    fold_of = np.full(n, -1)
    for k in FOLDS:
        fold = folds_doc["folds"][k]
        va = np.flatnonzero(np.isin(split.ds.day.astype(str), fold["val_days"]))
        fold_of[va] = k
        t_k = tau[str(k)]["training_labels_fit_days"]
        ok = split.h60_ok[va]
        for s in SEEDS:
            e4 = member_predict(split, "E4-B-fit", s, fold, va)
            e6 = member_predict(split, "E6-S", s, fold, va)
            q = np.empty((len(va), 9))
            if ok.any():
                a2 = member_predict(split, "A2", s, fold, va[ok])
                q[ok] = blend(a2, sub(e6, ok), gate_params("G3''", s, k), t_k)
            if (~ok).any():
                q[~ok] = blend(sub(e4, ~ok), sub(e6, ~ok), gate_params("G3", s, k), t_k)
            acc["G3''-F"][va] += q
            acc["E4-B-fit"][va] += e4["pred_q"]
            g_acc[va] += raw_features(e4, e6, t_k)["g"]
            cnt[va] += 1
            torch.cuda.empty_cache()
            print(f"fold {k} seed {s} done", flush=True)
    assert (cnt == 3).all() and (fold_of >= 0).all()
    q = {m: a / 3 for m, a in acc.items()}
    for m in q:
        assert (np.diff(q[m], axis=1) >= -1e-9).all()
    rl = ramp_labels("data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz", folds_doc)
    sid = split.ds.sample_ids
    day = split.ds.day.astype(str)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, sample_ids=sid, fold=fold_of, day=day,
                        origin_us=split.ds.origin_time.astype("datetime64[us]").astype(np.int64),
                        target_us=split.ds.target_time.astype("datetime64[us]").astype(np.int64),
                        y=split.pv[split.ds.target_rows], ramp=np.array([rl["is_ramp"][i] for i in sid]),
                        sunny=np.array([folds_doc["day_labels"][d] for d in day]), h60=split.h60_ok,
                        q_G3ppF=q["G3''-F"], q_E4=q["E4-B-fit"], g=g_acc / 3)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
