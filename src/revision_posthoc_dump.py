"""Revision post hoc: per-member test predictions of frozen (and revision) experts.

Each (seed, fold) member is run on every official test sample it can predict (60-min experts: eligible samples
only, NaN elsewhere), with exactly the inputs of src/final_eval.member_predict. Output (gitignored npz):
experiments/R_posthoc/test_members_<model>.npz with pred_q [15, N, 9], member context and, for E6-S, event outputs.
Checks: ensembles of the dumped E4-B-fit / E6-S / G3''-F / G4 members reproduce the frozen test predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.clear_sky import clear_sky_index
from src.final_eval import FEAT, SEEDS, Split, blend, gate_params, member_anchor, record_table_unsorted
from src.train import build_model, event_outputs, predict_raw

OUT = Path("experiments/R_posthoc")
EXPERTS = {  # name: (run root, window kind)
    "E4-B-fit": (Path("experiments/E4_calib/v1/csi"), "H15"),
    "E6-S": (Path("experiments/E6/v1/E6-S/csi"), "H15"),
    "A2": (Path("experiments/E9/v1/A2/csi"), "H60"),
    "D0": (Path("experiments/E8/v1/D0/csi"), "H15e8"),
    "D2": (Path("experiments/E8/v1/D2/csi"), "H60"),
    "P15": (Path("experiments/R_E1/P15/csi"), "H15"),
    "P60": (Path("experiments/R_E1/P60/csi"), "H60"),
    "A2-H15": (Path("experiments/R_E4/A2_H15/csi"), "H15e8"),
}


def run_member(split: Split, name: str, seed: int, fold: dict, tau: float | None, prev: np.ndarray | None = None) -> dict:
    """prev (C06 sensitivity): per-record index of the record ~60 s earlier (-1 if none); every PV/K input then
    uses that earlier record, so forward-averaged PV is available by the issue time. Images are unchanged."""
    root, kind = EXPERTS[name]
    run = root / f"seed{seed}"
    cfg = json.loads((run / "config.json").read_text())
    fdir = run / f"fold_{fold['fold']}"
    pc = member_anchor(fold)(split.rec_t)
    k_rec = clear_sky_index(split.pv, pc, 1.0)
    if prev is not None:  # records without an earlier record keep their own value; such samples are excluded by the caller
        k_rec = np.where(prev >= 0, k_rec[np.maximum(prev, 0)], k_rec)
    data, n = split.data, len(split.ds.sample_ids)
    data.set_history_feature(k_rec)
    data.residual = False
    data.k_now = torch.tensor(k_rec[split.ds.hist_rows[:, -1]], dtype=torch.float32, device=split.device)
    idx = np.arange(n)
    if kind == "H60":
        idx = idx[split.h60_ok]
        data.set_window(split.h60_rows)
        std = json.loads((fdir / "e8_standardisation.json").read_text())
        mask = np.array(std["mask"], np.float32)
        table = split.table if mask.any() else np.zeros((len(split.pv), 11), np.float32)
        data.set_extra(((table - np.array(std["mu"])) / np.array(std["sd"]) * mask).astype(np.float32))
    elif kind == "H15e8":
        std = json.loads((fdir / "e8_standardisation.json").read_text())
        assert not np.any(std["mask"]), "H15e8 members with active features need the feature table"
        data.set_window(split.ds.hist_rows)
        data.set_extra(np.zeros((len(split.pv), 11), np.float32))
    else:
        data.set_window(split.ds.hist_rows)
        data.set_extra(None)
    net = build_model(cfg).to(split.device)
    net.encoder.cnn_bf16 = cfg["cnn_bf16"]
    net.load_state_dict(torch.load(fdir / "best.pt", map_location=split.device))
    net.eval()
    it = torch.as_tensor(idx, device=split.device)
    q = np.full((n, 9), np.nan)
    q[idx] = predict_raw(net, data, it).cpu().numpy().astype(float) * np.maximum(pc[split.ds.target_rows[idx]], 1.0)[:, None]
    out = {"pred_q": q, "pclear_target_kw": pc[split.ds.target_rows], "pclear_now_kw": pc[split.ds.hist_rows[:, -1]]}
    if prev is not None:
        # gate features compute K(t) = pv_now / pclear_now; use the earlier record's PV and anchor consistently
        now = prev[split.ds.hist_rows[:, -1]]
        out["pclear_now_kw"] = pc[np.where(now >= 0, now, split.ds.hist_rows[:, -1])]
        out["pv_now_kw"] = split.pv[np.where(now >= 0, now, split.ds.hist_rows[:, -1])]
    if cfg.get("head") == "event":
        eo = event_outputs(net, data, it, tau)
        tp, pa = np.full((n, 4), np.nan), np.full(n, np.nan)
        tp[idx], pa[idx] = eo["time_probs"], eo["p_abs_change_ge_tau"]
        out |= {"time_probs": tp, "p_abs_change_ge_tau": pa}
    del net
    torch.cuda.empty_cache()
    return out


def member_dict(d: dict, i: int, pv_now) -> dict:
    return {"pred_q": d["pred_q"][i], "pclear_target_kw": d["pclear_target_kw"][i], "pclear_now_kw": d["pclear_now_kw"][i],
            "pv_now_kw": pv_now, "quantile_levels": np.array([.05, .1, .2, .3, .5, .7, .8, .9, .95])}


def main(models):
    import h5py

    from src.final_eval import H5, MAN
    with h5py.File(H5, "r") as f:
        gray = np.round(f["test/images_log"][...].astype(np.float32).mean(-1)).astype(np.uint8)
    m = np.load(MAN)
    sel = m["hdf5_group"] == 1
    t_test = m["times_us"][sel][np.argsort(m["hdf5_row"][sel])].astype("datetime64[us]")
    table = record_table_unsorted(t_test, gray, FEAT / "gray_u8_test.npy")
    split = Split("test", torch.device("cuda"), table)
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    frozen_tau = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    pv_now = split.pv[split.ds.hist_rows[:, -1]]
    manifest = {}
    for name in models:
        members, acc = [], {}
        for s in SEEDS:
            for k in range(5):
                fold = {**folds_doc["folds"][k]}
                r = run_member(split, name, s, fold, frozen_tau[str(k)]["training_labels_fit_days"])
                members.append((s, k))
                for key, v in r.items():
                    acc.setdefault(key, []).append(v)
        arr = {key: np.stack(v) for key, v in acc.items()}
        p = OUT / f"test_members_{name}.npz"
        np.savez_compressed(p, sample_ids=split.ds.sample_ids, members=np.array(members), pv_now_kw=pv_now, **arr)
        manifest[name] = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest().upper(),
                          "n_members": len(members), "n_predicted": int(np.isfinite(arr["pred_q"][0, :, 0]).sum())}
        print(name, manifest[name])
    # reproduction checks against the frozen 15-member test predictions
    frozen = np.load("experiments/final/test_predictions_v1.npz")
    have = lambda n: (OUT / f"test_members_{n}.npz").exists()
    load = lambda n: dict(np.load(OUT / f"test_members_{n}.npz"))
    checks = {}
    for n in ("E4-B-fit", "E6-S"):
        if have(n):
            checks[n] = float(np.abs(np.nanmean(load(n)["pred_q"], 0) - frozen[f"q|{n}"]).max())
    if have("E4-B-fit") and have("E6-S") and have("A2"):
        d, r, a2 = load("E4-B-fit"), load("E6-S"), load("A2")
        h60 = split.h60_ok
        g4, g3pp = [], []
        for i, (s, k) in enumerate(d["members"]):
            tau = frozen_tau[str(k)]["training_labels_fit_days"]
            md, mr, ma = member_dict(d, i, pv_now), member_dict(r, i, pv_now), member_dict(a2, i, pv_now)
            g4.append(blend(md, mr, gate_params("G4", s, k), tau))
            out = blend(md, mr, gate_params("G3", s, k), tau)
            sub = {key: (v[h60] if np.ndim(v) and len(v) == len(h60) else v) for key, v in ma.items()}
            subr = {key: (v[h60] if np.ndim(v) and len(v) == len(h60) else v) for key, v in mr.items()}
            out[h60] = blend(sub, subr, gate_params("G3''", s, k), tau)
            g3pp.append(out)
        checks["G4"] = float(np.abs(np.mean(g4, 0) - frozen["q|G4"]).max())
        checks["G3''-F"] = float(np.abs(np.mean(g3pp, 0) - frozen["q|G3''-F"]).max())
    manifest["reproduction_max_abs_kw"] = checks
    prev = OUT / "manifest.json"
    old = json.loads(prev.read_text()) if prev.exists() else {}
    prev.write_text(json.dumps(old | manifest, indent=1))
    print("reproduction (max |diff| kW):", checks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["E4-B-fit", "E6-S", "A2", "D0", "D2"])
    main(ap.parse_args().models)
