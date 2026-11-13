"""Revision C22: batch-1 end-to-end latency of G3''-F (protocol revision/protocols/R-C22_C21_latency_external.md).

All 45 frozen member networks (E4-B-fit, E6-S, A2; 15 each) are resident on the GPU. For each forecast the timed
steps are: preprocessing (clear-sky anchor per fold, K history, image gather/normalise), 15 members of each
expert of the branch, gate + level-wise ensemble, conformal update (insert one matured score, query, widen).
"""
from __future__ import annotations

import bisect
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.cal2 import ALPHA, conformal_q, widen
from src.calibrate import base_interval
from src.final_eval import FEAT, H5, MAN, SEEDS, Split, blend, gate_params, member_anchor, record_table_unsorted
from src.revision_common import LEVELS, save
from src.train import build_model, predict_raw

EXP = {"E4-B-fit": Path("experiments/E4_calib/v1/csi"), "E6-S": Path("experiments/E6/v1/E6-S/csi"), "A2": Path("experiments/E9/v1/A2/csi")}


class One:
    """Batch-1 input adapter with the GpuSeq call signature used by predict_raw."""

    def __init__(self, x, hist, extra, k_now):
        self.x, self.hist, self.extra, self.k_now, self.residual = x, hist, extra, k_now, False

    def __call__(self, ib):
        return (self.x, self.hist) if self.extra is None else (self.x, self.hist, self.extra)


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def main(n=500, warm=50):
    import h5py
    with h5py.File(H5, "r") as f:
        gray = np.round(f["test/images_log"][...].astype(np.float32).mean(-1)).astype(np.uint8)
    m = np.load(MAN)
    sel = m["hdf5_group"] == 1
    t_test = m["times_us"][sel][np.argsort(m["hdf5_row"][sel])].astype("datetime64[us]")
    split = Split("test", torch.device("cuda"), record_table_unsorted(t_test, gray, FEAT / "gray_u8_test.npy"))
    dev = torch.device("cuda")
    folds = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())["folds"]
    tau_f = json.loads(Path("reports/phase6_frozen_thresholds.json").read_text())
    anchors = {k: member_anchor(folds[k]) for k in range(5)}
    nets, params = {}, {}
    for name, root in EXP.items():
        for s in SEEDS:
            cfg = json.loads((root / f"seed{s}" / "config.json").read_text())
            for k in range(5):
                net = build_model(cfg).to(dev)
                net.encoder.cnn_bf16 = cfg["cnn_bf16"]
                net.load_state_dict(torch.load(root / f"seed{s}" / f"fold_{k}" / "best.pt", map_location=dev))
                nets[(name, s, k)] = net.eval()
        params[name] = int(sum(p.numel() for p in nets[(name, SEEDS[0], 0)].parameters()))
    gp = {(g, s, k): gate_params(g, s, k) for g in ("G3", "G3''") for s in SEEDS for k in range(5)}
    cal = np.load("experiments/final/dev_oof_calibration_v1.npz")
    window = sorted(cal["G3''-F|90"][-4620:].tolist())
    rng = np.random.default_rng(0)
    idx_all = rng.choice(len(split.ds.sample_ids), n + warm, replace=False)
    zeros_extra = torch.zeros(1, 16, 11, device=dev)
    times = {"G3''": [], "G3": []}
    images = split.data.images
    for it, i in enumerate(idx_all):
        for branch in ("G3''", "G3"):
            if branch == "G3''" and not split.h60_ok[i]:
                continue
            rows = split.h60_rows[i] if branch == "G3''" else split.ds.hist_rows[i]
            t0 = sync()
            # preprocessing
            rec = np.concatenate([rows, split.ds.hist_rows[i], [split.ds.target_rows[i]]])
            pre = {}
            for k in range(5):
                pc = anchors[k](split.rec_t[rec])
                kk = split.pv[rec] / np.maximum(pc, 1.0)
                pre[k] = {"hist": torch.tensor(kk[None, :16], dtype=torch.float32, device=dev),
                          "hist15": torch.tensor(kk[None, 16:32], dtype=torch.float32, device=dev),
                          "k_now": torch.tensor([kk[31]], dtype=torch.float32, device=dev), "pc_t": float(max(pc[-1], 1.0)),
                          "pc_n": float(pc[31])}
            x = images[torch.as_tensor(rows, device=dev)][None].permute(0, 1, 4, 2, 3).float().div_(255.0)
            x15 = images[torch.as_tensor(split.ds.hist_rows[i], device=dev)][None].permute(0, 1, 4, 2, 3).float().div_(255.0)
            t1 = sync()
            # experts (15 members each)
            outs = {}
            direct = "A2" if branch == "G3''" else "E4-B-fit"
            for s in SEEDS:
                for k in range(5):
                    p = pre[k]
                    if direct == "A2":
                        qd = predict_raw(nets[("A2", s, k)], One(x, p["hist"], zeros_extra, p["k_now"]), [0])
                    else:
                        qd = predict_raw(nets[("E4-B-fit", s, k)], One(x15, p["hist15"], None, p["k_now"]), [0])
                    qr = predict_raw(nets[("E6-S", s, k)], One(x15, p["hist15"], None, p["k_now"]), [0])
                    outs[(s, k)] = (qd.cpu().numpy().astype(float) * p["pc_t"], qr.cpu().numpy().astype(float) * p["pc_t"])
            t2 = sync()
            # gate + ensemble
            mem = []
            for (s, k), (qd, qr) in outs.items():
                ctx = {"pclear_target_kw": np.array([pre[k]["pc_t"]]), "pclear_now_kw": np.array([pre[k]["pc_n"]]),
                       "pv_now_kw": np.array([split.pv[split.ds.hist_rows[i, -1]]]), "quantile_levels": LEVELS}
                mem.append(blend({"pred_q": qd, **ctx}, {"pred_q": qr, **ctx}, gp[(branch, s, k)], tau_f[str(k)]["training_labels_fit_days"]))
            q = np.mean(mem, 0)
            t3 = sync()
            # conformal update: insert one matured score, drop the oldest, query and widen
            bisect.insort(window, float(rng.random()))
            del window[0]
            lo, hi = base_interval(q, LEVELS, "90")
            a, b, _ = widen(lo[0], hi[0], conformal_q(window, 1 - ALPHA["90"]))
            t4 = time.perf_counter()
            if it >= warm:
                times[branch].append((t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0))
    res = {"protocol": "revision/protocols/R-C22_C21_latency_external.md", "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "n_timed": {b: len(v) for b, v in times.items()}, "params_per_member": params,
           "resident_params_all_experts": int(15 * sum(params.values())),
           "active_params_per_forecast": {"G3''": int(15 * (params["A2"] + params["E6-S"])), "G3": int(15 * (params["E4-B-fit"] + params["E6-S"]))},
           "latency_ms": {}}
    for b, v in times.items():
        a = np.array(v) * 1e3
        res["latency_ms"][b] = {step: {"median": float(np.median(a[:, j])), "p95": float(np.percentile(a[:, j], 95))}
                                for j, step in enumerate(("preprocessing", "experts_15_members", "gate_ensemble", "conformal", "total"))}
    p = save("c22_latency.json", res)
    print("saved", p)
    print(json.dumps(res["latency_ms"], indent=1), res["resident_params_all_experts"])


if __name__ == "__main__":
    main()
