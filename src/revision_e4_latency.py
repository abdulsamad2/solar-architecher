"""Revision E4b: batch-1 inference time of the four history x encoder arms (protocol revision/protocols/R-E4b_latency.md).

Same procedure as src/revision_c22.py: 15 frozen members per arm resident on the GPU, batch size 1, members run
sequentially. Timed per forecast and arm: image gather/normalise + 15 member forward passes. All four arms run on the
same official-test samples with a valid 60-min history (R-C22 random draw, rng seed 0).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.final_eval import FEAT, H5, MAN, SEEDS, Split, member_anchor, record_table_unsorted
from src.revision_c22 import One, sync
from src.revision_common import save
from src.revision_posthoc_dump import EXPERTS
from src.train import build_model, predict_raw

ARMS = ("D0", "D2", "A2-H15", "A2")
REFERENCE = ("E6-S",)  # the deployed calm-condition expert, timed the same way for reference (not an E4 arm)


def main(n=500, warm=50):
    import h5py
    with h5py.File(H5, "r") as f:
        gray = np.round(f["test/images_log"][...].astype(np.float32).mean(-1)).astype(np.uint8)
    m = np.load(MAN)
    sel = m["hdf5_group"] == 1
    t_test = m["times_us"][sel][np.argsort(m["hdf5_row"][sel])].astype("datetime64[us]")
    dev = torch.device("cuda")
    split = Split("test", dev, record_table_unsorted(t_test, gray, FEAT / "gray_u8_test.npy"))
    folds = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())["folds"]
    anchors = {k: member_anchor(folds[k]) for k in range(5)}
    nets, params, kinds = {}, {}, {}
    for name in ARMS + REFERENCE:
        root, kind = EXPERTS[name]
        kinds[name] = kind
        for s in SEEDS:
            cfg = json.loads((root / f"seed{s}" / "config.json").read_text())
            for k in range(5):
                net = build_model(cfg).to(dev)
                net.encoder.cnn_bf16 = cfg["cnn_bf16"]
                net.load_state_dict(torch.load(root / f"seed{s}" / f"fold_{k}" / "best.pt", map_location=dev))
                nets[(name, s, k)] = net.eval()
        params[name] = int(sum(p.numel() for p in nets[(name, SEEDS[0], 0)].parameters()))
    rng = np.random.default_rng(0)
    eligible = np.flatnonzero(split.h60_ok)
    idx_all = rng.choice(eligible, n + warm, replace=False)
    zeros_extra = torch.zeros(1, 16, 11, device=dev)
    images = split.data.images
    times = {a: [] for a in ARMS + REFERENCE}
    for it, i in enumerate(idx_all):
        rows60, rows15 = split.h60_rows[i], split.ds.hist_rows[i]
        pre = {}
        for k in range(5):  # shared preprocessing (not part of the per-arm time)
            pc60, pc15 = anchors[k](split.rec_t[rows60]), anchors[k](split.rec_t[rows15])
            k60, k15 = split.pv[rows60] / np.maximum(pc60, 1.0), split.pv[rows15] / np.maximum(pc15, 1.0)
            pre[k] = {"h60": torch.tensor(k60[None], dtype=torch.float32, device=dev),
                      "h15": torch.tensor(k15[None], dtype=torch.float32, device=dev),
                      "k_now": torch.tensor([k15[-1]], dtype=torch.float32, device=dev)}
        for name in ARMS + REFERENCE:
            kind = kinds[name]
            rows = rows60 if kind == "H60" else rows15
            t0 = sync()
            x = images[torch.as_tensor(rows, device=dev)][None].permute(0, 1, 4, 2, 3).float().div_(255.0)
            for s in SEEDS:
                for k in range(5):
                    p = pre[k]
                    hist = p["h60"] if kind == "H60" else p["h15"]
                    extra = None if kind == "H15" else zeros_extra
                    # read each member's output back as in src/revision_c22.py (a deployed forecast needs it)
                    predict_raw(nets[(name, s, k)], One(x, hist, extra, p["k_now"]), [0]).cpu().numpy()
            t1 = sync()
            if it >= warm:
                times[name].append(t1 - t0)
    res = {"protocol": "revision/protocols/R-E4b_latency.md", "label": "POST HOC (test inputs; no outcome used)",
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "n_timed": n, "warm_up": warm,
           "timed_step": "image gather/normalise + 15 member forward passes, batch size 1, members sequential",
           "params_per_member": params, "window": kinds, "reference_not_e4_arm": list(REFERENCE), "latency_ms": {}}
    for name, v in times.items():
        a = np.array(v) * 1e3
        res["latency_ms"][name] = {"median": float(np.median(a)), "p95": float(np.percentile(a, 95))}
    p = save("e4_latency.json", res)
    print("saved", p)
    for name in ARMS + REFERENCE:
        r = res["latency_ms"][name]
        print(f"{name:7s} {params[name]:7d} params/member  median {r['median']:.1f} ms  p95 {r['p95']:.1f} ms")


if __name__ == "__main__":
    main()
