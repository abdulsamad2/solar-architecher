"""Inference cost of the E4 forecasters on the local GPU (run with the GPU otherwise idle).

Batch-1 latency and batched throughput (CUDA-synchronised, after warm-up), parameter count, and the
time to apply the calibration steps per sample.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from src.calibrate import base_interval, cqr_apply, recal_apply, recal_fit
from src.train import QUANTILES, build_model


def time_model(cfg, batch, steps, device):
    torch.manual_seed(0)
    model = build_model(cfg).to(device).eval()
    model.encoder.cnn_bf16 = True
    x = torch.rand(batch, 16, 3, 64, 64, device=device)
    h = torch.rand(batch, 16, device=device)
    with torch.no_grad():
        for _ in range(20):
            model(x, h)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            model(x, h)
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    return dt, sum(p.numel() for p in model.parameters())


def main():
    device = torch.device("cuda")
    cfg = {"image_features": 64, "hidden_size": 64, "pool": "avg", "head": "quantile"}
    lat1, n_params = time_model(cfg, 1, 300, device)
    lat_b, _ = time_model(cfg, 1024, 30, device)
    rng = np.random.default_rng(0)
    q = np.sort(rng.normal(15, 3, (50000, 9)), axis=1)
    y = q[:, 4] + rng.normal(0, 3, 50000)
    t0 = time.perf_counter()
    cal = recal_fit(q, y, QUANTILES)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    recal_apply(q, QUANTILES, cal)
    for n in ("50", "80", "90"):
        cqr_apply(*base_interval(q, QUANTILES, n), 0.5)
    apply_s = time.perf_counter() - t0
    out = {"gpu": torch.cuda.get_device_name(0), "parameters": int(n_params),
           "batch1_latency_ms": lat1 * 1e3, "batched_throughput_samples_per_s": 1024 / lat_b,
           "calibration_fit_s_per_50k_samples": fit_s, "calibration_apply_us_per_sample": apply_s / 50000 * 1e6}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    json.dump(main(), open("reports/gate2_inference_cost.json", "w"), indent=1)
