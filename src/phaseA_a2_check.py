"""Phase A2 adapter validation on DEVELOPMENT data only (no external labels).

Dev samples are repackaged into the external file's layout (16 chronological frames, 16 PV values, origin time) and
run through ExternalSplit. Frozen members must reproduce their stored fold-validation predictions (amendment-2
criterion), and the SUNSET port must match the Split-based GpuBatches path. Output: reports/phaseA_a2_adapter_check.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.final_eval import H5, MAN, TOL_KW, member_predict
from src.phaseA_a2 import ExternalDS, ExternalSplit, sunset_members
from src.skippd import SkippdForecastDataset


def main():
    folds_doc = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())
    full = SkippdForecastDataset(H5, MAN, "dev", preload=True)
    fold = folds_doc["folds"][0]
    va = np.flatnonzero(np.isin(full.day.astype(str), fold["val_days"]))[:3000]
    imgs = full.images.numpy()[full.hist_rows[va]]
    pv = full.pv.numpy().astype(float)
    m = np.load(MAN)
    sel = m["hdf5_group"] == 0
    rec_t = m["times_us"][sel][np.argsort(m["hdf5_row"][sel])].astype("datetime64[us]")
    times = np.concatenate([rec_t[full.hist_rows[va]], rec_t[full.target_rows[va]][:, None]], 1)  # true record times
    ds = ExternalDS(imgs, pv[full.hist_rows[va]], full.origin_time[va].astype(object), np.zeros(len(va)), rec_times=times)
    split = ExternalSplit(ds, torch.device("cuda"))
    pos = {s: i for i, s in enumerate(full.sample_ids[va])}
    out = {}
    for model, root in (("E4-B-fit", "experiments/E4_calib/v1/csi"), ("E6-S", "experiments/E6/v1/E6-S/csi")):
        st = np.load(f"{root}/seed42/fold_0/preds_val.npz")
        keep = np.array([i in pos for i in st["sample_ids"]])
        idx = np.array([pos[i] for i in st["sample_ids"][keep]])
        got = member_predict(split, model, 42, fold, idx)["pred_q"]
        per = np.abs(got - st["pred_q"][keep]).max(1)
        out[model] = {"n": int(len(idx)), "max": float(per.max()), "mean": float(per.mean()),
                      "share_within_tol": float((per <= TOL_KW).mean())}
        assert out[model]["share_within_tol"] >= 0.999 and per.max() <= 5e-3 and per.mean() <= 1e-6, f"STOP: {model} {out[model]}"
    # SUNSET port: adapter path vs. the original newest-first GpuBatches path on the same dev samples
    from src.sunset import GpuBatches, SunsetNet, predict
    full_nf = SkippdForecastDataset(H5, MAN, "dev", sample_ids=full.sample_ids[va], preload=True, newest_first=True)
    gb = GpuBatches(full_nf, torch.device("cuda"))
    net = SunsetNet().cuda()
    net.load_state_dict(torch.load("experiments/E1_sunset_port/tol30_v2/fold_1/best.pt"))
    ref = predict(net, gb, torch.arange(len(va), device="cuda")).squeeze(-1).cpu().numpy()
    got = sunset_members(ds, torch.device("cuda"))[0]
    out["SUNSET port fold 1"] = {"max": float(np.abs(got - ref).max())}
    assert out["SUNSET port fold 1"]["max"] <= 1e-3, f"STOP: SUNSET adapter {out['SUNSET port fold 1']}"
    Path("reports/phaseA_a2_adapter_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
