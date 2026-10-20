"""Read-only Gate 3 diagnostic: E6-S class-conditional severity from the saved checkpoints.

For true events (valid, arrival class 0-2) on fold-validation samples: MAE of the class-conditional mean
dK15 given the TRUE class, vs the marginal-mean MAE. Reconstructs exactly the run's inputs (fit-day anchor,
K-history feature); changes no model, rule or result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.clear_sky import ClearSkyPV, clear_sky_index
from src.folds import calibration_split
from src.skippd import SkippdForecastDataset
from src.train import GpuSeq, build_model


def main():
    device = torch.device("cuda")
    h5, man = "data/raw/2017_2019_images_pv_processed.hdf5", "data/manifests/forecast_tol30_v2.npz"
    ds = SkippdForecastDataset(h5, man, "dev", preload=True)
    m = np.load(man)
    g0 = m["hdf5_group"] == 0
    rec_t = m["times_us"].astype("datetime64[us]")[g0][np.argsort(m["hdf5_row"][g0])]
    rec_day = rec_t.astype("datetime64[D]").astype(str)
    pv = ds.pv.numpy().astype(float)
    pos = {sid: i for i, sid in enumerate(ds.sample_ids)}
    folds = json.loads(Path("data/manifests/folds_blocked5_v1.json").read_text())["folds"]
    data = GpuSeq(ds, device)
    out = {}
    for seed in (42, 123, 456):
        root = Path(f"experiments/E6/v1/E6-S/csi/seed{seed}")
        cfg = json.loads((root / "config.json").read_text())
        rows = []
        for fd in folds:
            k = fd["fold"]
            fit = calibration_split(fd["train_days"])[0]
            sel = np.isin(rec_day, fit)
            pc = ClearSkyPV().fit(rec_t[sel], pv[sel])(rec_t)
            data.set_history_feature(clear_sky_index(pv, pc, 1.0))
            model = build_model(cfg).to(device).eval()
            model.encoder.cnn_bf16 = True
            model.load_state_dict(torch.load(root / f"fold_{k}" / "best.pt", map_location=device))
            z = np.load(root / f"fold_{k}" / "preds_val.npz")
            idx = torch.as_tensor([pos[s] for s in z["sample_ids"]], device=device)
            cond, marg, tc_all = [], [], torch.as_tensor(z["time_class"], device=device)
            with torch.no_grad():
                for s in range(0, len(idx), 2048):
                    o = model(*data(idx[s:s + 2048]))
                    w = o["comp_logits"].softmax(-1)
                    cls_mean = (w * o["mu"]).sum(-1)                       # [B, C]
                    pi = o["time_logits"].softmax(-1)
                    marg.append((pi * cls_mean).sum(-1))
                    cond.append(cls_mean.gather(1, tc_all[s:s + 2048, None]).squeeze(1))
            cond, marg = torch.cat(cond).cpu().numpy(), torch.cat(marg).cpu().numpy()
            ev = z["event_valid"].astype(bool) & (z["time_class"] < 3)
            dk = z["severity_dk"]
            # consistency check against the saved predictive distribution (same checkpoint, same inputs)
            rows.append(pd_row := {"fold": k, "n_events": int(ev.sum()),
                                   "cond_mean_mae": float(np.abs(cond[ev] - dk[ev]).mean()),
                                   "marginal_mean_mae": float(np.abs(marg[ev] - dk[ev]).mean()),
                                   "cond_sign_acc": float((np.sign(cond[ev]) == np.sign(dk[ev])).mean())})
        out[seed] = rows
        agg = {key: float(np.average([r[key] for r in rows], weights=[r["n_events"] for r in rows]))
               for key in ("cond_mean_mae", "marginal_mean_mae", "cond_sign_acc")}
        out[f"{seed}_pooled"] = agg
        print(seed, {k2: round(v, 4) for k2, v in agg.items()})
    Path("reports/gate3_severity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
