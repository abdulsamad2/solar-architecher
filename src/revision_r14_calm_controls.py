"""Revision R-R14: image controls of the calm-condition expert (protocol revision/protocols/R-R14_calm_expert_image_controls.md).
Runs src.train.main unchanged with the recipe of E6-S and (a) images shuffled across samples or (b) the PV-only encoder.
Run: python -m src.revision_r14_calm_controls --arm shuffled|pvonly --seed 42
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import src.train as T
from src.revision_r8_controls import ShuffledSeq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("shuffled", "pvonly"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="D:/skippd_hires/R_R14")
    ap.add_argument("--smoke", action="store_true", help="debug: three days per fold, one epoch (never used for results)")
    a = ap.parse_args()
    if a.arm == "shuffled":
        T.GpuSeq = ShuffledSeq
    # E6-S settings (experiments/E6/v1/E6-S/csi/seed42/config.json); only the image input or the encoder differs
    T.main(SimpleNamespace(
        seed=a.seed, lr=1e-4, batch_size=128, max_epochs=1 if a.smoke else 15, patience=5, fp32_cnn=False, schedule="cosine", clip=1.0, ema_decay=0.999,
        pool="avg", input="csi", label=f"R-R14 {a.arm}", head="event", calib_holdout=True, event_unsupervised=False, residual=False,
        frozen_thresholds="" if a.smoke else "reports/phase6_frozen_thresholds.json", event_components=2, e8_window=None, e8_features="none",
        encoder="pv_gru" if a.arm == "pvonly" else "cnn_gru", loss_weight_alpha=0.0, min_clear_kw=1.0, physics_config="",
        out=f"{a.out}/{a.arm}", target="csi", folds="data/manifests/folds_blocked5_v1.json", manifest="data/manifests/forecast_tol30_v2.npz",
        h5="data/raw/2017_2019_images_pv_processed.hdf5", subset_days=3 if a.smoke else 0, fold_ids=a.fold_ids))


if __name__ == "__main__":
    main()
