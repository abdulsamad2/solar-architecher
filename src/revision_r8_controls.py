"""Revision R-R8: capacity-matched (shuffled images) and single-frame controls of the direct expert
(protocol revision/protocols/R-R8_image_controls.md). Runs src.train.main unchanged with a modified image loader.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import torch

import src.train as T

PERM_SEED = 20261116


class ShuffledSeq(T.GpuSeq):
    """Images of another, randomly chosen sample (fixed permutation); own history feature."""

    def __init__(self, ds, device):
        super().__init__(ds, device)
        self.perm = torch.from_numpy(np.random.default_rng(PERM_SEED).permutation(len(ds.hist_rows))).to(device)

    def __call__(self, idx):
        rows = self.hist[idx]
        x = self.images[self.hist[self.perm[idx]]].permute(0, 1, 4, 2, 3).float().div_(255.0)
        return x, self.hist_feat[rows]


class SingleFrameSeq(T.GpuSeq):
    """The latest frame repeated over the 16 positions; own history feature."""

    def __call__(self, idx):
        rows = self.hist[idx]
        x = self.images[rows[:, -1]].permute(0, 3, 1, 2).float().div_(255.0).unsqueeze(1).expand(-1, rows.shape[1], -1, -1, -1)
        return x, self.hist_feat[rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("shuffled", "single"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="D:/skippd_hires/R_R8")
    a = ap.parse_args()
    T.GpuSeq = {"shuffled": ShuffledSeq, "single": SingleFrameSeq}[a.arm]
    # E4-B-fit settings (experiments/E4_calib/v1/csi/seed42/config.json)
    T.main(SimpleNamespace(
        seed=a.seed, lr=1e-4, batch_size=128, max_epochs=15, patience=5, fp32_cnn=False, schedule="cosine", clip=1.0, ema_decay=0.999, pool="avg",
        input="csi", label=f"R-R8 {a.arm}", head="quantile", calib_holdout=True, event_unsupervised=False, residual=False, frozen_thresholds="",
        event_components=2, e8_window=None, e8_features="none", encoder="cnn_gru", loss_weight_alpha=0.0, min_clear_kw=1.0, physics_config="",
        out=f"{a.out}/{a.arm}", target="csi", folds="data/manifests/folds_blocked5_v1.json", manifest="data/manifests/forecast_tol30_v2.npz",
        h5="data/raw/2017_2019_images_pv_processed.hdf5", subset_days=0, fold_ids=a.fold_ids))


if __name__ == "__main__":
    main()
