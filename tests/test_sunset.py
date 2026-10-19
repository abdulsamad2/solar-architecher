import numpy as np
import torch

from src.sunset import SunsetNet, cv_split, day_block_shuffle


def test_sunset_shapes_and_param_count():
    model = SunsetNet()
    y = model(torch.rand(4, 48, 64, 64), torch.rand(4, 16))
    assert y.shape == (4,)
    # Keras summary: conv1 10,392 + conv2 10,416 + bn 288 + fc1 12,600,320 + fc2 1,049,600 + out 1,025
    assert sum(p.numel() for p in model.parameters()) + 2 * (24 + 48) == 13_672_041


def test_day_block_shuffle_and_cv_split_partition():
    dates = np.repeat(np.arange(30), 7).astype("datetime64[D]")
    shuffled = day_block_shuffle(dates)
    assert sorted(shuffled) == list(range(len(dates)))
    # whole days stay contiguous after shuffling
    runs = dates[shuffled]
    assert (np.unique(runs, return_counts=True)[1] == 7).all()
    assert len(np.flatnonzero(runs[1:] != runs[:-1])) == 29
    vals = []
    for k in range(10):
        tr, va = cv_split(shuffled, k, 10)
        assert len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == len(dates)
        vals.append(va)
    assert sorted(np.concatenate(vals)) == list(range(len(dates)))
