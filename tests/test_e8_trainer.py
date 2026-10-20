"""E8 trainer-integration unit tests (CPU, synthetic dataset)."""
from types import SimpleNamespace

import numpy as np
import torch

from src.train import E8_MASKS, GpuSeq, build_model

CFG = {"image_features": 64, "hidden_size": 64, "pool": "avg", "head": "quantile", "e8_window": "H60"}


def fake_ds(n_rec=50, n=8):
    rng = np.random.default_rng(0)
    hist = np.stack([np.arange(i, i + 16) for i in range(n)])
    return SimpleNamespace(images=torch.from_numpy(rng.integers(0, 255, (n_rec, 64, 64, 3), dtype=np.uint8)),
                           pv=torch.rand(n_rec), hist_rows=hist, target_rows=hist[:, -1] + 15)


def test_fallback_assembly_uses_arm_where_available_else_base():
    from src.e8_tracka_report import assemble_fallback

    full_ids = np.array(["a", "b", "c", "d"])
    base = np.zeros((4, 9))
    out, used = assemble_fallback(full_ids, base, np.array(["c", "a"]), np.ones((2, 9)) * [[3], [1]])
    assert used.tolist() == [True, False, True, False]
    assert out[0, 0] == 1 and out[2, 0] == 3 and out[1, 0] == 0 and out[3, 0] == 0
    import pytest

    with pytest.raises(AssertionError):
        assemble_fallback(full_ids, base, np.array(["a", "a"]), np.ones((2, 9)))


def test_all_arms_have_identical_parameter_counts():
    counts = {arm: sum(p.numel() for p in build_model({**CFG, "e8_features": arm}).parameters()) for arm in E8_MASKS}
    assert len(set(counts.values())) == 1
    base = sum(p.numel() for p in build_model({**CFG, "e8_window": None}).parameters())
    assert counts["none"] - base == 3 * 64 * 11  # GRU input grows by 11 columns (3 gates x hidden x 11)


def test_masks_zero_fill_exactly():
    table = np.random.default_rng(1).normal(size=(50, 11)).astype(np.float32)
    for arm, mask in E8_MASKS.items():
        masked = table * np.array(mask, np.float32)
        off = np.array(mask) == 0
        assert (masked[:, off] == 0).all() and np.allclose(masked[:, ~off], table[:, ~off])


def test_gpuseq_window_mode_shapes_and_rows():
    ds = fake_ds()
    data = GpuSeq(ds, torch.device("cpu"))
    win = ds.hist_rows[:, ::4][:, :4].repeat(4, 1)  # any valid chronological rows
    win = np.sort(np.stack([np.arange(i, i + 32, 2) for i in range(8)]), 1)
    data.set_window(win)
    data.set_extra(np.arange(50 * 11, dtype=np.float32).reshape(50, 11))
    x, h, e = data(torch.arange(3))
    assert x.shape == (3, 16, 3, 64, 64) and h.shape == (3, 16) and e.shape == (3, 16, 11)
    assert torch.equal(e[0, :, 0], torch.as_tensor(win[0] * 11, dtype=torch.float32))
    model = build_model({**CFG, "e8_features": "PM"})
    out = model(x, h, e)
    assert out.shape == (3, 1, 9)
    data.set_extra(None)
    assert len(data(torch.arange(2))) == 2  # legacy path unchanged
