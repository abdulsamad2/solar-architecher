import datetime as dt

import h5py
import numpy as np
import pytest
import torch

from src.baselines import official_persistence
from src.official.relative_op_func import Relative_output
from src.skippd import SkippdForecastDataset, build_forecast_manifest


@pytest.fixture()
def tiny_benchmark(tmp_path):
    """One dev day and one official test day (2018-06-12, sunny), 60 one-minute records each."""
    dev = [dt.datetime(2018, 6, 11, 10, 0) + dt.timedelta(minutes=k) for k in range(60)]
    test = [dt.datetime(2018, 6, 12, 10, 0) + dt.timedelta(minutes=k) for k in range(60)]
    np.save(tmp_path / "times_trainval.npy", np.array(dev, dtype=object), allow_pickle=True)
    np.save(tmp_path / "times_test.npy", np.array(test, dtype=object), allow_pickle=True)
    with h5py.File(tmp_path / "bench.hdf5", "w") as f:
        for name, n, offset in (("trainval", 60, 0), ("test", 60, 1000)):
            img = np.zeros((n, 64, 64, 3), np.uint8)
            img[:, 0, 0, 0] = np.arange(n)  # encode row id in a pixel
            f.create_dataset(f"{name}/images_log", data=img)
            f.create_dataset(f"{name}/pv_log", data=np.arange(n, dtype=float) + offset)
    info = build_forecast_manifest(tmp_path, tmp_path / "m.npz", tolerance_s=30)
    return tmp_path, info


@pytest.mark.parametrize("preload", [True, False])
def test_dataset_reads_correct_frames_and_targets(tiny_benchmark, preload):
    root, info = tiny_benchmark
    assert info["n_dev"] > 0 and info["n_test"] > 0 and info["n_test_sunny"] == info["n_test"]
    for split, offset in (("dev", 0), ("test", 1000)):
        ds = SkippdForecastDataset(root / "bench.hdf5", root / "m.npz", split=split, preload=preload)
        images, pv_hist, target = ds[0]
        assert images.shape == (16, 3, 64, 64) and pv_hist.shape == (16,) and target.shape == (1,)
        row_ids = (images[:, 0, 0, 0] * 255).round().long()
        assert torch.equal(row_ids, (pv_hist - offset).long())          # images and PV aligned
        assert (torch.diff(pv_hist) == 1).all()                          # chronological order
        assert target.item() == pv_hist[-1].item() + 15                  # t + 15 min
        assert (ds.target_time - ds.origin_time == np.timedelta64(15, "m")).all()


def test_dataset_rejects_cross_split_ids(tiny_benchmark):
    root, _ = tiny_benchmark
    test_ids = np.flatnonzero(np.load(root / "m.npz")["is_test"])
    with pytest.raises(ValueError):
        SkippdForecastDataset(root / "bench.hdf5", root / "m.npz", split="dev", sample_ids=test_ids)


def test_official_persistence_matches_reference_formula():
    t = dt.datetime(2018, 6, 12, 12, 0)
    pred = official_persistence([t], [20.0])
    csi, _ = Relative_output(t, 20.0)
    _, p_future = Relative_output(t + dt.timedelta(minutes=15), 5.0)
    assert pred[0] == pytest.approx(csi * p_future)
    assert 0 < pred[0] < 40
