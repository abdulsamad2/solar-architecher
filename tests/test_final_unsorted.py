import numpy as np

import src.final_eval as fe


def test_record_table_unsorted_maps_back_to_row_order(tmp_path, monkeypatch):
    # stub feature function: requires sorted input and returns features that identify each record by its time
    def fake(t, gray_path, workers=32):
        t_us = np.asarray(t, dtype="datetime64[us]").astype(np.int64)
        assert (np.diff(t_us) > 0).all()
        g = np.load(gray_path)
        return {"P": t_us[:, None].astype(float) / 1e6, "M": g.reshape(len(g), -1)[:, :1].astype(float)}

    import src.e8_features
    monkeypatch.setattr(src.e8_features, "record_features", fake)
    base = np.datetime64("2019-01-01T12:00:00", "us")
    t = np.array([base + np.timedelta64(m, "m") for m in (10, 11, 12, 0, 1, 2)])  # two out-of-order blocks
    gray = np.arange(6, dtype=np.uint8)[:, None, None] * np.ones((1, 2, 2), np.uint8)
    out = fe.record_table_unsorted(t, gray, tmp_path / "g.npy")
    assert np.allclose(out[:, 0], t.astype(np.int64) / 1e6)  # time feature back in original row order
    assert np.array_equal(out[:, 1], np.arange(6))  # image of each row stays attached to its record
