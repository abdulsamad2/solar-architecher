"""Guards of the Gatton one-shot evaluation (audit fixes A/B). Development rows only; no sealed row is read."""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not Path(r"D:/solar_data/gatton/bag/data/pv_2020.csv").exists() or
                                not (ROOT / "data/manifests/gatton_forecast_v1.npz").exists(), reason="Gatton data absent")


@pytest.fixture()
def ge(monkeypatch):
    monkeypatch.chdir(ROOT)
    import src.gatton_eval as mod
    return mod


def test_pv_window_reader_loads_only_requested_rows(ge):
    start, end = np.datetime64("2020-04-14"), np.datetime64("2020-04-16")
    pv = ge.read_pv_window(start, end)
    t = np.array(list(pv), dtype=np.int64).astype("datetime64[s]")
    assert len(t) > 0 and t.min() >= start and t.max() < end and end <= ge.TEST_START


def test_verify_all_artifacts_fails_closed_on_missing_and_mismatch(ge, tmp_path, monkeypatch):
    f = tmp_path / "a.bin"
    f.write_bytes(b"x")
    good = {str(f): ge.sha(f)}
    real = Path.read_text

    def fake_read(self, *a, **k):
        if self.name == "phaseE_gatton_inventory.json":
            return json.dumps({**good, **{f"{tmp_path}/pad{i}": "0" for i in range(143)}})
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", fake_read)
    with pytest.raises(AssertionError, match="missing artifact"):
        ge.verify_all_artifacts()


def test_official_run_refuses_without_recorded_token(ge):
    assert not (ROOT / "reports/gatton_test_approval.txt").exists()
    with pytest.raises((AssertionError, FileNotFoundError)):
        ge.run_test("anything")


def test_wrapper_refuses_rerun_when_official_log_exists(tmp_path, monkeypatch):
    import src.gatton_execute as gx
    log = tmp_path / "log.json"
    log.write_text("{}")
    monkeypatch.setattr(gx, "OFFICIAL_LOG", log)
    with pytest.raises(AssertionError, match="no automatic retries"):
        gx.main("official", "TOKEN")
