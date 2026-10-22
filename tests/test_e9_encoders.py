"""E9 encoder unit tests (CPU, synthetic)."""
import pytest
import torch

from src.models import DiffSkyEncoder
from src.train import build_model

KINDS = ["cnn_gru", "cnn_gru_wide", "cnn3d_gru", "convgru", "transformer", "diff_cnn_gru"]
BASE = {"image_features": 64, "hidden_size": 64, "pool": "avg", "head": "quantile", "e8_window": "H60"}


@pytest.mark.parametrize("kind", KINDS)
def test_encoder_shapes_gradients_and_extra_features(kind):
    torch.manual_seed(0)
    m = build_model({**BASE, "encoder": kind})
    x, h, e = torch.rand(2, 16, 3, 64, 64, requires_grad=False), torch.rand(2, 16), torch.rand(2, 16, 11)
    out = m(x, h, e)
    assert out.shape == (2, 1, 9) and torch.isfinite(out).all()
    out.sum().backward()
    assert all(p.grad is not None for n, p in m.named_parameters() if "encoder" in n and p.requires_grad)
    m.eval()
    with torch.no_grad():
        a, b = m(x, h, e), m(x, h, e + 1.0)
    assert not torch.allclose(a, b)  # the extra-feature path is wired in


def test_parameter_counts_reported():
    counts = {k: sum(p.numel() for p in build_model({**BASE, "encoder": k}).parameters()) for k in KINDS}
    assert counts["cnn_gru_wide"] > counts["cnn_gru"] and len(set(counts.values())) == len(KINDS)
    print(counts)


def test_diff_encoder_oldest_frame_has_zero_difference():
    enc = DiffSkyEncoder(image_features=8, hidden_size=8)
    captured = {}
    orig = enc.cnn.forward
    enc.cnn.forward = lambda z: captured.setdefault("z", z) is not None and orig(z)
    x = torch.rand(1, 4, 3, 64, 64)
    enc(x, torch.rand(1, 4))
    z = captured["z"].view(1, 4, 6, 64, 64)
    assert torch.equal(z[0, 0, 3:], torch.zeros(3, 64, 64)) and torch.allclose(z[0, 2, 3:], x[0, 2] - x[0, 1])
