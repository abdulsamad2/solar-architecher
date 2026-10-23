import numpy as np
import pytest
import torch

from src.metrics import pinball_loss
from src.physics_losses import (PhysicsConfig, bound_loss, direction_loss, physics_consistency_loss, ramp_loss,
                                reconstruct_power, total_loss)
from src.physics_metrics import physics_diagnostics

Q = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]


def test_power_reconstruction_matches_clear_sky_index_inverse():
    k = torch.tensor([0.5, 1.1, 0.2])
    pc = torch.tensor([20.0, 10.0, 0.4])  # last one below the 1-kW floor
    assert torch.allclose(reconstruct_power(k, pc, 1.0), torch.tensor([10.0, 11.0, 0.2]))
    kq = torch.tensor([[0.4, 0.5, 0.6]])
    assert torch.allclose(reconstruct_power(kq, torch.tensor([20.0]), 1.0), torch.tensor([[8.0, 10.0, 12.0]]))


def test_physics_consistency_is_zero_for_perfect_median_and_normalised():
    cfg = PhysicsConfig(phys_enabled=True, lambda_phys=1.0, p_scale_kw=30.0)
    pc = torch.tensor([20.0, 15.0])
    k50 = torch.tensor([0.5, 0.8])
    p_true = k50 * pc
    assert physics_consistency_loss(k50, p_true, pc, cfg).item() == pytest.approx(0.0)
    assert physics_consistency_loss(k50, p_true + 3.0, pc, cfg).item() == pytest.approx(3.0 / 30.0)
    hub = PhysicsConfig(phys_enabled=True, phys_type="huber", huber_delta=0.05)
    assert physics_consistency_loss(k50, p_true + 0.3, pc, hub).item() == pytest.approx(0.5 * (0.3 / 30) ** 2, rel=1e-5)


def test_bound_loss_soft_and_allows_cloud_enhancement_below_kmax():
    k = torch.tensor([[0.0, 0.5, 1.2, 1.4]])
    assert bound_loss(k, 1.5).item() == pytest.approx(0.0)  # K > 1 is allowed (cloud enhancement)
    k2 = torch.tensor([[-0.2, 0.5, 1.7, 1.0]])
    assert bound_loss(k2, 1.5).item() == pytest.approx((0.2 + 0.2) / 4)


def test_ramp_loss_weights_ramp_samples():
    k_true, k_now = torch.tensor([0.9, 0.5]), torch.tensor([0.4, 0.5])
    k50 = torch.tensor([0.5, 0.6])
    is_ramp = torch.tensor([True, False])
    # errors |dK_true - dK_pred| = 0.4 (ramp) and 0.1; weights 1 + 3*1 = 4 and 1
    assert ramp_loss(k50, k_true, k_now, is_ramp, alpha_ramp=3.0).item() == pytest.approx((4 * 0.4 + 0.1) / 2)
    assert ramp_loss(k50, k_true, k_now, is_ramp, alpha_ramp=0.0).item() == pytest.approx((0.4 + 0.1) / 2)


def test_direction_loss_penalises_wrong_direction_on_ramps_only():
    k_now = torch.tensor([0.5, 0.5, 0.5])
    k_true = torch.tensor([0.9, 0.1, 0.9])
    k50 = torch.tensor([0.4, 0.2, 0.3])        # wrong up, right down, wrong (non-ramp)
    is_ramp = torch.tensor([True, True, False])
    got = direction_loss(k50, k_true, k_now, is_ramp, margin=0.0).item()
    assert got == pytest.approx((0.1 + 0.0) / 2)


def test_disabled_config_equals_existing_pinball_exactly():
    torch.manual_seed(0)
    kq, y = torch.randn(64, 9).sort(-1).values, torch.randn(64)
    tot, parts = total_loss(kq, y, Q, PhysicsConfig())
    assert torch.equal(tot, pinball_loss(kq, y, Q)) and set(parts) == {"quantile"}


def test_total_loss_combines_and_logs_every_component():
    torch.manual_seed(1)
    kq = torch.rand(32, 9).sort(-1).values * 1.2
    y, kn = torch.rand(32), torch.rand(32)
    pc, pt = torch.rand(32) * 25, torch.rand(32) * 25
    ramp = torch.rand(32) > 0.8
    cfg = PhysicsConfig(phys_enabled=True, lambda_phys=0.1, ramp_enabled=True, lambda_ramp=0.05, alpha_ramp=2,
                        bound_enabled=True, lambda_bound=0.01, direction_enabled=True, lambda_direction=0.02)
    tot, parts = total_loss(kq, y, Q, cfg, k_now=kn, p_true_kw=pt, p_clear_kw=pc, is_ramp=ramp)
    want = parts["quantile"] + 0.1 * parts["phys"] + 0.05 * parts["ramp"] + 0.01 * parts["bound"] + 0.02 * parts["direction"]
    assert set(parts) == {"quantile", "phys", "ramp", "bound", "direction"}
    assert tot.item() == pytest.approx(want, rel=1e-6)


def test_config_from_yaml_dict():
    cfg = PhysicsConfig.from_dict({"physics_loss": {"enabled": True, "type": "huber", "lambda_phys": 0.1},
                                   "ramp_loss": {"enabled": True, "lambda_ramp": 0.05, "alpha_ramp": 2}})
    assert cfg.phys_enabled and cfg.phys_type == "huber" and cfg.ramp_enabled and not cfg.bound_enabled
    assert not PhysicsConfig.from_dict(None).any_enabled


def test_no_test_leakage_in_training_tensors():
    """Physics terms use only per-sample tensors indexed by the training batch; the normaliser is a fixed constant."""
    import inspect

    import src.physics_losses as pl
    import src.train as tr
    src = inspect.getsource(tr.run_fold)
    assert 'phys_t["k_true"][b]' in src and 'phys_t["p_true"][b]' in src  # batch-indexed (b draws from training idx)
    assert "fit_all = np.flatnonzero(np.isin(day_str, list(train_days)))" in src  # threshold from fit-day samples only
    assert "tau_p = float(np.quantile(np.abs(y - k_now_p)[fit_all], 0.9))" in src
    assert PhysicsConfig().p_scale_kw == 30.0                                   # fixed constant, not a data statistic
    assert "test" not in inspect.getsource(pl).lower().replace("a test statistic", "")


def test_physics_diagnostics():
    kq = np.tile(np.linspace(-0.1, 1.6, 9), (4, 1))
    d = physics_diagnostics(kq, k_true=np.array([0.9, 0.2, 0.5, 0.5]), k_now=np.array([0.4, 0.6, 0.5, 0.5]),
                            p_true_kw=np.full(4, 10.0), p_clear_kw=np.full(4, 20.0),
                            is_ramp=np.array([True, True, False, False]), k_max=1.5)
    assert d["frac_k_neg"] == pytest.approx(1 / 9) and d["frac_k_over_kmax"] == pytest.approx(1 / 9)
    assert 0 <= d["ramp_direction_accuracy"] <= 1 and d["n_ramp"] == 2
