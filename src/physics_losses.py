"""Physics-informed auxiliary losses for the SHADOW-MoE quantile experts (training objective only).

All terms act on the expert's predicted clear-sky-index quantiles K_hat[:, q] (shape [B, Q]); nothing here changes
the architecture, the quantile outputs, the gate, the ensemble or the calibration. With every term disabled the
total loss is exactly the existing pinball loss.

Notation (per sample i):
  K_true        target clear-sky index K(t+15)
  K_now         K(t) at the origin record (an input-time quantity)
  P_true        target PV in kW;  P_clear = P_clear(t+15) in kW (fold anchor, fitted on training days only)
  K50           median quantile K_hat[:, 0.5]
  r_i           ramp indicator, |K_true - K_now| >= tau_fold (existing per-fold event threshold, fit days)

  L_quantile  = mean_{i,q} rho_q(K_true - K_hat_q),  rho_q(e) = max(q e, (q-1) e)
  L_phys      = mean_i |P_true - K50 * max(P_clear, c_min)| / P_scale          (type "mae")
                or the Huber loss with threshold delta on the same normalised error (type "huber")
  L_bound     = mean_{i,q} [ ReLU(-K_hat_q) + ReLU(K_hat_q - K_max) ]
  L_ramp      = mean_i w_i |(K_true - K_now) - (K50 - K_now)|,  w_i = 1 + alpha_ramp * r_i
  L_direction = sum_i r_i ReLU(m - sign(K_true - K_now) * (K50 - K_now)) / max(1, sum_i r_i)

  L_total = L_quantile + lambda_phys L_phys + lambda_ramp L_ramp + lambda_bound L_bound + lambda_direction L_direction

Note: with K50 = K_hat median, (K_true - K_now) - (K50 - K_now) = K_true - K50, i.e. L_ramp is a ramp-weighted
absolute error of the median in K units. The ramp weighting is what distinguishes it from the pinball term.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from src.metrics import pinball_loss


@dataclass
class PhysicsConfig:
    phys_enabled: bool = False
    phys_type: str = "mae"            # "mae" | "huber"
    lambda_phys: float = 0.0
    huber_delta: float = 0.05         # in normalised power units (fraction of P_scale)
    ramp_enabled: bool = False
    lambda_ramp: float = 0.0
    alpha_ramp: float = 0.0
    bound_enabled: bool = False
    lambda_bound: float = 0.0
    k_max: float = 1.5
    direction_enabled: bool = False
    lambda_direction: float = 0.0
    direction_margin: float = 0.02
    p_scale_kw: float = 30.0          # fixed nameplate-scale normaliser (SKIPP'D PV_SCALE); never a test statistic
    c_min_kw: float = 1.0             # same floor as the clear-sky index definition

    @classmethod
    def from_dict(cls, d: dict | None) -> "PhysicsConfig":
        if not d:
            return cls()
        pl, rl, bl, dl = (d.get(k) or {} for k in ("physics_loss", "ramp_loss", "bound_loss", "direction_loss"))
        return cls(phys_enabled=bool(pl.get("enabled", False)), phys_type=pl.get("type", "mae"),
                   lambda_phys=float(pl.get("lambda_phys", 0.0)), huber_delta=float(pl.get("huber_delta", 0.05)),
                   ramp_enabled=bool(rl.get("enabled", False)), lambda_ramp=float(rl.get("lambda_ramp", 0.0)),
                   alpha_ramp=float(rl.get("alpha_ramp", 0.0)),
                   bound_enabled=bool(bl.get("enabled", False)), lambda_bound=float(bl.get("lambda_bound", 0.0)),
                   k_max=float(bl.get("k_max", 1.5)),
                   direction_enabled=bool(dl.get("enabled", False)), lambda_direction=float(dl.get("lambda_direction", 0.0)),
                   direction_margin=float(dl.get("margin", 0.02)),
                   p_scale_kw=float(d.get("p_scale_kw", 30.0)), c_min_kw=float(d.get("c_min_kw", 1.0)))

    @property
    def any_enabled(self) -> bool:
        return self.phys_enabled or self.ramp_enabled or self.bound_enabled or self.direction_enabled


def reconstruct_power(k: torch.Tensor, p_clear_kw: torch.Tensor, c_min_kw: float) -> torch.Tensor:
    """P_hat = K_hat * max(P_clear, c_min) (the repository's clear-sky-index inverse); broadcasts over quantiles."""
    pc = torch.clamp(p_clear_kw, min=c_min_kw)
    return k * (pc.unsqueeze(-1) if k.ndim == pc.ndim + 1 else pc)


def physics_consistency_loss(k50, p_true_kw, p_clear_kw, cfg: PhysicsConfig) -> torch.Tensor:
    err = (p_true_kw - reconstruct_power(k50, p_clear_kw, cfg.c_min_kw)) / cfg.p_scale_kw
    if cfg.phys_type == "huber":
        return F.huber_loss(err, torch.zeros_like(err), delta=cfg.huber_delta)
    if cfg.phys_type != "mae":
        raise ValueError(cfg.phys_type)
    return err.abs().mean()


def bound_loss(k_q: torch.Tensor, k_max: float) -> torch.Tensor:
    return (F.relu(-k_q) + F.relu(k_q - k_max)).mean()


def ramp_loss(k50, k_true, k_now, is_ramp, alpha_ramp: float) -> torch.Tensor:
    w = 1.0 + alpha_ramp * is_ramp.to(k50.dtype)
    return (w * ((k_true - k_now) - (k50 - k_now)).abs()).mean()


def direction_loss(k50, k_true, k_now, is_ramp, margin: float) -> torch.Tensor:
    r = is_ramp.to(k50.dtype)
    s = torch.sign(k_true - k_now)
    return (r * F.relu(margin - s * (k50 - k_now))).sum() / torch.clamp(r.sum(), min=1.0)


def total_loss(k_q: torch.Tensor, k_true, quantiles, cfg: PhysicsConfig, *, k_now=None, p_true_kw=None,
               p_clear_kw=None, is_ramp=None) -> tuple[torch.Tensor, dict]:
    """k_q: [B, Q] predicted K quantiles. Returns (L_total, {component: float}) with every component logged."""
    q = list(quantiles)
    k50 = k_q[:, q.index(0.5)]
    parts = {"quantile": pinball_loss(k_q, k_true, q)}
    if cfg.phys_enabled:
        parts["phys"] = physics_consistency_loss(k50, p_true_kw, p_clear_kw, cfg)
    if cfg.ramp_enabled:
        parts["ramp"] = ramp_loss(k50, k_true, k_now, is_ramp, cfg.alpha_ramp)
    if cfg.bound_enabled:
        parts["bound"] = bound_loss(k_q, cfg.k_max)
    if cfg.direction_enabled:
        parts["direction"] = direction_loss(k50, k_true, k_now, is_ramp, cfg.direction_margin)
    lam = {"quantile": 1.0, "phys": cfg.lambda_phys, "ramp": cfg.lambda_ramp, "bound": cfg.lambda_bound,
           "direction": cfg.lambda_direction}
    total = sum(lam[k] * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}
