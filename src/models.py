"""CNN-GRU encoder and output heads (point, non-crossing quantile, event-time/severity)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallSkyEncoder(nn.Module):
    """Shared per-frame CNN -> concat PV history -> GRU -> final hidden state."""

    def __init__(self, in_channels: int = 3, image_features: int = 64, hidden_size: int = 64, pool: str = "avg",
                 extra_features: int = 0, channels=(16, 32, 64, 64)):
        """pool='avg': global average (discards where clouds are relative to the sun);
        pool='spatial': keep a 4x4 grid and flatten (position-aware, like SUNSET's flatten).
        extra_features: per-frame scalars appended to the GRU input (E8 solar/motion; zero-filled when unused)."""
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2)
            )

        grid = {"avg": 1, "spatial": 4}[pool]
        c = (in_channels,) + tuple(channels)
        self.cnn = nn.Sequential(
            *[block(c[i], c[i + 1]) for i in range(4)],
            nn.AdaptiveAvgPool2d(grid), nn.Flatten(), nn.Linear(c[-1] * grid * grid, image_features), nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(image_features + 1 + extra_features, hidden_size, batch_first=True)
        self.hidden_size = hidden_size
        self.extra_features = extra_features
        self.cnn_bf16 = False  # bf16 autocast for the CNN only; GRU and heads stay fp32

    def forward(self, images: torch.Tensor, pv_history: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        b, t = images.shape[:2]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cnn_bf16 and images.is_cuda):
            feats = self.cnn(images.flatten(0, 1))
        feats = feats.float().view(b, t, -1)
        parts = [feats, pv_history.unsqueeze(-1)]
        if self.extra_features:
            parts.append(extra)
        seq = torch.cat(parts, dim=-1)
        self.gru.flatten_parameters()  # needed after deepcopy (e.g. EMA copies)
        _, h = self.gru(seq)
        return h[-1]


def _cnn_block2d(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2))


class DiffSkyEncoder(SmallSkyEncoder):
    """E9 A5: CNN-GRU whose per-frame input is [frame, frame - previous window frame] (6 channels)."""

    def __init__(self, **kw):
        super().__init__(in_channels=6, **kw)

    def forward(self, images, pv_history, extra=None):
        diff = torch.cat([torch.zeros_like(images[:, :1]), images[:, 1:] - images[:, :-1]], dim=1)
        return super().forward(torch.cat([images, diff], dim=2), pv_history, extra)


class Conv3dGRUEncoder(nn.Module):
    """E9 A2: Conv3d blocks (temporal kernel 3, spatial stride 2, no temporal stride) -> per-time features -> GRU."""

    def __init__(self, image_features: int = 64, hidden_size: int = 64, extra_features: int = 0):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
                                 nn.MaxPool3d((1, 2, 2)))

        self.conv = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 64))
        self.proj = nn.Sequential(nn.Linear(64, image_features), nn.ReLU(inplace=True))
        self.gru = nn.GRU(image_features + 1 + extra_features, hidden_size, batch_first=True)
        self.hidden_size, self.extra_features, self.cnn_bf16 = hidden_size, extra_features, False

    def forward(self, images, pv_history, extra=None):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cnn_bf16 and images.is_cuda):
            f = self.conv(images.permute(0, 2, 1, 3, 4))          # [B, C, T, h, w]
        f = self.proj(f.float().mean(dim=(3, 4)).transpose(1, 2))  # [B, T, F]
        parts = [f, pv_history.unsqueeze(-1)] + ([extra] if self.extra_features else [])
        self.gru.flatten_parameters()
        _, h = self.gru(torch.cat(parts, -1))
        return h[-1]


class ConvGRUCell(nn.Module):
    def __init__(self, cin: int, ch: int):
        super().__init__()
        self.gates = nn.Conv2d(cin + ch, 2 * ch, 3, padding=1)
        self.cand = nn.Conv2d(cin + ch, ch, 3, padding=1)

    def forward(self, x, h):
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)
        n = torch.tanh(self.cand(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


class ConvGRUEncoder(nn.Module):
    """E9 A3: per-frame CNN to an 8x8x32 map; ConvGRU over time keeps spatial layout; K (+extra) as constant channels."""

    def __init__(self, hidden_size: int = 64, extra_features: int = 0, ch: int = 32):
        super().__init__()
        self.cnn = nn.Sequential(_cnn_block2d(3, 16), _cnn_block2d(16, 32), _cnn_block2d(32, ch))  # 64 -> 8
        self.cell = ConvGRUCell(ch + 1 + extra_features, ch)
        self.out = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(ch * 4, hidden_size), nn.ReLU(inplace=True))
        self.hidden_size, self.extra_features, self.ch, self.cnn_bf16 = hidden_size, extra_features, ch, False

    def forward(self, images, pv_history, extra=None):
        b, t = images.shape[:2]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cnn_bf16 and images.is_cuda):
            maps = self.cnn(images.flatten(0, 1))
        maps = maps.float().view(b, t, self.ch, *maps.shape[-2:])
        h = torch.zeros(b, self.ch, *maps.shape[-2:], device=images.device)
        for k in range(t):
            scal = pv_history[:, k:k + 1] if not self.extra_features else torch.cat([pv_history[:, k:k + 1], extra[:, k]], 1)
            const = scal[:, :, None, None].expand(-1, -1, *maps.shape[-2:])
            h = self.cell(torch.cat([maps[:, k], const], 1), h)
        return self.out(h)


class TemporalTransformerEncoder(nn.Module):
    """E9 A4: per-frame CNN tokens (+K, +extra) with learned positions -> 2-layer transformer -> last token."""

    def __init__(self, image_features: int = 64, hidden_size: int = 64, extra_features: int = 0, n_frames: int = 16):
        super().__init__()
        self.cnn = nn.Sequential(_cnn_block2d(3, 16), _cnn_block2d(16, 32), _cnn_block2d(32, 64), _cnn_block2d(64, 64),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, image_features), nn.ReLU(inplace=True))
        self.tok = nn.Linear(image_features + 1 + extra_features, hidden_size)
        self.pos = nn.Parameter(torch.zeros(1, n_frames, hidden_size))
        layer = nn.TransformerEncoderLayer(hidden_size, 4, 128, dropout=0.1, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.hidden_size, self.extra_features, self.cnn_bf16 = hidden_size, extra_features, False

    def forward(self, images, pv_history, extra=None):
        b, t = images.shape[:2]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cnn_bf16 and images.is_cuda):
            f = self.cnn(images.flatten(0, 1))
        f = f.float().view(b, t, -1)
        parts = [f, pv_history.unsqueeze(-1)] + ([extra] if self.extra_features else [])
        z = self.tr(self.tok(torch.cat(parts, -1)) + self.pos[:, :t])
        return z[:, -1]


class PointHead(nn.Module):
    def __init__(self, hidden_size: int, n_horizons: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, n_horizons)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # [B, H]
        return self.fc(h)


class NonCrossingQuantileHead(nn.Module):
    """Lowest quantile + cumulative softplus increments => monotone in q by construction."""

    def __init__(self, hidden_size: int, n_horizons: int, quantiles: Sequence[float]):
        super().__init__()
        q = torch.as_tensor(quantiles, dtype=torch.float32)
        if not torch.all(q[1:] > q[:-1]):
            raise ValueError("quantiles must be strictly increasing")
        self.register_buffer("quantiles", q)
        self.n_horizons, self.n_q = n_horizons, len(quantiles)
        self.fc = nn.Linear(hidden_size, n_horizons * self.n_q)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # [B, H, Q]
        raw = self.fc(h).view(-1, self.n_horizons, self.n_q)
        base, inc = raw[..., :1], F.softplus(raw[..., 1:])
        return torch.cat([base, base + torch.cumsum(inc, dim=-1)], dim=-1)

    @torch.no_grad()
    def init_from_targets(self, y_train: torch.Tensor, weight_scale: float = 0.01, min_gap: float = 1e-4) -> None:
        """Start at the unconditional climatology of the TRAINING targets.

        Without this, softplus(0) ~ 0.69 per increment gives an initial median of ~2.8 target units
        (84 kW for kW/30 targets) and a 90% interval ~5.5 units wide.
        """
        emp = torch.quantile(y_train.float(), self.quantiles.to(y_train.device)).to(self.fc.bias)
        gaps = torch.clamp(emp[1:] - emp[:-1], min=min_gap)
        inv_softplus = gaps + torch.log(-torch.expm1(-gaps))
        bias = self.fc.bias.view(self.n_horizons, self.n_q)
        bias[:, 0] = emp[0]
        bias[:, 1:] = inv_softplus
        self.fc.weight.mul_(weight_scale)


class EventSeverityHead(nn.Module):
    """Scaffold: p(arrival bin | x) over n_bins + 'no event', and a Normal severity per bin.

    Severity is the normalized change Delta K, conditional on the arrival bin.
    """

    def __init__(self, hidden_size: int, n_time_bins: int = 3, min_scale: float = 1e-3):
        super().__init__()
        self.n_time_bins = n_time_bins
        self.min_scale = min_scale
        self.time_fc = nn.Linear(hidden_size, n_time_bins + 1)  # last class == no event
        self.sev_fc = nn.Linear(hidden_size, 2 * n_time_bins)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.time_fc(h)
        mu, raw_scale = self.sev_fc(h).chunk(2, dim=-1)
        return {
            "time_logits": logits,
            "time_probs": logits.softmax(-1),
            "sev_mean": mu,
            "sev_scale": F.softplus(raw_scale) + self.min_scale,
        }

    def nll(self, out: dict[str, torch.Tensor], time_class: torch.Tensor, severity: torch.Tensor) -> torch.Tensor:
        """Cross-entropy on arrival + Gaussian NLL on severity for event samples only."""
        loss = F.cross_entropy(out["time_logits"], time_class)
        is_event = time_class < self.n_time_bins
        if is_event.any():
            idx = time_class[is_event].unsqueeze(-1)
            mu = out["sev_mean"][is_event].gather(-1, idx).squeeze(-1)
            sd = out["sev_scale"][is_event].gather(-1, idx).squeeze(-1)
            loss = loss - torch.distributions.Normal(mu, sd).log_prob(severity[is_event]).mean()
        return loss


class EventMixtureHead(nn.Module):
    """Phase 6 (SHADOW-MoE core): arrival-time classes x conditional severity mixtures -> marginal dK15.

    Classes c = 0..n_bins-1 (event arrives in bin c) and c = n_bins (no event). For each class, a mixture of
    ``n_comp`` Gaussians over dK15 = K(t+15) - K(t). The predictive distribution is the class-weighted
    mixture: p(dK) = sum_c pi_c sum_m w_cm N(dK; mu_cm, s_cm).

    Loss (``supervised=True``): CE(time class) + NLL(dK | true class) on event-valid samples, plus
    ``lambda_marginal`` * marginal NLL on all samples. ``supervised=False`` uses only the marginal NLL:
    the same architecture and parameter count as an unsupervised mixture-density head, isolating the value
    of explicit event-time supervision.
    """

    def __init__(self, hidden_size: int, n_bins: int = 3, n_comp: int = 2, min_scale: float = 1e-3):
        super().__init__()
        self.n_cls, self.n_comp, self.min_scale = n_bins + 1, n_comp, min_scale
        self.time_fc = nn.Linear(hidden_size, self.n_cls)
        self.mix_fc = nn.Linear(hidden_size, self.n_cls * n_comp * 3)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        b = h.shape[0]
        w, mu, s = self.mix_fc(h).view(b, self.n_cls, self.n_comp, 3).unbind(-1)
        return {"time_logits": self.time_fc(h), "comp_logits": w, "mu": mu, "scale": F.softplus(s) + self.min_scale}

    @torch.no_grad()
    def init_from_targets(self, dk: torch.Tensor, time_class: torch.Tensor, weight_scale: float = 0.01) -> None:
        """Start at the TRAINING climatology: class bias = log class frequency; per class, components at the
        class's dK quantiles (spread over 25..75%) with the class's dK spread; small weights."""
        freq = torch.bincount(time_class, minlength=self.n_cls).float().clamp(min=1)
        self.time_fc.bias.copy_((freq / freq.sum()).log())
        bias = self.mix_fc.bias.view(self.n_cls, self.n_comp, 3)
        qs = torch.linspace(0.25, 0.75, self.n_comp, device=dk.device)
        for c in range(self.n_cls):
            x = dk[time_class == c].float()
            if len(x) < 10:
                x = dk.float()
            sd = x.std().clamp(min=1e-3)
            bias[c, :, 0] = 0.0
            bias[c, :, 1] = torch.quantile(x, qs)
            bias[c, :, 2] = sd + torch.log(-torch.expm1(-sd))  # inverse softplus
        self.time_fc.weight.mul_(weight_scale)
        self.mix_fc.weight.mul_(weight_scale)

    @staticmethod
    def _log_mix(out, x, cls_log_w):
        """log sum_{c,m} exp(cls_log_w[c] + log w_cm + log N(x; mu_cm, s_cm)); x [B]."""
        lw = cls_log_w[..., None] + out["comp_logits"].log_softmax(-1)
        lp = torch.distributions.Normal(out["mu"], out["scale"]).log_prob(x[:, None, None])
        return torch.logsumexp((lw + lp).flatten(1), dim=1)

    def marginal_log_prob(self, out, x):
        return self._log_mix(out, x, out["time_logits"].log_softmax(-1))

    def loss(self, out, dk, time_class, event_valid, supervised: bool = True, lambda_marginal: float = 1.0):
        marg = -self.marginal_log_prob(out, dk).mean()
        if not supervised:
            return marg
        v = event_valid
        ce = F.cross_entropy(out["time_logits"][v], time_class[v]) if v.any() else dk.new_zeros(())
        onehot = F.one_hot(time_class[v], self.n_cls).to(dk.dtype)
        cond_log_w = torch.where(onehot > 0, torch.zeros_like(onehot), torch.full_like(onehot, -1e9))
        sub = {k: t[v] for k, t in out.items()}
        cond = -self._log_mix(sub, dk[v], cond_log_w).mean() if v.any() else dk.new_zeros(())
        return ce + cond + lambda_marginal * marg

    @torch.no_grad()
    def marginal_quantiles(self, out, levels, iters: int = 60) -> torch.Tensor:
        """Quantiles of the marginal dK15 mixture by bisection on its CDF; [B, len(levels)], non-decreasing."""
        pi = out["time_logits"].softmax(-1)[..., None] * out["comp_logits"].softmax(-1)  # [B, C, M]
        dist = torch.distributions.Normal(out["mu"][:, None], out["scale"][:, None])  # [B, 1, C, M]
        lo = (out["mu"] - 8 * out["scale"]).amin(dim=(1, 2))
        hi = (out["mu"] + 8 * out["scale"]).amax(dim=(1, 2))
        q = torch.as_tensor(levels, dtype=lo.dtype, device=lo.device)
        a, b = lo[:, None].expand(-1, len(q)).clone(), hi[:, None].expand(-1, len(q)).clone()
        for _ in range(iters):
            mid = (a + b) / 2
            cdf = (pi[:, None] * dist.cdf(mid[..., None, None])).sum(dim=(2, 3))  # [B, Q]
            go_right = cdf < q
            a, b = torch.where(go_right, mid, a), torch.where(go_right, b, mid)
        return torch.cummax((a + b) / 2, dim=1).values

    @torch.no_grad()
    def prob_abs_change_at_least(self, out, tau: float) -> torch.Tensor:
        """P(|dK15| >= tau) from the marginal mixture (comparable with the direct-quantile comparator)."""
        pi = out["time_logits"].softmax(-1)[..., None] * out["comp_logits"].softmax(-1)
        dist = torch.distributions.Normal(out["mu"], out["scale"])
        inside = dist.cdf(torch.full_like(out["mu"], tau)) - dist.cdf(torch.full_like(out["mu"], -tau))
        return (1 - (pi * inside).sum(dim=(1, 2))).clamp(0, 1)


class Forecaster(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, images: torch.Tensor, pv_history: torch.Tensor, extra: torch.Tensor | None = None):
        return self.head(self.encoder(images, pv_history, extra))
