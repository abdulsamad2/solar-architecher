"""CNN-GRU encoder and output heads (point, non-crossing quantile, event-time/severity)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallSkyEncoder(nn.Module):
    """Shared per-frame CNN -> concat PV history -> GRU -> final hidden state."""

    def __init__(self, in_channels: int = 3, image_features: int = 64, hidden_size: int = 64):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2)
            )

        self.cnn = nn.Sequential(
            block(in_channels, 16), block(16, 32), block(32, 64), block(64, 64),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, image_features), nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(image_features + 1, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, images: torch.Tensor, pv_history: torch.Tensor) -> torch.Tensor:
        b, t = images.shape[:2]
        feats = self.cnn(images.flatten(0, 1)).view(b, t, -1)
        seq = torch.cat([feats, pv_history.unsqueeze(-1)], dim=-1)
        _, h = self.gru(seq)
        return h[-1]


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


class Forecaster(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, images: torch.Tensor, pv_history: torch.Tensor):
        return self.head(self.encoder(images, pv_history))
