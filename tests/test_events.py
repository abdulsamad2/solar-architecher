import numpy as np
import torch
from statistics import NormalDist

from src.events import event_labels, future_rows, implied_event_probability
from src.models import EventMixtureHead

LV = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
MIN = 60_000_000


def test_future_rows_tolerance_and_missing():
    rec = np.array([0, 60, 121, 185, 300], dtype=np.int64) * 1_000_000  # 0s, 60s, 121s, 185s (gap), 300s
    rows = future_rows(rec, np.array([0], dtype=np.int64), horizon=5)
    assert rows.tolist() == [[1, 2, 3, -1, 4]]  # 180s->185s within 30s; 240s missing; 300s exact


def test_event_labels_classes_validity_and_severity():
    k_now = np.array([1.0, 1.0, 1.0, 1.0])
    kf = np.ones((4, 15))
    kf[0, 3:] = 0.5        # crossing at minute 4 -> class 0
    kf[1, 11:] = 0.4       # minute 12 -> class 2
    kf[3, 2] = np.nan      # no crossing, but minute 3 missing -> invalid event label
    lab = event_labels(k_now, kf, kf[:, -1], tau=0.3)
    assert lab["time_class"].tolist() == [0, 2, 3, 3]
    assert lab["event_valid"].tolist() == [True, True, True, False]
    assert np.allclose(lab["severity"], [-0.5, -0.6, 0.0, 0.0])


def test_head_probabilities_quantiles_and_both_losses():
    torch.manual_seed(0)
    head = EventMixtureHead(64, n_bins=3, n_comp=2)
    h = torch.randn(32, 64)
    out = head(h)
    assert torch.allclose(out["time_logits"].softmax(-1).sum(-1), torch.ones(32))
    q = head.marginal_quantiles(out, LV)
    assert q.shape == (32, 9) and (q[:, 1:] >= q[:, :-1]).all()
    dk, cls = torch.randn(32) * 0.2, torch.randint(0, 4, (32,))
    valid = torch.ones(32, dtype=torch.bool)
    for sup in (True, False):
        loss = head.loss(out, dk, cls, valid, supervised=sup)
        assert torch.isfinite(loss)
    p = head.prob_abs_change_at_least(out, 0.3)
    assert ((p >= 0) & (p <= 1)).all()


def test_marginal_quantiles_match_a_single_gaussian():
    head = EventMixtureHead(4, n_bins=3, n_comp=2)
    out = {"time_logits": torch.tensor([[0.0, 0.0, 0.0, 0.0]]), "comp_logits": torch.zeros(1, 4, 2),
           "mu": torch.full((1, 4, 2), 0.1), "scale": torch.full((1, 4, 2), 0.2)}
    q = head.marginal_quantiles(out, LV)[0].numpy()
    assert np.allclose(q, [0.1 + 0.2 * NormalDist().inv_cdf(t) for t in LV], atol=1e-4)


def test_supervised_head_learns_event_timing_on_toy_data():
    torch.manual_seed(0)
    n = 4096
    x = torch.randn(n, 8)
    cls = (x[:, 0] > 0.5).long() * 0 + (x[:, 0] <= 0.5).long() * 3  # feature 0 decides "event in bin 0" vs none
    dk = torch.where(cls == 0, -0.5 + 0.05 * torch.randn(n), 0.02 * torch.randn(n))
    head = EventMixtureHead(8, n_bins=3, n_comp=2)
    opt = torch.optim.Adam(head.parameters(), lr=0.02)
    for _ in range(300):
        loss = head.loss(head(x), dk, cls, torch.ones(n, dtype=torch.bool))
        opt.zero_grad(); loss.backward(); opt.step()
    acc = (head(x)["time_logits"].argmax(-1) == cls).float().mean()
    assert acc > 0.95


def test_event_head_init_matches_training_climatology():
    torch.manual_seed(0)
    n = 20000
    cls = torch.multinomial(torch.tensor([0.06, 0.05, 0.04, 0.85]), n, replacement=True)
    dk = torch.where(cls < 3, -0.4 + 0.1 * torch.randn(n), 0.03 * torch.randn(n))
    head = EventMixtureHead(64, 3, 2)
    head.init_from_targets(dk, cls)
    out = head(torch.randn(256, 64))
    pi = out["time_logits"].softmax(-1).mean(0)
    assert torch.allclose(pi, torch.tensor([0.06, 0.05, 0.04, 0.85]), atol=0.01)
    q = head.marginal_quantiles(out, [0.05, 0.5, 0.95]).mean(0)
    emp = torch.quantile(dk, torch.tensor([0.05, 0.5, 0.95]))
    assert torch.allclose(q, emp, atol=0.05)


def test_implied_event_probability_from_quantiles():
    k_now = np.array([1.0])
    q = 1.0 + 0.3 * np.array([[NormalDist().inv_cdf(t) for t in LV]])
    p = implied_event_probability(q, LV, k_now, tau=0.3)[0]
    assert abs(p - 2 * (1 - NormalDist().cdf(1.0))) < 0.02
