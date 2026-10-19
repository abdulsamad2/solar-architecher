import numpy as np
import pytest
from statistics import NormalDist

from src.train import QUANTILES, probabilistic_scores


def test_calibrated_gaussian_forecast_scores_as_expected():
    rng = np.random.default_rng(0)
    n = 20000
    mu = rng.uniform(5, 25, n)
    y = mu + rng.normal(0, 2.0, n)
    pred_q = mu[:, None] + 2.0 * np.array([NormalDist().inv_cdf(q) for q in QUANTILES])[None, :]
    clear = rng.random(n) < 0.4
    s = probabilistic_scores(pred_q, y, persistence=mu, clear=clear)
    o = s["overall"]
    # calibrated: coverage ~ nominal, reliability ~ levels
    assert o["coverage"]["90"] == pytest.approx(0.90, abs=0.01)
    assert o["coverage"]["80"] == pytest.approx(0.80, abs=0.01)
    assert o["coverage"]["50"] == pytest.approx(0.50, abs=0.015)  # interpolated 0.25/0.75
    assert np.allclose(list(o["reliability"].values()), QUANTILES, atol=0.012)
    # expected CRPS of a calibrated N(mu, s) is s / sqrt(pi) ~ 1.128; a point forecast at mu has CRPS = MAE ~ s*sqrt(2/pi)
    assert o["crps_kw"] == pytest.approx(2.0 / np.sqrt(np.pi), rel=0.03)
    assert o["persistence_crps_kw"] == pytest.approx(2 * np.sqrt(2 / np.pi), rel=0.03)
    assert o["crps_skill_vs_persistence"] > 0.25
    assert set(s) == {"overall", "sunny", "cloudy"} and s["sunny"]["n"] + s["cloudy"]["n"] == n


def test_quantile_head_initialises_at_training_climatology():
    import torch
    from src.models import NonCrossingQuantileHead

    torch.manual_seed(0)
    y = torch.rand(10000) ** 2  # skewed target in [0, 1]
    head = NonCrossingQuantileHead(64, 1, QUANTILES)
    head.init_from_targets(y)
    q = head(torch.randn(256, 64))[:, 0, :].detach()
    emp = torch.quantile(y, torch.tensor(QUANTILES))
    assert torch.allclose(q.mean(0), emp, atol=0.02)
    assert (q[:, 1:] >= q[:, :-1]).all()


def test_overconfident_forecast_undercovers():
    rng = np.random.default_rng(1)
    mu = rng.uniform(5, 25, 5000)
    y = mu + rng.normal(0, 2.0, 5000)
    narrow = mu[:, None] + 0.5 * np.array([NormalDist().inv_cdf(q) for q in QUANTILES])[None, :]
    s = probabilistic_scores(narrow, y, persistence=mu, clear=np.zeros(5000, bool))
    assert s["overall"]["coverage"]["90"] < 0.5 and "sunny" not in s
