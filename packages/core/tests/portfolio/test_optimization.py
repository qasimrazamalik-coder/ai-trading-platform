import numpy as np
import pytest

from quant_os_core.portfolio import (
    black_litterman_posterior,
    cvar_minimization_weights,
    hierarchical_equal_risk_contribution_weights,
    hierarchical_risk_parity_weights,
    inverse_volatility_weights,
    maximum_sharpe_weights,
    minimum_variance_weights,
    risk_parity_weights,
    summarize_allocation,
)


def test_inverse_volatility_prefers_lower_risk_asset() -> None:
    cov = np.diag([0.01, 0.04])
    weights = inverse_volatility_weights(cov)
    assert weights == pytest.approx([2 / 3, 1 / 3])


def test_minimum_variance_prefers_lower_variance_asset() -> None:
    cov = np.array([[0.01, 0.0], [0.0, 0.04]])
    weights = minimum_variance_weights(cov)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[1]


def test_maximum_sharpe_prefers_stronger_risk_adjusted_asset() -> None:
    mu = np.array([0.05, 0.15])
    cov = np.array([[0.01, 0.0], [0.0, 0.04]])
    weights = maximum_sharpe_weights(mu, cov)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[1] > 0.0
    summary = summarize_allocation(weights, mu, cov)
    assert summary.sharpe > 0.0


def test_risk_parity_equalizes_diagonal_risk_contributions() -> None:
    cov = np.diag([0.01, 0.04, 0.09])
    weights = risk_parity_weights(cov)
    variance = float(weights @ cov @ weights)
    contributions = weights * (cov @ weights) / variance
    assert contributions == pytest.approx([1 / 3] * 3, abs=2e-5)


def test_cvar_minimization_avoids_crash_prone_asset() -> None:
    scenarios = np.array(
        [
            [0.01, 0.01],
            [0.01, 0.02],
            [0.00, 0.03],
            [-0.01, -0.40],
            [0.01, 0.02],
            [0.00, 0.01],
        ]
    )
    weights = cvar_minimization_weights(scenarios, confidence=0.80)
    assert weights[0] > weights[1]


def test_black_litterman_view_moves_posterior_mean() -> None:
    prior = np.array([0.05, 0.05])
    cov = np.array([[0.04, 0.01], [0.01, 0.04]])
    p = np.array([[1.0, -1.0]])
    q = np.array([0.04])
    posterior, posterior_cov = black_litterman_posterior(prior, cov, p, q)
    assert posterior[0] > posterior[1]
    assert posterior_cov.shape == (2, 2)
    assert np.linalg.eigvalsh(posterior_cov).min() >= -1e-10


def test_hrp_returns_positive_fully_invested_weights() -> None:
    cov = np.array(
        [
            [0.04, 0.03, 0.00, 0.00],
            [0.03, 0.05, 0.00, 0.00],
            [0.00, 0.00, 0.01, 0.005],
            [0.00, 0.00, 0.005, 0.02],
        ]
    )
    weights = hierarchical_risk_parity_weights(cov)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0.0)


def test_optimizer_rejects_infeasible_weight_cap() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        minimum_variance_weights(np.eye(2), max_weight=0.4)


def test_herc_returns_positive_fully_invested_weights() -> None:
    cov = np.array(
        [
            [0.04, 0.03, 0.00, 0.00],
            [0.03, 0.05, 0.00, 0.00],
            [0.00, 0.00, 0.01, 0.005],
            [0.00, 0.00, 0.005, 0.02],
        ]
    )
    weights = hierarchical_equal_risk_contribution_weights(cov)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0.0)
