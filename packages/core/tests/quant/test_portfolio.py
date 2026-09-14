import numpy as np
import pytest

from quant_os_core.quant.portfolio import (
    component_risk_contribution,
    correlation_matrix,
    covariance_matrix,
    nearest_psd,
    portfolio_variance,
    portfolio_volatility,
    risk_contribution_percentages,
)


def test_covariance_and_correlation_matrices() -> None:
    returns = np.array(
        [
            [0.01, 0.02],
            [0.02, 0.01],
            [-0.01, -0.02],
            [0.03, 0.04],
        ]
    )
    np.testing.assert_allclose(covariance_matrix(returns), np.cov(returns, rowvar=False, ddof=1))
    np.testing.assert_allclose(correlation_matrix(returns), np.corrcoef(returns, rowvar=False))


def test_portfolio_variance_and_volatility() -> None:
    weights = np.array([0.6, 0.4])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    expected = float(weights @ cov @ weights)
    assert portfolio_variance(weights, cov) == pytest.approx(expected)
    assert portfolio_volatility(weights, cov) == pytest.approx(np.sqrt(expected))


def test_risk_contributions_sum_to_portfolio_volatility() -> None:
    weights = [0.5, 0.5]
    cov = [[0.04, 0.01], [0.01, 0.09]]
    components = component_risk_contribution(weights, cov)
    assert components.sum() == pytest.approx(portfolio_volatility(weights, cov))
    percentages = risk_contribution_percentages(weights, cov)
    assert percentages.sum() == pytest.approx(1.0)


def test_non_psd_covariance_rejected() -> None:
    with pytest.raises(ValueError):
        portfolio_variance([0.5, 0.5], [[1.0, 2.0], [2.0, 1.0]])


def test_nearest_psd_repairs_matrix() -> None:
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])
    repaired = nearest_psd(bad)
    assert np.min(np.linalg.eigvalsh(repaired)) >= -1e-12
    np.testing.assert_allclose(repaired, repaired.T)


def test_single_asset_covariance_stays_matrix() -> None:
    result = covariance_matrix([[0.01], [0.02], [-0.01]])
    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(np.var([0.01, 0.02, -0.01], ddof=1))


def test_nearest_psd_rejects_nonfinite_floor() -> None:
    with pytest.raises(ValueError):
        nearest_psd([[1.0, 0.0], [0.0, 1.0]], floor=float("nan"))
