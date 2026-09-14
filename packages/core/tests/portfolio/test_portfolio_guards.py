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


def test_portfolio_optimizers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        inverse_volatility_weights(np.array([[1.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(ValueError):
        maximum_sharpe_weights([0.1, np.nan], np.eye(2))
    with pytest.raises(ValueError):
        risk_parity_weights(np.eye(2), target_budgets=[1.0])
    with pytest.raises(ValueError):
        cvar_minimization_weights([[0.1, 0.2]], confidence=0.95)
    with pytest.raises(ValueError):
        hierarchical_risk_parity_weights(np.eye(1))
    with pytest.raises(ValueError):
        hierarchical_equal_risk_contribution_weights(np.eye(1))


def test_black_litterman_rejects_bad_view_shapes() -> None:
    with pytest.raises(ValueError, match="pick_matrix"):
        black_litterman_posterior([0.1, 0.2], np.eye(2), [[1.0]], [0.1])
    with pytest.raises(ValueError, match="view_uncertainty"):
        black_litterman_posterior(
            [0.1, 0.2],
            np.eye(2),
            [[1.0, -1.0]],
            [0.1],
            view_uncertainty=np.eye(2),
        )


def test_allocation_summary_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="equal length"):
        summarize_allocation([1.0], [0.1, 0.2], np.eye(1))


def test_low_level_portfolio_validation_guards() -> None:
    with pytest.raises(ValueError):
        summarize_allocation([[1.0]], [0.1], [[0.1]])
    with pytest.raises(ValueError):
        summarize_allocation([-1.0, 2.0], [0.1, 0.2], np.eye(2))
    with pytest.raises(ValueError):
        summarize_allocation([0.0, 0.0], [0.1, 0.2], np.eye(2))
    with pytest.raises(ValueError):
        maximum_sharpe_weights([0.1, 0.2], [[1.0]])
    with pytest.raises(ValueError):
        maximum_sharpe_weights([0.1, 0.2], [[1.0, np.nan], [np.nan, 1.0]])
    with pytest.raises(ValueError):
        maximum_sharpe_weights([0.1, 0.2], [[1.0, 0.5], [0.1, 1.0]])
    with pytest.raises(ValueError):
        inverse_volatility_weights([1.0, 2.0])
    with pytest.raises(ValueError):
        minimum_variance_weights([1.0, 2.0])
    with pytest.raises(ValueError):
        minimum_variance_weights(np.eye(2), max_weight=0.0)
    with pytest.raises(ValueError):
        maximum_sharpe_weights([0.1, 0.2], np.eye(2), risk_free_rate=np.nan)
    with pytest.raises(ValueError):
        maximum_sharpe_weights([0.1, 0.2], np.eye(2), max_weight=0.0)
    with pytest.raises(ValueError):
        maximum_sharpe_weights([0.1, 0.2], np.eye(2), max_weight=0.4)
    with pytest.raises(ValueError):
        risk_parity_weights([1.0, 2.0])


def test_cvar_and_black_litterman_validation_guards() -> None:
    scenarios = np.array([[0.1, 0.2], [0.0, -0.1]])
    with pytest.raises(ValueError):
        cvar_minimization_weights([[0.1, np.nan], [0.0, 0.1]])
    with pytest.raises(ValueError):
        cvar_minimization_weights(scenarios, confidence=0.5)
    with pytest.raises(ValueError):
        cvar_minimization_weights(scenarios, max_weight=0.0)
    with pytest.raises(ValueError):
        cvar_minimization_weights(scenarios, max_weight=0.4)
    with pytest.raises(ValueError):
        black_litterman_posterior(
            [0.1, 0.2], np.eye(2), [[np.nan, 0.0]], [0.1]
        )
    with pytest.raises(ValueError):
        black_litterman_posterior([0.1, 0.2], np.eye(2), [[1.0, 0.0]], [0.1], tau=0.0)
    with pytest.raises(ValueError):
        black_litterman_posterior(
            [0.1, 0.2],
            np.eye(2),
            [[1.0, 0.0]],
            [0.1],
            view_uncertainty=np.array([[np.nan]]),
        )
