import numpy as np
import pytest

from quant_os_core.quant._validation import as_float_array, paired_float_arrays
from quant_os_core.quant.performance import (
    annualized_return,
    calmar_ratio,
    omega_ratio,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)
from quant_os_core.quant.portfolio import (
    correlation_matrix,
    covariance_matrix,
    marginal_risk_contribution,
    nearest_psd,
    portfolio_variance,
    risk_contribution_percentages,
)
from quant_os_core.quant.regression import (
    beta,
    engle_granger,
    information_ratio,
    mean_reversion_half_life,
    ordinary_least_squares,
)
from quant_os_core.quant.returns import cumulative_return, log_returns, wealth_index
from quant_os_core.quant.risk import gaussian_cvar, gaussian_var
from quant_os_core.quant.sizing import fractional_kelly, kelly_from_moments
from quant_os_core.quant.statistics import autocorrelation, rolling_mean, rolling_std, zscore
from quant_os_core.quant.volatility import ewma_variance


def test_array_validation_rejects_shape_size_infinity_and_pair_mismatch() -> None:
    with pytest.raises(ValueError):
        as_float_array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        as_float_array([], min_size=1)
    with pytest.raises(ValueError):
        as_float_array([1.0, float("inf")])
    with pytest.raises(ValueError):
        paired_float_arrays([1.0, 2.0], [1.0, 2.0, 3.0])


def test_return_domain_guards() -> None:
    with pytest.raises(ValueError):
        log_returns([100.0, -1.0])
    with pytest.raises(ValueError):
        cumulative_return([-1.1])
    with pytest.raises(ValueError):
        wealth_index([0.01], initial=0.0)


def test_performance_undefined_cases_raise() -> None:
    with pytest.raises(ValueError):
        annualized_return([-1.1])
    assert annualized_return([-1.0]) == -1.0
    with pytest.raises(ValueError):
        sharpe_ratio([0.01, 0.02], risk_free_rate=-1.0)
    with pytest.raises(ValueError):
        sortino_ratio([0.01, 0.02])
    with pytest.raises(ValueError):
        calmar_ratio([0.01, 0.02])
    with pytest.raises(ValueError):
        omega_ratio([0.01, 0.02])
    with pytest.raises(ValueError):
        omega_ratio([0.01, -0.01], threshold=float("inf"))
    with pytest.raises(ValueError):
        profit_factor([0.01, 0.02])


def test_statistical_guard_cases_raise() -> None:
    with pytest.raises(ValueError):
        zscore([1.0, 2.0], ddof=2)
    with pytest.raises(ValueError):
        zscore([1.0, 1.0])
    with pytest.raises(ValueError):
        rolling_mean([1.0, 2.0], 0)
    with pytest.raises(ValueError):
        rolling_std([1.0, 2.0], 1)
    with pytest.raises(ValueError):
        rolling_std([1.0, 2.0, 3.0], 3, ddof=2)
    with pytest.raises(ValueError):
        autocorrelation([1.0, 2.0, 3.0], lag=3)


def test_portfolio_input_guards() -> None:
    with pytest.raises(ValueError):
        covariance_matrix([0.01, 0.02])
    with pytest.raises(ValueError):
        covariance_matrix([[0.01, float("nan")], [0.02, 0.03]])
    with pytest.raises(ValueError):
        correlation_matrix([[0.01, 1.0], [0.02, 1.0], [0.03, 1.0]])
    with pytest.raises(ValueError):
        portfolio_variance([1.0], [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        portfolio_variance([0.5, 0.5], [[1.0, 0.2], [0.1, 1.0]])
    with pytest.raises(ValueError):
        marginal_risk_contribution([1.0], [[0.0]])
    with pytest.raises(ValueError):
        risk_contribution_percentages([1.0], [[0.0]])
    with pytest.raises(ValueError):
        nearest_psd([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError):
        nearest_psd([[1.0, float("inf")], [0.0, 1.0]])


def test_regression_guards() -> None:
    with pytest.raises(ValueError):
        ordinary_least_squares([1.0, 2.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        beta([1.0, 2.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        information_ratio([0.01, 0.02], [0.01, 0.02])
    with pytest.raises(ValueError):
        engle_granger(np.arange(20.0), np.arange(21.0))

    trend = np.arange(20.0)
    with pytest.raises(ValueError):
        mean_reversion_half_life(trend)


def test_sizing_and_volatility_guards() -> None:
    with pytest.raises(ValueError):
        fractional_kelly(0.5, 1.0, fraction=1.1)
    with pytest.raises(ValueError):
        kelly_from_moments(float("nan"), 0.1)
    with pytest.raises(ValueError):
        ewma_variance([0.01, 0.02], initial_variance=-0.1)


def test_gaussian_zero_volatility_behavior() -> None:
    assert gaussian_var([-0.01, -0.01]) == pytest.approx(0.01)
    assert gaussian_cvar([-0.01, -0.01]) == pytest.approx(0.01)
