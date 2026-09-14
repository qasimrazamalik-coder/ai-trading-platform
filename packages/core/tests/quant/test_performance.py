import numpy as np
import pytest

from quant_os_core.quant.performance import (
    annualized_return,
    calmar_ratio,
    downside_deviation,
    expectancy,
    hit_rate,
    omega_ratio,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)


def test_annualized_return_one_period() -> None:
    assert annualized_return([0.10], periods_per_year=1) == pytest.approx(0.10)


def test_sharpe_matches_manual_formula() -> None:
    returns = np.array([0.01, 0.02, -0.01, 0.03])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_zero_volatility_sharpe_rejected() -> None:
    with pytest.raises(ValueError):
        sharpe_ratio([0.01, 0.01, 0.01])


def test_downside_and_sortino() -> None:
    returns = np.array([0.02, -0.01, 0.01, -0.02])
    downside = np.minimum(returns, 0.0)
    expected_dev = np.sqrt(np.mean(downside**2)) * np.sqrt(12)
    assert downside_deviation(returns, periods_per_year=12) == pytest.approx(expected_dev)
    expected_sortino = returns.mean() / np.sqrt(np.mean(downside**2)) * np.sqrt(12)
    assert sortino_ratio(returns, periods_per_year=12) == pytest.approx(expected_sortino)


def test_calmar_uses_absolute_max_drawdown() -> None:
    result = calmar_ratio([0.10, -0.20, 0.05, 0.03], periods_per_year=4)
    assert np.isfinite(result)


def test_omega_ratio() -> None:
    assert omega_ratio([0.03, 0.02, -0.01, -0.02]) == pytest.approx(5 / 3)


def test_trade_metrics() -> None:
    trades = [0.10, -0.05, 0.03, -0.02]
    assert profit_factor(trades) == pytest.approx(0.13 / 0.07)
    assert expectancy(trades) == pytest.approx(0.015)
    assert hit_rate(trades) == pytest.approx(0.5)


def test_downside_metrics_reject_invalid_annual_mar() -> None:
    with pytest.raises(ValueError):
        downside_deviation([0.01, -0.01], minimum_acceptable_return=-1.0)
    with pytest.raises(ValueError):
        sortino_ratio([0.01, -0.01], minimum_acceptable_return=float("nan"))
