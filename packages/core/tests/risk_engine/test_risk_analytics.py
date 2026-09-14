import numpy as np
import pytest

from quant_os_core.risk_engine import (
    correlation_concentration,
    leverage_ratios,
    portfolio_heat_fraction,
    stress_portfolio,
)


def test_portfolio_heat_fraction() -> None:
    value = portfolio_heat_fraction([2, -1], [100, 50], [90, 55], 1000)
    assert value == pytest.approx(0.025)


def test_leverage_ratios() -> None:
    gross, net = leverage_ratios([500, -200], 1000)
    assert gross == pytest.approx(0.7)
    assert net == pytest.approx(0.3)


def test_correlation_concentration_detects_cluster() -> None:
    corr = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
    assert correlation_concentration([0.4, 0.3, 0.3], corr) == pytest.approx(0.7)


def test_stress_portfolio_returns_named_scenarios() -> None:
    results = stress_portfolio(
        [0.6, 0.4],
        {"crash": [-0.20, -0.10], "rally": [0.10, 0.05]},
    )
    assert results[0].shock_name == "crash"
    assert results[0].pnl_fraction == pytest.approx(-0.16)
    assert results[1].stressed_equity_fraction == pytest.approx(1.08)


def test_risk_analytics_reject_malformed_inputs() -> None:
    with pytest.raises(ValueError):
        portfolio_heat_fraction([], [], [], 1000)
    with pytest.raises(ValueError):
        portfolio_heat_fraction([1], [100], [90], 0)
    with pytest.raises(ValueError):
        leverage_ratios([], 1000)
    with pytest.raises(ValueError):
        leverage_ratios([100], 0)
    with pytest.raises(ValueError):
        correlation_concentration([0.5, 0.5], np.eye(3))
    with pytest.raises(ValueError):
        correlation_concentration([0.5, 0.5], np.eye(2), threshold=2.0)
    with pytest.raises(ValueError):
        stress_portfolio([1.0], {})
    with pytest.raises(ValueError):
        stress_portfolio([1.0], {"bad": [0.1, 0.2]})


def test_zero_weight_correlation_concentration_is_zero() -> None:
    assert correlation_concentration([0.0, 0.0], np.eye(2)) == 0.0
