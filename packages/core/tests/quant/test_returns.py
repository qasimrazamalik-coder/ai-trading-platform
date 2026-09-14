import numpy as np
import pytest

from quant_os_core.quant.returns import (
    cumulative_return,
    drawdown_series,
    log_returns,
    max_drawdown,
    simple_returns,
    wealth_index,
)


def test_simple_returns_known_values() -> None:
    result = simple_returns([100, 110, 99])
    np.testing.assert_allclose(result, [0.10, -0.10])


def test_log_returns_compound_exactly() -> None:
    result = log_returns([100, 110, 121])
    np.testing.assert_allclose(result, [np.log(1.1), np.log(1.1)])
    assert np.isclose(result.sum(), np.log(1.21))


def test_prices_must_be_positive() -> None:
    with pytest.raises(ValueError):
        simple_returns([100, 0, 101])


def test_cumulative_return_compounds() -> None:
    assert cumulative_return([0.10, -0.10]) == pytest.approx(-0.01)


def test_total_loss_stops_wealth_at_zero() -> None:
    assert cumulative_return([0.1, -1.0, 0.5]) == -1.0


def test_impossible_return_rejected() -> None:
    with pytest.raises(ValueError):
        wealth_index([0.1, -1.01])


def test_wealth_and_drawdown() -> None:
    returns = [0.10, -0.20, 0.10]
    np.testing.assert_allclose(wealth_index(returns), [1.0, 1.1, 0.88, 0.968])
    np.testing.assert_allclose(drawdown_series(returns), [0.0, 0.0, -0.2, -0.12])
    assert max_drawdown(returns) == pytest.approx(-0.2)
