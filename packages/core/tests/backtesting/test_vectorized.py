import numpy as np
import pytest

from quant_os_core.backtesting.vectorized import run_vectorized_backtest


def test_vectorized_delays_signal():
    result = run_vectorized_backtest(
        [100, 110, 121, 133.1],
        [1, 1, 1, 1],
        signal_delay=1,
    )
    np.testing.assert_allclose(result.applied_positions, [0, 1, 1])
    np.testing.assert_allclose(result.strategy_returns, [0, 0.1, 0.1])
    assert result.total_return == pytest.approx(0.21)


def test_vectorized_transaction_cost_on_turnover():
    result = run_vectorized_backtest(
        [100, 100, 100, 100],
        [1, -1, 0, 0],
        signal_delay=0,
        fee_bps=10,
    )
    np.testing.assert_allclose(result.turnover, [1, 2, 1])
    assert result.total_return < 0


def test_vectorized_rejects_nonpositive_prices():
    with pytest.raises(ValueError):
        run_vectorized_backtest([100, 0], [1, 1])


def test_vectorized_rejects_negative_costs():
    with pytest.raises(ValueError):
        run_vectorized_backtest([100, 101], [1, 1], fee_bps=-1)
