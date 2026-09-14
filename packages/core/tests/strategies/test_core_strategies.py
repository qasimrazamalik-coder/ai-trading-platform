import numpy as np
import pytest

from quant_os_core.strategies import (
    bollinger_mean_reversion,
    cross_sectional_momentum,
    donchian_breakout,
    moving_average_crossover,
    time_series_momentum,
    zscore_mean_reversion,
)


def test_ma_crossover_detects_uptrend() -> None:
    prices = np.linspace(100, 200, 100)
    result = moving_average_crossover(prices, fast_window=5, slow_window=20)
    assert result.positions[-1] == 1.0
    assert result.metadata.strategy_id == "trend.ma_crossover"


def test_ma_crossover_guard() -> None:
    with pytest.raises(ValueError, match="fast_window"):
        moving_average_crossover(np.arange(1.0, 20.0), fast_window=10, slow_window=5)


def test_donchian_breakout_uses_prior_channel() -> None:
    close = np.array([10, 10, 10, 10, 12, 12, 9, 8], dtype=float)
    high = close + 0.1
    low = close - 0.1
    result = donchian_breakout(high, low, close, entry_window=3, exit_window=2)
    assert result.positions[4] == 1.0
    assert result.positions[-1] <= 0.0


def test_donchian_window_guard() -> None:
    values = np.arange(1.0, 20.0)
    with pytest.raises(ValueError, match="exit_window"):
        donchian_breakout(values, values, values, entry_window=5, exit_window=5)


def test_time_series_momentum_sign() -> None:
    prices = [100, 100, 102, 104, 100]
    result = time_series_momentum(prices, lookback=2)
    assert result.positions[3] == 1.0
    assert result.positions[4] == -1.0


def test_momentum_threshold_creates_neutral_zone() -> None:
    prices = [100, 100.5, 100.7, 100.8, 100.9]
    result = time_series_momentum(prices, lookback=1, threshold=0.01)
    assert np.all(result.positions == 0.0)


def test_cross_sectional_momentum_is_market_neutral() -> None:
    time = np.arange(30, dtype=float)
    matrix = np.column_stack(
        [100 + 3 * time, 100 + 2 * time, 100 + time, 100 - 0.3 * time]
    )
    weights = cross_sectional_momentum(matrix, lookback=5, long_fraction=0.25, short_fraction=0.25)
    assert weights[-1, 0] == pytest.approx(1.0)
    assert weights[-1, 3] == pytest.approx(-1.0)
    assert weights[-1].sum() == pytest.approx(0.0)


def test_cross_sectional_fraction_guard() -> None:
    with pytest.raises(ValueError, match="fraction"):
        cross_sectional_momentum(np.ones((10, 4)), lookback=2, long_fraction=0.9)


def test_zscore_mean_reversion_fades_extreme() -> None:
    prices = np.concatenate([np.full(25, 100.0), [80.0, 100.0, 120.0]])
    result = zscore_mean_reversion(prices, window=20, entry_z=2.0, exit_z=0.5)
    assert result.positions[25] == 1.0
    assert result.positions[26] == 0.0
    assert result.positions[27] == -1.0


def test_zscore_threshold_guard() -> None:
    with pytest.raises(ValueError, match="entry_z"):
        zscore_mean_reversion(np.arange(1.0, 30.0), entry_z=1.0, exit_z=1.0)


def test_bollinger_mean_reversion_generates_valid_positions() -> None:
    prices = np.concatenate([np.full(25, 100.0), [80.0, 100.0]])
    result = bollinger_mean_reversion(prices, window=20, num_std=1.5)
    assert set(np.unique(result.positions)).issubset({-1.0, 0.0, 1.0})
    assert result.positions[25] == 1.0
