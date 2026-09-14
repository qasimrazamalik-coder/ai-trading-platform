"""Trend-following strategy families."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from quant_os_core.quant._validation import as_float_array, paired_float_arrays
from quant_os_core.strategies.base import (
    ParameterDefinition,
    StrategyCategory,
    StrategyMetadata,
    StrategySignals,
)
from quant_os_core.strategies.indicators import donchian_channels, ema

MA_CROSSOVER_METADATA = StrategyMetadata(
    strategy_id="trend.ma_crossover",
    name="Moving Average Crossover",
    category=StrategyCategory.TREND,
    description="Long above the slow EMA and short below it using a fast/slow EMA pair.",
    parameters=(
        ParameterDefinition("fast_window", 20, 2, 250),
        ParameterDefinition("slow_window", 50, 3, 500),
    ),
    known_failure_modes=("range-bound whipsaw", "transaction-cost sensitivity"),
)

DONCHIAN_METADATA = StrategyMetadata(
    strategy_id="breakout.donchian",
    name="Donchian Breakout",
    category=StrategyCategory.BREAKOUT,
    description="Turtle-style breakout using prior-window highs and lows.",
    parameters=(
        ParameterDefinition("entry_window", 20, 2, 500),
        ParameterDefinition("exit_window", 10, 2, 250),
    ),
    known_failure_modes=("false breakouts", "gap risk", "choppy markets"),
)


def moving_average_crossover(
    prices: npt.ArrayLike,
    *,
    fast_window: int = 20,
    slow_window: int = 50,
) -> StrategySignals:
    values = as_float_array(prices, name="prices", min_size=3)
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window")
    fast = ema(values, fast_window)
    slow = ema(values, slow_window)
    positions = np.zeros(values.size, dtype=np.float64)
    valid = np.isfinite(fast) & np.isfinite(slow)
    positions[valid & (fast > slow)] = 1.0
    positions[valid & (fast < slow)] = -1.0
    return StrategySignals(
        positions=positions,
        metadata=MA_CROSSOVER_METADATA,
        diagnostics={
            "fast_ema": np.nan_to_num(fast),
            "slow_ema": np.nan_to_num(slow),
        },
    )


def donchian_breakout(
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    close: npt.ArrayLike,
    *,
    entry_window: int = 20,
    exit_window: int = 10,
) -> StrategySignals:
    highs, lows = paired_float_arrays(high, low, left_name="high", right_name="low", min_size=3)
    closes = as_float_array(close, name="close", min_size=3)
    if closes.size != highs.size:
        raise ValueError("high, low and close must have equal length")
    if exit_window >= entry_window:
        raise ValueError("exit_window must be smaller than entry_window")
    upper_entry, lower_entry = donchian_channels(highs, lows, entry_window)
    upper_exit, lower_exit = donchian_channels(highs, lows, exit_window)
    positions = np.zeros(closes.size, dtype=np.float64)
    state = 0.0
    for index in range(closes.size):
        if np.isfinite(upper_entry[index]) and closes[index] > upper_entry[index]:
            state = 1.0
        elif np.isfinite(lower_entry[index]) and closes[index] < lower_entry[index]:
            state = -1.0
        elif state > 0.0 and np.isfinite(lower_exit[index]) and closes[index] < lower_exit[index]:
            state = 0.0
        elif state < 0.0 and np.isfinite(upper_exit[index]) and closes[index] > upper_exit[index]:
            state = 0.0
        positions[index] = state
    return StrategySignals(
        positions=positions,
        metadata=DONCHIAN_METADATA,
        diagnostics={
            "entry_upper": np.nan_to_num(upper_entry),
            "entry_lower": np.nan_to_num(lower_entry),
            "exit_upper": np.nan_to_num(upper_exit),
            "exit_lower": np.nan_to_num(lower_exit),
        },
    )
