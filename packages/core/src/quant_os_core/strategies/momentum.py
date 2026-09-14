"""Time-series and cross-sectional momentum strategies."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from quant_os_core.quant._validation import as_float_array
from quant_os_core.strategies.base import (
    ParameterDefinition,
    StrategyCategory,
    StrategyMetadata,
    StrategySignals,
)
from quant_os_core.strategies.indicators import rate_of_change

MOMENTUM_METADATA = StrategyMetadata(
    strategy_id="momentum.time_series",
    name="Time-Series Momentum",
    category=StrategyCategory.MOMENTUM,
    description="Direction follows the sign of trailing return with an optional neutral band.",
    parameters=(
        ParameterDefinition("lookback", 20, 1, 500),
        ParameterDefinition("threshold", 0.0, 0.0, 1.0),
    ),
    known_failure_modes=("momentum crashes", "fast reversals", "high turnover at low thresholds"),
)


def time_series_momentum(
    prices: npt.ArrayLike,
    *,
    lookback: int = 20,
    threshold: float = 0.0,
) -> StrategySignals:
    values = as_float_array(prices, name="prices", min_size=2)
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or threshold_value < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    momentum = rate_of_change(values, lookback)
    positions = np.zeros(values.size, dtype=np.float64)
    positions[momentum > threshold_value] = 1.0
    positions[momentum < -threshold_value] = -1.0
    return StrategySignals(
        positions=positions,
        metadata=MOMENTUM_METADATA,
        diagnostics={"momentum": np.nan_to_num(momentum)},
    )


def cross_sectional_momentum(
    prices: npt.ArrayLike,
    *,
    lookback: int = 20,
    long_fraction: float = 0.2,
    short_fraction: float = 0.2,
) -> npt.NDArray[np.float64]:
    """Return time x asset target weights based on trailing relative momentum ranks."""
    matrix = np.asarray(prices, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError("prices must be a 2-D time x asset matrix")
    if not np.isfinite(matrix).all() or np.any(matrix <= 0.0):
        raise ValueError("prices must be finite and > 0")
    window = int(lookback)
    if window != lookback or window <= 0 or window >= matrix.shape[0]:
        raise ValueError("lookback must be a positive integer smaller than time dimension")
    long_share = float(long_fraction)
    short_share = float(short_fraction)
    if not 0.0 < long_share <= 0.5 or not 0.0 < short_share <= 0.5:
        raise ValueError("long_fraction and short_fraction must be in (0, 0.5]")

    assets = matrix.shape[1]
    long_count = max(1, int(np.floor(assets * long_share)))
    short_count = max(1, int(np.floor(assets * short_share)))
    if long_count + short_count > assets:
        raise ValueError("long and short baskets cannot overlap")

    weights = np.zeros_like(matrix)
    for index in range(window, matrix.shape[0]):
        trailing = matrix[index] / matrix[index - window] - 1.0
        ranking = np.argsort(trailing)
        short_assets = ranking[:short_count]
        long_assets = ranking[-long_count:]
        weights[index, short_assets] = -1.0 / short_count
        weights[index, long_assets] = 1.0 / long_count
    return weights
