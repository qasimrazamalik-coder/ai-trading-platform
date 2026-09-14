"""Mean-reversion strategy families."""

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
from quant_os_core.strategies.indicators import bollinger_bands, rolling_zscore

ZSCORE_METADATA = StrategyMetadata(
    strategy_id="mean_reversion.zscore",
    name="Rolling Z-Score Mean Reversion",
    category=StrategyCategory.MEAN_REVERSION,
    description="Fades statistically extreme deviations from a rolling mean.",
    parameters=(
        ParameterDefinition("window", 20, 2, 500),
        ParameterDefinition("entry_z", 2.0, 0.1, 10.0),
        ParameterDefinition("exit_z", 0.5, 0.0, 5.0),
    ),
    known_failure_modes=("structural breaks", "strong trends", "non-stationary series"),
)

BOLLINGER_METADATA = StrategyMetadata(
    strategy_id="mean_reversion.bollinger",
    name="Bollinger Mean Reversion",
    category=StrategyCategory.MEAN_REVERSION,
    description="Fades closes outside volatility bands and exits near the rolling mean.",
    parameters=(
        ParameterDefinition("window", 20, 2, 500),
        ParameterDefinition("num_std", 2.0, 0.1, 10.0),
    ),
    known_failure_modes=("band walking in strong trends", "volatility regime shifts"),
)


def zscore_mean_reversion(
    prices: npt.ArrayLike,
    *,
    window: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> StrategySignals:
    values = as_float_array(prices, name="prices", min_size=3)
    entry = float(entry_z)
    exit_value = float(exit_z)
    if not np.isfinite(entry) or not np.isfinite(exit_value):
        raise ValueError("z-score thresholds must be finite")
    if entry <= 0.0 or exit_value < 0.0 or exit_value >= entry:
        raise ValueError("require entry_z > exit_z >= 0")
    scores = rolling_zscore(values, window)
    positions = np.zeros(values.size, dtype=np.float64)
    state = 0.0
    for index, score in enumerate(scores):
        if not np.isfinite(score):
            positions[index] = 0.0
            continue
        if state == 0.0:
            if score <= -entry:
                state = 1.0
            elif score >= entry:
                state = -1.0
        elif abs(score) <= exit_value:
            state = 0.0
        positions[index] = state
    return StrategySignals(
        positions=positions,
        metadata=ZSCORE_METADATA,
        diagnostics={"zscore": np.nan_to_num(scores)},
    )


def bollinger_mean_reversion(
    prices: npt.ArrayLike,
    *,
    window: int = 20,
    num_std: float = 2.0,
) -> StrategySignals:
    values = as_float_array(prices, name="prices", min_size=3)
    lower, middle, upper = bollinger_bands(values, window=window, num_std=num_std)
    positions = np.zeros(values.size, dtype=np.float64)
    state = 0.0
    for index, price in enumerate(values):
        if not np.isfinite(middle[index]):
            continue
        if state == 0.0 and price < lower[index]:
            state = 1.0
        elif state == 0.0 and price > upper[index]:
            state = -1.0
        elif state > 0.0 and price >= middle[index]:
            state = 0.0
        elif state < 0.0 and price <= middle[index]:
            state = 0.0
        positions[index] = state
    return StrategySignals(
        positions=positions,
        metadata=BOLLINGER_METADATA,
        diagnostics={
            "lower": np.nan_to_num(lower),
            "middle": np.nan_to_num(middle),
            "upper": np.nan_to_num(upper),
        },
    )
