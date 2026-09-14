"""Built-in strategy registry."""

from __future__ import annotations

from quant_os_core.strategies.mean_reversion import (
    BOLLINGER_METADATA,
    ZSCORE_METADATA,
    bollinger_mean_reversion,
    zscore_mean_reversion,
)
from quant_os_core.strategies.momentum import MOMENTUM_METADATA, time_series_momentum
from quant_os_core.strategies.registry import StrategyRegistry
from quant_os_core.strategies.trend import (
    DONCHIAN_METADATA,
    MA_CROSSOVER_METADATA,
    donchian_breakout,
    moving_average_crossover,
)


def create_default_registry() -> StrategyRegistry:
    registry = StrategyRegistry()
    registry.register(MA_CROSSOVER_METADATA, moving_average_crossover)
    registry.register(DONCHIAN_METADATA, donchian_breakout)
    registry.register(MOMENTUM_METADATA, time_series_momentum)
    registry.register(ZSCORE_METADATA, zscore_mean_reversion)
    registry.register(BOLLINGER_METADATA, bollinger_mean_reversion)
    return registry
