"""Example built-in plugin: the moving-average crossover, clipped to long-only spot.

It doubles as the template for third-party plugins: a module-level ``register(registrar)``
that builds a manifest and hands over a component.

The wrapped strategy can emit ``-1`` (short). Short selling is blocked (R4), so the wrapper
clips exposure to ``[0, 1]``: a bearish cross means "flat", never "short".

``halal_reviewed`` stays ``False`` until the owner (and, for Shariah questions, a scholar)
signs it off, so the registry keeps it research-only.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from quant_os_core.plugins.registry import (
    Permission,
    PluginKind,
    PluginManifest,
    RegisteredPlugin,
    StrategyPlugin,
)
from quant_os_core.strategies.base import StrategyMetadata, StrategySignals
from quant_os_core.strategies.trend import MA_CROSSOVER_METADATA, moving_average_crossover


class Registrar(Protocol):
    def register(self, manifest: PluginManifest, component: object) -> RegisteredPlugin | None: ...


LONG_ONLY_MA_METADATA = StrategyMetadata(
    strategy_id="plugin.long_only_ma_crossover",
    name="Long-only MA crossover (plugin)",
    category=MA_CROSSOVER_METADATA.category,
    description="EMA crossover held long when fast > slow, flat otherwise. No shorts.",
    supported_markets=MA_CROSSOVER_METADATA.supported_markets,
    parameters=MA_CROSSOVER_METADATA.parameters,
    known_failure_modes=MA_CROSSOVER_METADATA.known_failure_modes,
)


def long_only_ma_crossover(
    prices: npt.ArrayLike, *, fast_window: int = 20, slow_window: int = 50
) -> StrategySignals:
    raw = moving_average_crossover(prices, fast_window=fast_window, slow_window=slow_window)
    return StrategySignals(
        positions=np.clip(raw.positions, 0.0, 1.0),
        metadata=LONG_ONLY_MA_METADATA,
        diagnostics=raw.diagnostics,
    )


MANIFEST = PluginManifest(
    name="builtin.long_only_ma_crossover",
    version="1.0.0",
    kind=PluginKind.STRATEGY,
    halal_reviewed=False,
    permissions=frozenset({Permission.READ_MARKET_DATA}),
    description=LONG_ONLY_MA_METADATA.description,
)


def register(registrar: Registrar) -> None:
    registrar.register(MANIFEST, StrategyPlugin(LONG_ONLY_MA_METADATA, long_only_ma_crossover))
