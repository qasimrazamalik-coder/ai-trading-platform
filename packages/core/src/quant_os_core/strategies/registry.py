"""Discoverable strategy registry with duplicate protection."""

from __future__ import annotations

from collections.abc import Callable

from quant_os_core.strategies.base import StrategyMetadata, StrategySignals

StrategyFactory = Callable[..., StrategySignals]


class StrategyRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, tuple[StrategyMetadata, StrategyFactory]] = {}

    def register(self, metadata: StrategyMetadata, factory: StrategyFactory) -> None:
        if metadata.strategy_id in self._strategies:
            raise ValueError(f"strategy already registered: {metadata.strategy_id}")
        self._strategies[metadata.strategy_id] = (metadata, factory)

    def get(self, strategy_id: str) -> tuple[StrategyMetadata, StrategyFactory]:
        try:
            return self._strategies[strategy_id]
        except KeyError as exc:
            raise KeyError(f"unknown strategy: {strategy_id}") from exc

    def list_metadata(self) -> tuple[StrategyMetadata, ...]:
        ordered = sorted(
            self._strategies.values(),
            key=lambda item: item[0].strategy_id,
        )
        return tuple(item[0] for item in ordered)

    def __contains__(self, strategy_id: object) -> bool:
        return strategy_id in self._strategies
