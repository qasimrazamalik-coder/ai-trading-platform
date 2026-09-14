"""Storage contracts and a deterministic in-memory implementation for tests/replay."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Protocol, runtime_checkable

from quant_os_core.instruments import InstrumentId
from quant_os_core.market_data.models import Candle


@runtime_checkable
class CandleStore(Protocol):
    def append(self, candles: list[Candle]) -> None: ...

    def read(
        self,
        instrument_id: InstrumentId,
        timeframe: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Candle]: ...


class InMemoryCandleStore:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], list[Candle]] = defaultdict(list)

    def append(self, candles: list[Candle]) -> None:
        for candle in candles:
            key = (candle.instrument_id.key, candle.timeframe)
            self._items[key].append(candle)
            self._items[key].sort(key=lambda item: item.timestamp)

    def read(
        self,
        instrument_id: InstrumentId,
        timeframe: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Candle]:
        items = list(self._items.get((instrument_id.key, timeframe), ()))
        if start is not None:
            items = [item for item in items if item.timestamp >= start]
        if end is not None:
            items = [item for item in items if item.timestamp < end]
        return items
