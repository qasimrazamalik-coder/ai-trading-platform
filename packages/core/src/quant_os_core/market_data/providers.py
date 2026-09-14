"""Provider interfaces keep exchange/broker SDK details outside the domain core."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Protocol, runtime_checkable

from quant_os_core.instruments import Instrument, InstrumentId
from quant_os_core.market_data.models import Candle, OrderBookSnapshot, Quote, Trade

MarketDataStreamItem = Candle | Trade | Quote | OrderBookSnapshot


@runtime_checkable
class MarketDataProvider(Protocol):
    @property
    def name(self) -> str: ...

    async def instruments(self) -> list[Instrument]: ...

    async def historical_candles(
        self,
        instrument_id: InstrumentId,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]: ...

    def stream(self, instruments: set[InstrumentId]) -> AsyncIterator[MarketDataStreamItem]: ...
