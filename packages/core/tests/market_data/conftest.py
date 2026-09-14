from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_os_core.instruments import InstrumentId
from quant_os_core.market_data.models import Candle


@pytest.fixture
def btc() -> InstrumentId:
    return InstrumentId(venue="binance", symbol="btcusdt")


@pytest.fixture
def minute_candles(btc: InstrumentId) -> list[Candle]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles: list[Candle] = []
    for i in range(5):
        price = Decimal("100") + Decimal(i)
        candles.append(
            Candle(
                instrument_id=btc,
                timestamp=start + timedelta(minutes=i),
                close_time=start + timedelta(minutes=i + 1),
                source="fixture",
                timeframe="1m",
                open=price,
                high=price + Decimal("2"),
                low=price - Decimal("1"),
                close=price + Decimal("1"),
                volume=Decimal("10") + Decimal(i),
                quote_volume=Decimal("1000") + Decimal(i),
                trade_count=100 + i,
            )
        )
    return candles
