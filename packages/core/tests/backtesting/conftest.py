from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_os_core.instruments import InstrumentId
from quant_os_core.market_data.models import Candle


@pytest.fixture
def instrument() -> InstrumentId:
    return InstrumentId(venue="TEST", symbol="ABCUSD")


@pytest.fixture
def candles(instrument: InstrumentId) -> tuple[Candle, ...]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        ("100", "103", "99", "102", "100"),
        ("103", "106", "101", "105", "100"),
        ("104", "108", "103", "107", "100"),
        ("106", "109", "100", "101", "100"),
        ("100", "104", "98", "103", "100"),
    ]
    result = []
    for i, (open_, high, low, close, volume) in enumerate(rows):
        result.append(
            Candle(
                instrument_id=instrument,
                timestamp=start + timedelta(days=i),
                close_time=start + timedelta(days=i + 1),
                source="fixture",
                timeframe="1d",
                open=Decimal(open_),
                high=Decimal(high),
                low=Decimal(low),
                close=Decimal(close),
                volume=Decimal(volume),
            )
        )
    return tuple(result)
