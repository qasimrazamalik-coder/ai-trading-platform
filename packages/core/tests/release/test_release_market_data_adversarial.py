from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from quant_os_core.instruments import InstrumentId
from quant_os_core.market_data.models import BookLevel, Candle, OrderBookSnapshot, Quote, Trade
from quant_os_core.market_data.validation import MarketDataValidator

BTC = InstrumentId(venue="TEST", symbol="BTCUSDT")
NOW = datetime(2026, 9, 14, 0, 0, tzinfo=UTC)


def _candle(**overrides: object) -> Candle:
    values: dict[str, object] = {
        "instrument_id": BTC,
        "timestamp": NOW,
        "source": "release-test",
        "timeframe": "1m",
        "open": Decimal("100"),
        "high": Decimal("102"),
        "low": Decimal("99"),
        "close": Decimal("101"),
        "volume": Decimal("10"),
        "close_time": NOW + timedelta(minutes=1),
    }
    values.update(overrides)
    return Candle(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("open", 100.1),
        ("high", 102.1),
        ("low", 99.1),
        ("close", 101.1),
        ("volume", 10.1),
        ("quote_volume", 1000.1),
    ],
)
def test_candle_rejects_binary_float_financial_fields(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match="binary floats"):
        _candle(**{field: value})


def test_trade_quote_and_book_levels_reject_binary_floats() -> None:
    with pytest.raises(ValidationError, match="binary floats"):
        Trade(
            instrument_id=BTC,
            timestamp=NOW,
            source="test",
            trade_id="1",
            price=100.1,
            quantity="1",
        )
    with pytest.raises(ValidationError, match="binary floats"):
        Quote(
            instrument_id=BTC,
            timestamp=NOW,
            source="test",
            bid_price="100",
            ask_price="101",
            bid_size=1.1,
            ask_size="1",
        )
    with pytest.raises(ValidationError, match="binary floats"):
        BookLevel(price=100.1, quantity="1")


def test_market_data_validator_flags_duplicate_gap_future_and_unsorted() -> None:
    validator = MarketDataValidator(max_future_drift=timedelta(seconds=0))
    candles = [
        _candle(timestamp=NOW + timedelta(minutes=3), close_time=NOW + timedelta(minutes=4)),
        _candle(timestamp=NOW, close_time=NOW + timedelta(minutes=1)),
        _candle(timestamp=NOW, close_time=NOW + timedelta(minutes=1)),
    ]
    report = validator.validate_candles(candles, now=NOW + timedelta(minutes=2))
    assert report.is_clean is False
    assert report.is_sorted is False
    assert report.duplicate_timestamps == (NOW,)
    assert report.gaps[0].missing_intervals == 2
    assert report.future_timestamps == (NOW + timedelta(minutes=3),)


def test_crossed_or_badly_sorted_book_is_rejected() -> None:
    with pytest.raises(ValidationError, match="bids must be sorted"):
        OrderBookSnapshot(
            instrument_id=BTC,
            timestamp=NOW,
            source="test",
            bids=(BookLevel(price="99", quantity="1"), BookLevel(price="100", quantity="1")),
            asks=(BookLevel(price="101", quantity="1"),),
        )
    with pytest.raises(ValidationError, match="crossed order book"):
        OrderBookSnapshot(
            instrument_id=BTC,
            timestamp=NOW,
            source="test",
            bids=(BookLevel(price="102", quantity="1"),),
            asks=(BookLevel(price="101", quantity="1"),),
        )
