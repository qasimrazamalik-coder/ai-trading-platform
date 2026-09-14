from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from quant_os_core.instruments import AssetClass, Instrument, InstrumentId, InstrumentType
from quant_os_core.market_data.models import BookLevel, OrderBookSnapshot, Quote, Trade


def test_instrument_id_normalizes_tokens() -> None:
    instrument = InstrumentId(venue=" binance ", symbol=" btcusdt ")
    assert instrument.key == "BINANCE:BTCUSDT"


def test_instrument_metadata_normalizes_currency() -> None:
    instrument = Instrument(
        instrument_id=InstrumentId(venue="psx", symbol="ogdc"),
        asset_class=AssetClass.EQUITY,
        instrument_type=InstrumentType.CASH,
        base_currency="ogdc",
        quote_currency="pkr",
        price_precision=2,
        quantity_precision=0,
    )
    assert instrument.quote_currency == "PKR"


def test_quote_computes_exact_midpoint_and_spread(btc: InstrumentId) -> None:
    quote = Quote(
        instrument_id=btc,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        source="test",
        bid_price="99.9",
        ask_price="100.1",
        bid_size="3",
        ask_size="2",
    )
    assert quote.midpoint == Decimal("100.0")
    assert quote.spread == Decimal("0.2")


def test_crossed_quote_is_rejected(btc: InstrumentId) -> None:
    with pytest.raises(ValidationError):
        Quote(
            instrument_id=btc,
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            source="test",
            bid_price="101",
            ask_price="100",
            bid_size="1",
            ask_size="1",
        )


def test_trade_requires_positive_quantity(btc: InstrumentId) -> None:
    with pytest.raises(ValidationError):
        Trade(
            instrument_id=btc,
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            source="test",
            trade_id="t1",
            price="100",
            quantity="0",
        )


def test_order_book_requires_sorted_sides(btc: InstrumentId) -> None:
    with pytest.raises(ValidationError):
        OrderBookSnapshot(
            instrument_id=btc,
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            source="test",
            bids=(BookLevel(price="99", quantity="1"), BookLevel(price="100", quantity="1")),
            asks=(BookLevel(price="101", quantity="1"),),
        )


def test_order_book_best_prices(btc: InstrumentId) -> None:
    book = OrderBookSnapshot(
        instrument_id=btc,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        source="test",
        bids=(BookLevel(price="100", quantity="1"), BookLevel(price="99", quantity="2")),
        asks=(BookLevel(price="101", quantity="3"), BookLevel(price="102", quantity="4")),
    )
    assert book.best_bid.price == Decimal("100")
    assert book.best_ask.price == Decimal("101")
