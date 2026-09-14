import pytest
from pydantic import ValidationError

from quant_os_core.events import BarEvent, OrderIntent, Side


def test_valid_bar_event() -> None:
    bar = BarEvent(
        source="test",
        symbol="BTCUSDT",
        timeframe="1m",
        open=100,
        high=110,
        low=90,
        close=105,
        volume=12,
    )
    assert bar.symbol == "BTCUSDT"


def test_invalid_bar_rejected() -> None:
    with pytest.raises(ValidationError):
        BarEvent(
            source="test",
            symbol="BTCUSDT",
            timeframe="1m",
            open=100,
            high=99,
            low=90,
            close=105,
            volume=12,
        )


def test_order_intent_defaults_to_typed_event() -> None:
    intent = OrderIntent(
        source="strategy",
        strategy_id="trend-1",
        symbol="EURUSD",
        side=Side.BUY,
        quantity=1,
    )
    assert intent.side is Side.BUY


def test_financial_event_float_is_rejected() -> None:
    with pytest.raises(ValidationError):
        OrderIntent(
            source="strategy",
            strategy_id="trend-1",
            symbol="BTCUSDT",
            side=Side.BUY,
            quantity=1.1,
        )
