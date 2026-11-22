from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_os_core.backtesting.models import OrderSide, OrderStatus, OrderType
from quant_os_core.execution import BrokerOrderRequest, PaperBroker
from quant_os_core.instruments import InstrumentId

BTC = InstrumentId(venue="PAPER", symbol="BTCUSDT")


def request(
    client_id: str,
    side: OrderSide,
    quantity: str = "1",
    *,
    order_type: OrderType = OrderType.MARKET,
    limit_price: str | None = None,
    stop_price: str | None = None,
) -> BrokerOrderRequest:
    return BrokerOrderRequest(
        client_order_id=client_id,
        instrument_id=BTC,
        side=side,
        quantity=Decimal(quantity),
        order_type=order_type,
        limit_price=None if limit_price is None else Decimal(limit_price),
        stop_price=None if stop_price is None else Decimal(stop_price),
    )


def test_market_order_fills_and_marks_account() -> None:
    broker = PaperBroker(
        initial_cash=Decimal("10000"),
        fee_rate=Decimal("0"),
        slippage_bps=Decimal("0"),
    )
    broker.update_mark(BTC, Decimal("100"))
    state = broker.submit(request("one", OrderSide.BUY, "2"))
    assert state.status is OrderStatus.FILLED
    assert state.average_fill_price == Decimal("100")
    account = broker.account()
    assert account.cash == Decimal("9800")
    assert account.equity == Decimal("10000")
    assert account.positions[0].quantity == Decimal("2")


def test_client_order_id_is_idempotent_but_not_mutable() -> None:
    broker = PaperBroker(fee_rate=Decimal("0"), slippage_bps=Decimal("0"))
    broker.update_mark(BTC, Decimal("100"))
    original = request("same", OrderSide.BUY)
    first = broker.submit(original)
    second = broker.submit(original)
    assert first == second
    assert len(broker.fills) == 1
    with pytest.raises(ValueError, match="different request"):
        broker.submit(request("same", OrderSide.BUY, "2"))


def test_limit_order_waits_and_never_fills_above_buy_limit() -> None:
    broker = PaperBroker(fee_rate=Decimal("0"), slippage_bps=Decimal("100"))
    broker.update_mark(BTC, Decimal("105"))
    state = broker.submit(
        request("limit", OrderSide.BUY, order_type=OrderType.LIMIT, limit_price="100")
    )
    assert state.status is OrderStatus.PENDING
    broker.update_mark(BTC, Decimal("99"), timestamp=datetime(2026, 1, 1, tzinfo=UTC))
    filled = broker.order("limit")
    assert filled is not None
    assert filled.status is OrderStatus.FILLED
    assert filled.average_fill_price == Decimal("99.99")
    assert filled.average_fill_price <= Decimal("100")


def test_stop_order_triggers_in_correct_direction() -> None:
    broker = PaperBroker(fee_rate=Decimal("0"), slippage_bps=Decimal("0"))
    broker.update_mark(BTC, Decimal("100"))
    state = broker.submit(
        request("stop", OrderSide.BUY, order_type=OrderType.STOP, stop_price="105")
    )
    assert state.status is OrderStatus.PENDING
    assert broker.update_mark(BTC, Decimal("104")) == ()
    fills = broker.update_mark(BTC, Decimal("106"))
    assert len(fills) == 1
    assert fills[0].price == Decimal("106")


def test_sell_can_reduce_and_flip_position() -> None:
    broker = PaperBroker(fee_rate=Decimal("0"), slippage_bps=Decimal("0"))
    broker.update_mark(BTC, Decimal("100"))
    broker.submit(request("buy", OrderSide.BUY, "2"))
    broker.update_mark(BTC, Decimal("110"))
    broker.submit(request("sell", OrderSide.SELL, "3"))
    position = broker.account().positions[0]
    assert position.quantity == Decimal("-1")
    assert position.average_price == Decimal("110")


def test_cancel_pending_order() -> None:
    broker = PaperBroker()
    broker.update_mark(BTC, Decimal("100"))
    broker.submit(request("pending", OrderSide.BUY, order_type=OrderType.LIMIT, limit_price="90"))
    state = broker.cancel("pending")
    assert state.status is OrderStatus.CANCELLED
    assert broker.open_orders() == ()


def test_market_order_requires_mark() -> None:
    broker = PaperBroker()
    with pytest.raises(ValueError, match="current mark"):
        broker.submit(request("missing", OrderSide.BUY))


def test_bad_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        PaperBroker(initial_cash=Decimal("0"))
    with pytest.raises(ValueError):
        PaperBroker(fee_rate=Decimal("-1"))
    broker = PaperBroker()
    with pytest.raises(ValueError):
        broker.update_mark(BTC, Decimal("0"))
    with pytest.raises(ValueError, match="binary floats"):
        BrokerOrderRequest(
            client_order_id="float",
            instrument_id=BTC,
            side=OrderSide.BUY,
            quantity=1.2,
        )
