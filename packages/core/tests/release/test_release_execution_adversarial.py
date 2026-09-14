from __future__ import annotations

import random
from decimal import Decimal

import pytest
from pydantic import ValidationError

from quant_os_core.backtesting.models import OrderSide, OrderStatus, OrderType
from quant_os_core.execution import BrokerOrderRequest, PaperBroker
from quant_os_core.instruments import InstrumentId

BTC = InstrumentId(venue="PAPER", symbol="BTCUSDT")


def _request(
    client_id: str,
    side: OrderSide,
    quantity: Decimal,
    *,
    order_type: OrderType = OrderType.MARKET,
    limit_price: Decimal | None = None,
    stop_price: Decimal | None = None,
) -> BrokerOrderRequest:
    return BrokerOrderRequest(
        client_order_id=client_id,
        instrument_id=BTC,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
    )


def test_market_order_without_mark_is_atomic_and_retryable() -> None:
    broker = PaperBroker(fee_rate=Decimal("0"), slippage_bps=Decimal("0"))
    order = _request("atomic", OrderSide.BUY, Decimal("1"))

    with pytest.raises(ValueError, match="current mark"):
        broker.submit(order)

    assert broker.order("atomic") is None
    assert broker.open_orders() == ()
    assert broker.fills == ()

    broker.update_mark(BTC, Decimal("100"))
    state = broker.submit(order)
    assert state.status is OrderStatus.FILLED
    assert len(broker.fills) == 1


def test_execution_request_rejects_malformed_price_fields() -> None:
    with pytest.raises(ValidationError, match="limit orders require limit_price"):
        _request("bad-limit", OrderSide.BUY, Decimal("1"), order_type=OrderType.LIMIT)
    with pytest.raises(ValidationError, match="stop orders require stop_price"):
        _request("bad-stop", OrderSide.BUY, Decimal("1"), order_type=OrderType.STOP)
    with pytest.raises(ValidationError, match="only valid for limit orders"):
        _request(
            "market-limit",
            OrderSide.BUY,
            Decimal("1"),
            order_type=OrderType.MARKET,
            limit_price=Decimal("90"),
        )
    with pytest.raises(ValidationError, match="only valid for stop orders"):
        _request(
            "limit-stop",
            OrderSide.BUY,
            Decimal("1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("90"),
            stop_price=Decimal("80"),
        )


def test_randomized_market_order_accounting_and_idempotency() -> None:
    rng = random.Random(20260914)
    initial_cash = Decimal("100000")
    fee_rate = Decimal("0.0007")
    broker = PaperBroker(initial_cash=initial_cash, fee_rate=fee_rate, slippage_bps=Decimal("0"))

    expected_cash = initial_cash
    expected_quantity = Decimal("0")
    last_mark = Decimal("100")

    for index in range(500):
        last_mark = Decimal(rng.randrange(5000, 25001)) / Decimal("100")
        quantity = Decimal(rng.randrange(1, 41)) / Decimal("10")
        side = OrderSide.BUY if rng.randrange(2) == 0 else OrderSide.SELL
        broker.update_mark(BTC, last_mark)
        request = _request(f"rnd-{index}", side, quantity)
        state = broker.submit(request)
        assert state.status is OrderStatus.FILLED

        notional = quantity * last_mark
        fee = notional * fee_rate
        if side is OrderSide.BUY:
            expected_cash -= notional + fee
            expected_quantity += quantity
        else:
            expected_cash += notional - fee
            expected_quantity -= quantity

        before = len(broker.fills)
        assert broker.submit(request) == state
        assert len(broker.fills) == before

    account = broker.account()
    assert account.cash == expected_cash
    actual_quantity = sum((p.quantity for p in account.positions), Decimal("0"))
    assert actual_quantity == expected_quantity
    assert account.equity == expected_cash + expected_quantity * last_mark
    assert len({fill.fill_id for fill in broker.fills}) == len(broker.fills)
    assert len(broker.fills) == 500
