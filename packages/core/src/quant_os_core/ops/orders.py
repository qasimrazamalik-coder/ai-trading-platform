"""Order construction: patient limit entries, no chasing, deterministic client order ids."""

from __future__ import annotations

import re
from datetime import datetime
from decimal import ROUND_DOWN, Decimal

from quant_os_core.agents.trade_plan import TradePlan
from quant_os_core.backtesting.models import OrderSide, OrderType
from quant_os_core.execution.models import BrokerOrderRequest
from quant_os_core.instruments import InstrumentId

_SAFE = re.compile(r"[^A-Za-z0-9]")


def client_order_id(symbol: str, side: OrderSide, when: datetime) -> str:
    """Same symbol, side and minute -> same id, so a retried submit cannot double-fill."""

    return f"qos-{_SAFE.sub('', symbol)}-{side.value}-{when.strftime('%Y%m%d%H%M')}"


def tick(price: Decimal) -> Decimal:
    return Decimal("0.01") if price >= 1 else Decimal("0.0001")


def entry_order(
    plan: TradePlan,
    instrument_id: InstrumentId,
    quantity: Decimal,
    last_price: Decimal,
    when: datetime,
    *,
    max_chase: Decimal = Decimal("0.002"),
) -> BrokerOrderRequest | None:
    """Limit buy at the better of the plan's entry and the last price, never chasing.

    Returns ``None`` when price has already run more than ``max_chase`` above the planned
    entry: the reward-to-risk the plan was approved on no longer holds.
    """

    if quantity <= 0 or last_price <= 0:
        return None
    ceiling = plan.entry_price * (Decimal("1") + max_chase)
    if last_price > ceiling:
        return None
    limit = min(last_price, ceiling).quantize(tick(last_price), rounding=ROUND_DOWN)
    if limit <= plan.stop_loss:
        return None
    return BrokerOrderRequest(
        client_order_id=client_order_id(plan.symbol, OrderSide.BUY, when),
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        limit_price=limit,
    )


def exit_order(
    symbol: str, instrument_id: InstrumentId, quantity: Decimal, when: datetime
) -> BrokerOrderRequest:
    """Exits are market orders: getting out matters more than the last tick."""

    return BrokerOrderRequest(
        client_order_id=client_order_id(symbol, OrderSide.SELL, when),
        instrument_id=instrument_id,
        side=OrderSide.SELL,
        quantity=quantity,
        order_type=OrderType.MARKET,
    )
