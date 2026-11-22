"""Deterministic candle-based simulated broker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from quant_os_core.backtesting.costs import FeeModel, ProportionalFeeModel
from quant_os_core.backtesting.models import (
    Fill,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    SimulatedOrder,
    TimeInForce,
)
from quant_os_core.backtesting.slippage import BpsSlippageModel, SlippageModel
from quant_os_core.instruments import InstrumentId
from quant_os_core.market_data.models import Candle


@dataclass(frozen=True, slots=True)
class BrokerConfig:
    latency_bars: int = 1
    max_bar_volume_fraction: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        if self.latency_bars < 1:
            raise ValueError("latency_bars must be >= 1 to prevent same-bar look-ahead")
        if not Decimal("0") < self.max_bar_volume_fraction <= Decimal("1"):
            raise ValueError("max_bar_volume_fraction must be in (0, 1]")


class SimulatedBroker:
    def __init__(
        self,
        *,
        config: BrokerConfig | None = None,
        fee_model: FeeModel | None = None,
        slippage_model: SlippageModel | None = None,
    ) -> None:
        self.config = config or BrokerConfig()
        self.fee_model = fee_model or ProportionalFeeModel()
        self.slippage_model = slippage_model or BpsSlippageModel()
        self.orders: list[SimulatedOrder] = []
        self.fills: list[Fill] = []
        self.rejected_orders = 0
        self.cancelled_orders = 0

    def reset(self) -> None:
        self.orders.clear()
        self.fills.clear()
        self.rejected_orders = 0
        self.cancelled_orders = 0

    def cancel_open_orders(self) -> int:
        cancelled = 0
        for order in self.orders:
            if order.status in {OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED}:
                order.status = OrderStatus.CANCELLED
                self.cancelled_orders += 1
                cancelled += 1
        return cancelled


    def pending_signed_quantity(self, instrument_id: InstrumentId) -> Decimal:
        """Return signed unfilled quantity for one instrument across open orders."""
        total = Decimal("0")
        for order in self.orders:
            if order.status not in {OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED}:
                continue
            if order.request.instrument_id != instrument_id:
                continue
            total += order.remaining_quantity * order.request.side.sign
        return total

    def submit(
        self, request: OrderRequest, *, submitted_at: datetime, bar_index: int
    ) -> SimulatedOrder:
        order = SimulatedOrder(
            request=request,
            submitted_at=submitted_at,
            eligible_bar_index=bar_index + self.config.latency_bars,
            remaining_quantity=request.quantity,
        )
        self.orders.append(order)
        return order

    def process_bar(self, candle: Candle, *, bar_index: int) -> tuple[Fill, ...]:
        fills: list[Fill] = []
        available_volume = candle.volume * self.config.max_bar_volume_fraction

        for order in self.orders:
            if order.status not in {OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED}:
                continue
            if order.request.instrument_id != candle.instrument_id:
                continue
            if bar_index < order.eligible_bar_index:
                continue
            if candle.timestamp < order.submitted_at:
                continue

            reference = self._trigger_price(order.request, candle)
            if reference is None:
                if order.request.time_in_force is TimeInForce.IOC:
                    order.status = OrderStatus.CANCELLED
                    self.cancelled_orders += 1
                continue

            fill_qty = min(order.remaining_quantity, available_volume)
            if fill_qty <= 0:
                if order.request.time_in_force is TimeInForce.IOC:
                    order.status = OrderStatus.CANCELLED
                    self.cancelled_orders += 1
                continue

            fill_price = self.slippage_model.adjust(
                reference_price=reference,
                side=order.request.side,
                quantity=fill_qty,
                bar_volume=candle.volume,
            )
            if order.request.order_type is OrderType.LIMIT:
                assert order.request.limit_price is not None
                if order.request.side is OrderSide.BUY:
                    fill_price = min(fill_price, order.request.limit_price)
                else:
                    fill_price = max(fill_price, order.request.limit_price)
            is_maker = order.request.order_type is OrderType.LIMIT
            fee = self.fee_model.calculate(notional=fill_qty * fill_price, is_maker=is_maker)
            fill = Fill(
                order_id=order.order_id,
                instrument_id=order.request.instrument_id,
                side=order.request.side,
                quantity=fill_qty,
                price=fill_price,
                fee=fee,
                timestamp=candle.timestamp,
                bar_index=bar_index,
            )
            fills.append(fill)
            self.fills.append(fill)
            available_volume -= fill_qty
            order.remaining_quantity -= fill_qty

            if order.remaining_quantity == 0:
                order.status = OrderStatus.FILLED
            elif order.request.time_in_force is TimeInForce.IOC:
                order.status = OrderStatus.CANCELLED
                self.cancelled_orders += 1
            else:
                order.status = OrderStatus.PARTIALLY_FILLED

        return tuple(fills)

    @staticmethod
    def _trigger_price(request: OrderRequest, candle: Candle) -> Decimal | None:
        if request.order_type is OrderType.MARKET:
            return candle.open

        if request.order_type is OrderType.LIMIT:
            assert request.limit_price is not None
            if request.side is OrderSide.BUY and candle.low <= request.limit_price:
                return min(candle.open, request.limit_price)
            if request.side is OrderSide.SELL and candle.high >= request.limit_price:
                return max(candle.open, request.limit_price)
            return None

        assert request.stop_price is not None
        if request.side is OrderSide.BUY and candle.high >= request.stop_price:
            return max(candle.open, request.stop_price)
        if request.side is OrderSide.SELL and candle.low <= request.stop_price:
            return min(candle.open, request.stop_price)
        return None
