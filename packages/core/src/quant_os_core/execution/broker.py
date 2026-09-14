"""Broker protocol and deterministic paper-broker implementation."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol
from uuid import uuid4

from quant_os_core.backtesting.models import OrderSide, OrderStatus, OrderType
from quant_os_core.execution.models import (
    AccountSnapshot,
    BrokerOrderRequest,
    BrokerOrderState,
    BrokerPosition,
    ExecutionFill,
)
from quant_os_core.instruments import InstrumentId


class ExecutionBroker(Protocol):
    def submit(self, request: BrokerOrderRequest) -> BrokerOrderState: ...
    def cancel(self, client_order_id: str) -> BrokerOrderState: ...
    def order(self, client_order_id: str) -> BrokerOrderState | None: ...
    def open_orders(self) -> tuple[BrokerOrderState, ...]: ...
    def account(self) -> AccountSnapshot: ...


class PaperBroker:
    """Simple mark-driven paper execution engine.

    Market orders fill immediately at the current mark plus configured adverse slippage.
    Limit/stop orders remain pending until a later mark makes them executable. All money
    values remain Decimal at the execution boundary.
    """

    def __init__(
        self,
        *,
        initial_cash: Decimal = Decimal("100000"),
        fee_rate: Decimal = Decimal("0.0005"),
        slippage_bps: Decimal = Decimal("1"),
    ) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if fee_rate < 0 or slippage_bps < 0:
            raise ValueError("fees and slippage cannot be negative")
        self._cash = initial_cash
        self._fee_rate = fee_rate
        self._slippage_bps = slippage_bps
        self._marks: dict[str, Decimal] = {}
        self._orders: dict[str, BrokerOrderState] = {}
        self._requests: dict[str, BrokerOrderRequest] = {}
        self._positions: dict[str, tuple[Decimal, Decimal, InstrumentId]] = {}
        self._fills: list[ExecutionFill] = []

    @property
    def fills(self) -> tuple[ExecutionFill, ...]:
        return tuple(self._fills)

    def update_mark(
        self,
        instrument_id: InstrumentId,
        price: Decimal,
        *,
        timestamp: datetime | None = None,
    ) -> tuple[ExecutionFill, ...]:
        if price <= 0:
            raise ValueError("mark price must be positive")
        self._marks[instrument_id.key] = price
        before = len(self._fills)
        now = timestamp or datetime.now(UTC)
        for client_order_id in tuple(self._orders):
            state = self._orders[client_order_id]
            if state.status not in {OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED}:
                continue
            request = self._requests[client_order_id]
            if request.instrument_id != instrument_id:
                continue
            if self._triggered(request, price):
                self._fill(request, state, price, now)
        return tuple(self._fills[before:])

    def submit(self, request: BrokerOrderRequest) -> BrokerOrderState:
        existing = self._orders.get(request.client_order_id)
        if existing is not None:
            if self._requests[request.client_order_id] != request:
                raise ValueError("client_order_id already exists with a different request")
            return existing
        mark = self._marks.get(request.instrument_id.key)
        if request.order_type is OrderType.MARKET and mark is None:
            raise ValueError("paper market order requires a current mark")
        now = datetime.now(UTC)
        state = BrokerOrderState(
            client_order_id=request.client_order_id,
            venue_order_id=f"PAPER-{uuid4().hex[:16]}",
            instrument_id=request.instrument_id,
            status=OrderStatus.PENDING,
            submitted_quantity=request.quantity,
            updated_at=now,
        )
        self._orders[request.client_order_id] = state
        self._requests[request.client_order_id] = request
        if request.order_type is OrderType.MARKET:
            assert mark is not None
            return self._fill(request, state, mark, now)
        if mark is not None and self._triggered(request, mark):
            return self._fill(request, state, mark, now)
        return state

    def cancel(self, client_order_id: str) -> BrokerOrderState:
        state = self._require_order(client_order_id)
        if state.status in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}:
            return state
        cancelled = state.model_copy(
            update={"status": OrderStatus.CANCELLED, "updated_at": datetime.now(UTC)}
        )
        self._orders[client_order_id] = cancelled
        return cancelled

    def order(self, client_order_id: str) -> BrokerOrderState | None:
        return self._orders.get(client_order_id)

    def open_orders(self) -> tuple[BrokerOrderState, ...]:
        return tuple(
            state
            for state in self._orders.values()
            if state.status in {OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED}
        )

    def orders(self) -> tuple[BrokerOrderState, ...]:
        """Return a snapshot of every order known to the paper broker."""
        return tuple(self._orders.values())

    def account(self) -> AccountSnapshot:
        positions: list[BrokerPosition] = []
        market_value = Decimal("0")
        for key, (quantity, average_price, instrument_id) in self._positions.items():
            mark = self._marks.get(key, average_price)
            position = BrokerPosition(instrument_id, quantity, average_price, mark)
            positions.append(position)
            market_value += position.notional
        equity = self._cash + market_value
        return AccountSnapshot(
            cash=self._cash,
            equity=equity,
            buying_power=max(Decimal("0"), equity),
            positions=tuple(positions),
            timestamp=datetime.now(UTC),
        )

    def _require_order(self, client_order_id: str) -> BrokerOrderState:
        state = self._orders.get(client_order_id)
        if state is None:
            raise KeyError(client_order_id)
        return state

    @staticmethod
    def _triggered(request: BrokerOrderRequest, mark: Decimal) -> bool:
        if request.order_type is OrderType.MARKET:
            return True
        if request.order_type is OrderType.LIMIT:
            assert request.limit_price is not None
            if request.side is OrderSide.BUY:
                return mark <= request.limit_price
            return mark >= request.limit_price
        assert request.stop_price is not None
        if request.side is OrderSide.BUY:
            return mark >= request.stop_price
        return mark <= request.stop_price

    def _execution_price(self, request: BrokerOrderRequest, mark: Decimal) -> Decimal:
        slip = self._slippage_bps / Decimal("10000")
        if request.side is OrderSide.BUY:
            execution = mark * (Decimal("1") + slip)
            if request.order_type is OrderType.LIMIT and request.limit_price is not None:
                return min(execution, request.limit_price)
            return execution
        execution = mark * (Decimal("1") - slip)
        if request.order_type is OrderType.LIMIT and request.limit_price is not None:
            return max(execution, request.limit_price)
        return execution

    def _fill(
        self,
        request: BrokerOrderRequest,
        state: BrokerOrderState,
        mark: Decimal,
        timestamp: datetime,
    ) -> BrokerOrderState:
        price = self._execution_price(request, mark)
        quantity = state.remaining_quantity
        notional = quantity * price
        fee = abs(notional) * self._fee_rate
        signed_quantity = quantity if request.side is OrderSide.BUY else -quantity
        if request.side is OrderSide.BUY:
            self._cash -= notional + fee
        else:
            self._cash += notional - fee
        self._apply_position(request.instrument_id, signed_quantity, price)
        filled = state.model_copy(
            update={
                "status": OrderStatus.FILLED,
                "filled_quantity": state.submitted_quantity,
                "average_fill_price": price,
                "updated_at": timestamp,
            }
        )
        self._orders[request.client_order_id] = filled
        self._fills.append(
            ExecutionFill(
                fill_id=uuid4(),
                client_order_id=request.client_order_id,
                venue_order_id=filled.venue_order_id,
                instrument_id=request.instrument_id,
                side=request.side,
                quantity=quantity,
                price=price,
                fee=fee,
                timestamp=timestamp,
            )
        )
        return filled

    def _apply_position(
        self,
        instrument_id: InstrumentId,
        delta: Decimal,
        price: Decimal,
    ) -> None:
        key = instrument_id.key
        old_quantity, old_average, _ = self._positions.get(
            key, (Decimal("0"), Decimal("0"), instrument_id)
        )
        new_quantity = old_quantity + delta
        if new_quantity == 0:
            self._positions.pop(key, None)
            return
        if old_quantity == 0 or (old_quantity > 0) == (delta > 0):
            gross_quantity = abs(old_quantity) + abs(delta)
            average = (
                abs(old_quantity) * old_average + abs(delta) * price
            ) / gross_quantity
        elif abs(delta) > abs(old_quantity):
            average = price
        else:
            average = old_average
        self._positions[key] = (new_quantity, average, instrument_id)
