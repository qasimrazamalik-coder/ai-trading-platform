"""Canonical order, fill, position and backtest result models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_os_core.instruments import InstrumentId


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> Decimal:
        return Decimal("1") if self is OrderSide.BUY else Decimal("-1")


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class TimeInForce(StrEnum):
    GTC = "gtc"
    IOC = "ioc"


class OrderStatus(StrEnum):
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderRequest(BaseModel):
    """Strategy-produced order request.

    Orders are intentionally venue-independent. The simulated broker decides when and how
    they fill. Binary floats are rejected at the financial boundary.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: InstrumentId
    side: OrderSide
    quantity: Decimal = Field(gt=0)
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = Field(default=None, gt=0)
    stop_price: Decimal | None = Field(default=None, gt=0)
    time_in_force: TimeInForce = TimeInForce.GTC
    tag: str | None = Field(default=None, max_length=100)

    @field_validator("quantity", "limit_price", "stop_price", mode="before")
    @classmethod
    def reject_binary_float(cls, value: object) -> object:
        if isinstance(value, float):
            raise ValueError("financial order values must use Decimal, int, string, or null")
        return value

    @model_validator(mode="after")
    def validate_prices(self) -> OrderRequest:
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit orders require limit_price")
        if self.order_type is OrderType.STOP and self.stop_price is None:
            raise ValueError("stop orders require stop_price")
        if self.order_type is not OrderType.LIMIT and self.limit_price is not None:
            raise ValueError("limit_price is only valid for limit orders")
        if self.order_type is not OrderType.STOP and self.stop_price is not None:
            raise ValueError("stop_price is only valid for stop orders")
        return self


class SimulatedOrder(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")

    order_id: UUID = Field(default_factory=uuid4)
    request: OrderRequest
    submitted_at: datetime
    eligible_bar_index: int = Field(ge=0)
    remaining_quantity: Decimal = Field(gt=0)
    status: OrderStatus = OrderStatus.PENDING

    @property
    def filled_quantity(self) -> Decimal:
        return self.request.quantity - self.remaining_quantity


class Fill(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fill_id: UUID = Field(default_factory=uuid4)
    order_id: UUID
    instrument_id: InstrumentId
    side: OrderSide
    quantity: Decimal = Field(gt=0)
    price: Decimal = Field(gt=0)
    fee: Decimal = Field(ge=0)
    timestamp: datetime
    bar_index: int = Field(ge=0)

    @property
    def signed_quantity(self) -> Decimal:
        return self.quantity * self.side.sign

    @property
    def notional(self) -> Decimal:
        return self.quantity * self.price


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    instrument_id: InstrumentId
    quantity: Decimal
    average_price: Decimal
    realized_pnl: Decimal
    market_price: Decimal | None
    unrealized_pnl: Decimal


@dataclass(frozen=True, slots=True)
class EquityPoint:
    timestamp: datetime
    cash: Decimal
    equity: Decimal
    gross_exposure: Decimal
    net_exposure: Decimal


@dataclass(frozen=True, slots=True)
class BacktestResult:
    initial_cash: Decimal
    final_equity: Decimal
    equity_curve: tuple[EquityPoint, ...]
    fills: tuple[Fill, ...]
    submitted_orders: int
    rejected_orders: int
    cancelled_orders: int

    @property
    def total_return(self) -> Decimal:
        if self.initial_cash == 0:
            return Decimal("0")
        return self.final_equity / self.initial_cash - Decimal("1")
