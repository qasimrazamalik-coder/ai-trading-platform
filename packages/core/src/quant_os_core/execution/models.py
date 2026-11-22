"""Venue-independent execution and account state models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_os_core.backtesting.models import OrderSide, OrderStatus, OrderType, TimeInForce
from quant_os_core.instruments import InstrumentId


class ExecutionMode(StrEnum):
    PAPER = "paper"
    LIVE = "live"


class BrokerOrderRequest(BaseModel):
    """Idempotent order command sent to an execution adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_order_id: str = Field(min_length=1, max_length=128)
    instrument_id: InstrumentId
    side: OrderSide
    quantity: Decimal = Field(gt=0)
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.GTC
    limit_price: Decimal | None = Field(default=None, gt=0)
    stop_price: Decimal | None = Field(default=None, gt=0)

    @field_validator("quantity", "limit_price", "stop_price", mode="before")
    @classmethod
    def reject_float(cls, value: object) -> object:
        if isinstance(value, float):
            raise ValueError("financial values must not use binary floats")
        return value

    @model_validator(mode="after")
    def validate_prices(self) -> BrokerOrderRequest:
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit orders require limit_price")
        if self.order_type is OrderType.STOP and self.stop_price is None:
            raise ValueError("stop orders require stop_price")
        if self.order_type is not OrderType.LIMIT and self.limit_price is not None:
            raise ValueError("limit_price is only valid for limit orders")
        if self.order_type is not OrderType.STOP and self.stop_price is not None:
            raise ValueError("stop_price is only valid for stop orders")
        return self


class BrokerOrderState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    client_order_id: str
    venue_order_id: str
    instrument_id: InstrumentId
    status: OrderStatus
    submitted_quantity: Decimal
    filled_quantity: Decimal = Decimal("0")
    average_fill_price: Decimal | None = None
    updated_at: datetime

    @property
    def remaining_quantity(self) -> Decimal:
        return max(Decimal("0"), self.submitted_quantity - self.filled_quantity)


@dataclass(frozen=True, slots=True)
class ExecutionFill:
    fill_id: UUID
    client_order_id: str
    venue_order_id: str
    instrument_id: InstrumentId
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee: Decimal
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    instrument_id: InstrumentId
    quantity: Decimal
    average_price: Decimal
    mark_price: Decimal

    @property
    def notional(self) -> Decimal:
        return self.quantity * self.mark_price


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    cash: Decimal
    equity: Decimal
    buying_power: Decimal
    positions: tuple[BrokerPosition, ...]
    timestamp: datetime
