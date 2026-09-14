"""Typed domain events shared by research, risk, execution and adapters."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EventType(StrEnum):
    BAR = "bar"
    ORDER_INTENT = "order_intent"
    RISK_DECISION = "risk_decision"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str = Field(min_length=1, max_length=100)


class BarEvent(Event):
    event_type: EventType = EventType.BAR
    symbol: str = Field(min_length=1, max_length=64)
    timeframe: str = Field(min_length=1, max_length=16)
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    volume: Decimal = Field(ge=0)

    @field_validator("open", "high", "low", "close", "volume", mode="before")
    @classmethod
    def reject_binary_float(cls, value: Any) -> Any:
        if isinstance(value, float):
            raise ValueError("financial event values must use Decimal, int, or string")
        return value

    @model_validator(mode="after")
    def validate_ohlc(self) -> BarEvent:
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be >= open, close and low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be <= open, close and high")
        return self


class OrderIntent(Event):
    event_type: EventType = EventType.ORDER_INTENT
    strategy_id: str = Field(min_length=1, max_length=100)
    symbol: str = Field(min_length=1, max_length=64)
    side: Side
    quantity: Decimal = Field(gt=0)
    limit_price: Decimal | None = Field(default=None, gt=0)

    @field_validator("quantity", "limit_price", mode="before")
    @classmethod
    def reject_binary_float(cls, value: Any) -> Any:
        if isinstance(value, float):
            raise ValueError("financial event values must use Decimal, int, string, or null")
        return value


class RiskDecision(Event):
    event_type: EventType = EventType.RISK_DECISION
    order_intent_id: UUID
    approved: bool
    reason: str = Field(min_length=1, max_length=500)
