"""Canonical immutable market-data records.

Prices and quantities use ``Decimal`` to preserve exact values at ingestion boundaries.
Research code may later convert validated columns to vectorized floating-point arrays.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_os_core.instruments import InstrumentId
from quant_os_core.market_data.timeframes import Timeframe, ensure_aware


class MarketDataKind(StrEnum):
    CANDLE = "candle"
    TRADE = "trade"
    QUOTE = "quote"
    ORDER_BOOK = "order_book"


class MarketDataRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: InstrumentId
    timestamp: datetime
    source: str = Field(min_length=1, max_length=100)

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_aware(cls, value: datetime) -> datetime:
        return ensure_aware(value)


class Candle(MarketDataRecord):
    kind: MarketDataKind = MarketDataKind.CANDLE
    timeframe: str = Field(min_length=2, max_length=16)
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    volume: Decimal = Field(ge=0)
    quote_volume: Decimal | None = Field(default=None, ge=0)
    trade_count: int | None = Field(default=None, ge=0)
    close_time: datetime
    is_closed: bool = True

    @field_validator(
        "open", "high", "low", "close", "volume", "quote_volume", mode="before"
    )
    @classmethod
    def reject_binary_float(cls, value: object) -> object:
        if isinstance(value, float):
            raise ValueError("financial market-data values must not use binary floats")
        return value

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, value: str) -> str:
        Timeframe.parse(value)
        return value.strip().lower()

    @field_validator("close_time")
    @classmethod
    def close_time_must_be_aware(cls, value: datetime) -> datetime:
        return ensure_aware(value, field="close_time")

    @model_validator(mode="after")
    def validate_candle(self) -> "Candle":
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be >= open, close and low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be <= open, close and high")
        if self.close_time <= self.timestamp:
            raise ValueError("close_time must be after timestamp")
        return self


class TradeSide(StrEnum):
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class Trade(MarketDataRecord):
    kind: MarketDataKind = MarketDataKind.TRADE
    trade_id: str = Field(min_length=1, max_length=160)
    price: Decimal = Field(gt=0)
    quantity: Decimal = Field(gt=0)
    taker_side: TradeSide = TradeSide.UNKNOWN

    @field_validator("price", "quantity", mode="before")
    @classmethod
    def reject_binary_float(cls, value: object) -> object:
        if isinstance(value, float):
            raise ValueError("financial market-data values must not use binary floats")
        return value


class Quote(MarketDataRecord):
    kind: MarketDataKind = MarketDataKind.QUOTE
    bid_price: Decimal = Field(gt=0)
    ask_price: Decimal = Field(gt=0)
    bid_size: Decimal = Field(ge=0)
    ask_size: Decimal = Field(ge=0)

    @field_validator("bid_price", "ask_price", "bid_size", "ask_size", mode="before")
    @classmethod
    def reject_binary_float(cls, value: object) -> object:
        if isinstance(value, float):
            raise ValueError("financial market-data values must not use binary floats")
        return value

    @model_validator(mode="after")
    def validate_quote(self) -> "Quote":
        if self.bid_price > self.ask_price:
            raise ValueError("bid_price cannot exceed ask_price")
        return self

    @property
    def midpoint(self) -> Decimal:
        return (self.bid_price + self.ask_price) / Decimal(2)

    @property
    def spread(self) -> Decimal:
        return self.ask_price - self.bid_price


class BookLevel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    price: Decimal = Field(gt=0)
    quantity: Decimal = Field(gt=0)

    @field_validator("price", "quantity", mode="before")
    @classmethod
    def reject_binary_float(cls, value: object) -> object:
        if isinstance(value, float):
            raise ValueError("financial market-data values must not use binary floats")
        return value


class OrderBookSnapshot(MarketDataRecord):
    kind: MarketDataKind = MarketDataKind.ORDER_BOOK
    sequence: int | None = Field(default=None, ge=0)
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]

    @model_validator(mode="after")
    def validate_book(self) -> "OrderBookSnapshot":
        if not self.bids or not self.asks:
            raise ValueError("order book must contain at least one bid and ask")
        if any(a.price < b.price for a, b in zip(self.bids, self.bids[1:])):
            raise ValueError("bids must be sorted descending by price")
        if any(a.price > b.price for a, b in zip(self.asks, self.asks[1:])):
            raise ValueError("asks must be sorted ascending by price")
        if self.bids[0].price > self.asks[0].price:
            raise ValueError("crossed order book: best bid exceeds best ask")
        return self

    @property
    def best_bid(self) -> BookLevel:
        return self.bids[0]

    @property
    def best_ask(self) -> BookLevel:
        return self.asks[0]
