"""Canonical instrument identities shared across every market-data provider."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AssetClass(StrEnum):
    CRYPTO = "crypto"
    FOREX = "forex"
    EQUITY = "equity"
    ETF = "etf"
    FUTURE = "future"
    OPTION = "option"
    COMMODITY = "commodity"
    INDEX = "index"
    CFD = "cfd"
    BINARY = "binary"
    PREDICTION = "prediction"


class InstrumentType(StrEnum):
    SPOT = "spot"
    PERPETUAL = "perpetual"
    FUTURE = "future"
    OPTION = "option"
    CASH = "cash"
    CFD = "cfd"
    BINARY = "binary"


class InstrumentId(BaseModel):
    """Provider-independent instrument key.

    ``venue`` identifies the execution/data venue and ``symbol`` preserves the venue's
    canonical symbol. Keeping both prevents collisions such as BTCUSD existing on many venues.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    venue: str = Field(min_length=1, max_length=40)
    symbol: str = Field(min_length=1, max_length=80)

    @field_validator("venue", "symbol")
    @classmethod
    def normalize_token(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("instrument token cannot be blank")
        return normalized

    @property
    def key(self) -> str:
        return f"{self.venue}:{self.symbol}"


class Instrument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: InstrumentId
    asset_class: AssetClass
    instrument_type: InstrumentType
    base_currency: str = Field(min_length=3, max_length=12)
    quote_currency: str = Field(min_length=3, max_length=12)
    price_precision: int = Field(ge=0, le=18)
    quantity_precision: int = Field(ge=0, le=18)
    contract_multiplier: str = "1"
    active: bool = True

    @field_validator("base_currency", "quote_currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        return value.strip().upper()
