"""Corporate actions required for survivorship- and adjustment-aware equity research."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_os_core.instruments import InstrumentId
from quant_os_core.market_data.timeframes import ensure_aware


class CorporateActionType(StrEnum):
    CASH_DIVIDEND = "cash_dividend"
    STOCK_DIVIDEND = "stock_dividend"
    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    SYMBOL_CHANGE = "symbol_change"
    DELISTING = "delisting"
    SPINOFF = "spinoff"


class CorporateAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: InstrumentId
    action_type: CorporateActionType
    effective_at: datetime
    source: str = Field(min_length=1, max_length=100)
    cash_amount: Decimal | None = None
    currency: str | None = None
    ratio: Decimal | None = Field(default=None, gt=0)
    new_symbol: str | None = None

    @field_validator("effective_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return ensure_aware(value, field="effective_at")

    @model_validator(mode="after")
    def validate_payload(self) -> CorporateAction:
        if self.action_type is CorporateActionType.CASH_DIVIDEND and (
            self.cash_amount is None or self.currency is None
        ):
            raise ValueError("cash dividend requires cash_amount and currency")
        splits = {CorporateActionType.SPLIT, CorporateActionType.REVERSE_SPLIT}
        if self.action_type in splits and self.ratio is None:
            raise ValueError("split action requires ratio")
        if self.action_type is CorporateActionType.SYMBOL_CHANGE and not self.new_symbol:
            raise ValueError("symbol change requires new_symbol")
        return self
