"""Fee and financing models for deterministic backtests."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol


class FeeModel(Protocol):
    def calculate(self, *, notional: Decimal, is_maker: bool = False) -> Decimal: ...


@dataclass(frozen=True, slots=True)
class ProportionalFeeModel:
    """Percentage fee model expressed in basis points."""

    taker_bps: Decimal = Decimal("0")
    maker_bps: Decimal | None = None
    minimum_fee: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if self.taker_bps < 0 or self.minimum_fee < 0:
            raise ValueError("fees cannot be negative")
        if self.maker_bps is not None and self.maker_bps < 0:
            raise ValueError("maker_bps cannot be negative")

    def calculate(self, *, notional: Decimal, is_maker: bool = False) -> Decimal:
        if notional < 0:
            raise ValueError("notional cannot be negative")
        bps = self.maker_bps if is_maker and self.maker_bps is not None else self.taker_bps
        fee = notional * bps / Decimal("10000")
        return max(fee, self.minimum_fee) if notional > 0 else Decimal("0")


@dataclass(frozen=True, slots=True)
class FinancingModel:
    """Simple annualized financing model for borrow/funding simulations."""

    annual_long_rate: Decimal = Decimal("0")
    annual_short_rate: Decimal = Decimal("0")
    periods_per_year: Decimal = Decimal("365")

    def __post_init__(self) -> None:
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be > 0")

    def charge(self, *, signed_notional: Decimal) -> Decimal:
        if signed_notional == 0:
            return Decimal("0")
        rate = self.annual_long_rate if signed_notional > 0 else self.annual_short_rate
        return abs(signed_notional) * rate / self.periods_per_year
