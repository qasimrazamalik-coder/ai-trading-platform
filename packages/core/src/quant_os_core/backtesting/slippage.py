"""Spread, slippage and simple market-impact models."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from quant_os_core.backtesting.models import OrderSide


class SlippageModel(Protocol):
    def adjust(
        self,
        *,
        reference_price: Decimal,
        side: OrderSide,
        quantity: Decimal,
        bar_volume: Decimal,
    ) -> Decimal: ...


@dataclass(frozen=True, slots=True)
class BpsSlippageModel:
    """Adverse fixed spread/slippage plus optional square-root participation impact."""

    fixed_bps: Decimal = Decimal("0")
    half_spread_bps: Decimal = Decimal("0")
    impact_bps_at_full_volume: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if (
            self.fixed_bps < 0
            or self.half_spread_bps < 0
            or self.impact_bps_at_full_volume < 0
        ):
            raise ValueError("slippage parameters cannot be negative")

    def adjust(
        self,
        *,
        reference_price: Decimal,
        side: OrderSide,
        quantity: Decimal,
        bar_volume: Decimal,
    ) -> Decimal:
        if reference_price <= 0 or quantity <= 0:
            raise ValueError("reference_price and quantity must be > 0")
        participation = Decimal("0")
        if bar_volume > 0:
            participation = min(quantity / bar_volume, Decimal("1"))
        impact = self.impact_bps_at_full_volume * participation.sqrt()
        total_bps = self.half_spread_bps + self.fixed_bps + impact
        multiplier = Decimal("1") + side.sign * total_bps / Decimal("10000")
        adjusted = reference_price * multiplier
        if adjusted <= 0:
            raise ValueError("slippage produced non-positive fill price")
        return adjusted
