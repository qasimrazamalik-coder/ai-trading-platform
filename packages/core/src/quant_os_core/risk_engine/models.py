"""Typed inputs and outputs for independent portfolio risk approval."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

import numpy as np

from quant_os_core.events import OrderIntent


class RiskAction(StrEnum):
    APPROVE = "approve"
    RESIZE = "resize"
    REJECT = "reject"
    HALT = "halt"


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_position_fraction: Decimal = Decimal("0.10")
    max_gross_leverage: Decimal = Decimal("1.50")
    max_net_leverage: Decimal = Decimal("1.00")
    max_daily_loss_fraction: Decimal = Decimal("0.03")
    max_drawdown_fraction: Decimal = Decimal("0.10")
    max_portfolio_heat: Decimal = Decimal("0.06")
    max_group_fraction: Decimal = Decimal("0.35")
    max_correlated_fraction: Decimal = Decimal("0.50")
    correlation_threshold: float = 0.80
    min_equity: Decimal = Decimal("1")
    allow_short: bool = True

    def __post_init__(self) -> None:
        fraction_fields = (
            self.max_position_fraction,
            self.max_daily_loss_fraction,
            self.max_drawdown_fraction,
            self.max_portfolio_heat,
            self.max_group_fraction,
            self.max_correlated_fraction,
        )
        if any(value <= 0 or value > 1 for value in fraction_fields):
            raise ValueError("fractional risk limits must be in (0, 1]")
        if self.max_gross_leverage <= 0 or self.max_net_leverage <= 0:
            raise ValueError("leverage limits must be positive")
        if self.min_equity <= 0:
            raise ValueError("min_equity must be positive")
        if not np.isfinite(self.correlation_threshold):
            raise ValueError("correlation_threshold must be finite")
        if not -1.0 <= self.correlation_threshold <= 1.0:
            raise ValueError("correlation_threshold must be in [-1, 1]")


@dataclass(frozen=True, slots=True)
class RiskPosition:
    symbol: str
    quantity: Decimal
    mark_price: Decimal
    stop_price: Decimal | None = None
    group: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol cannot be blank")
        if self.mark_price <= 0:
            raise ValueError("mark_price must be positive")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("stop_price must be positive when provided")

    @property
    def notional(self) -> Decimal:
        return self.quantity * self.mark_price

    @property
    def absolute_notional(self) -> Decimal:
        return abs(self.notional)

    @property
    def heat(self) -> Decimal:
        if self.stop_price is None or self.quantity == 0:
            return Decimal("0")
        return abs(self.quantity) * abs(self.mark_price - self.stop_price)


@dataclass(frozen=True, slots=True)
class RiskContext:
    equity: Decimal
    peak_equity: Decimal
    start_of_day_equity: Decimal
    positions: tuple[RiskPosition, ...]
    mark_prices: dict[str, Decimal]
    stop_prices: dict[str, Decimal] = field(default_factory=dict)
    symbol_groups: dict[str, str] = field(default_factory=dict)
    correlations: dict[tuple[str, str], float] = field(default_factory=dict)
    kill_switch: bool = False
    stale_market_data: bool = False

    def __post_init__(self) -> None:
        if self.equity <= 0 or self.peak_equity <= 0 or self.start_of_day_equity <= 0:
            raise ValueError("equity values must be positive")
        for symbol, price in self.mark_prices.items():
            if not symbol.strip() or price <= 0:
                raise ValueError("mark prices require nonblank symbols and positive prices")

    @property
    def drawdown_fraction(self) -> Decimal:
        return max(Decimal("0"), (self.peak_equity - self.equity) / self.peak_equity)

    @property
    def daily_loss_fraction(self) -> Decimal:
        return max(
            Decimal("0"),
            (self.start_of_day_equity - self.equity) / self.start_of_day_equity,
        )

    @property
    def gross_exposure(self) -> Decimal:
        return sum((position.absolute_notional for position in self.positions), Decimal("0"))

    @property
    def net_exposure(self) -> Decimal:
        return sum((position.notional for position in self.positions), Decimal("0"))

    @property
    def portfolio_heat(self) -> Decimal:
        return sum((position.heat for position in self.positions), Decimal("0"))


@dataclass(frozen=True, slots=True)
class RiskEvaluation:
    action: RiskAction
    approved_quantity: Decimal
    reasons: tuple[str, ...]
    order_intent: OrderIntent
    projected_gross_leverage: Decimal
    projected_net_leverage: Decimal
    projected_heat_fraction: Decimal

    @property
    def approved(self) -> bool:
        return self.action in {RiskAction.APPROVE, RiskAction.RESIZE}

    @property
    def approved_intent(self) -> OrderIntent | None:
        if not self.approved or self.approved_quantity <= 0:
            return None
        return self.order_intent.model_copy(update={"quantity": self.approved_quantity})
