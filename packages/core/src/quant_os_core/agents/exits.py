"""Exit manager: enforces the exit plan registered with every trade.

A :class:`~quant_os_core.agents.trade_plan.TradePlan` carries a stop loss, a profit target and
a written invalidation condition. Free text cannot be evaluated by code, so each open position
also carries machine-checkable :class:`InvalidationRule` objects; the text stays for humans and
the audit log. Exit priority on a bar: stop (worst case first), invalidation, target, time stop,
day-trade flatten, then the trailing-stop update. A strict-profile position cannot exit at
all before settlement. Stops use the bar's low and targets its high, so results are conservative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from quant_os_core.agents.trade_plan import TradePlan
from quant_os_core.risk_engine import ClosedTrade, ExitReason


class RuleKind(StrEnum):
    CLOSE_BELOW = "close_below"  # close < level
    CLOSE_BELOW_SMA = "close_below_sma"  # close < simple moving average of `window` closes


@dataclass(frozen=True, slots=True)
class InvalidationRule:
    kind: RuleKind
    level: Decimal | None = None
    window: int | None = None

    def __post_init__(self) -> None:
        if self.kind is RuleKind.CLOSE_BELOW and (self.level is None or self.level <= 0):
            raise ValueError("close_below needs a positive level")
        if self.kind is RuleKind.CLOSE_BELOW_SMA and (self.window is None or self.window < 2):
            raise ValueError("close_below_sma needs a window of at least 2")

    def triggered(self, closes: tuple[Decimal, ...]) -> bool:
        if not closes:
            return False
        last = closes[-1]
        if self.kind is RuleKind.CLOSE_BELOW:
            assert self.level is not None
            return last < self.level
        assert self.window is not None
        if len(closes) < self.window:
            return False
        sma = sum(closes[-self.window :], Decimal("0")) / Decimal(self.window)
        return last < sma


@dataclass(frozen=True, slots=True)
class Bar:
    time: datetime
    high: Decimal
    low: Decimal
    close: Decimal

    def __post_init__(self) -> None:
        if not (self.low <= self.close <= self.high) or self.low <= 0:
            raise ValueError("bar requires 0 < low <= close <= high")


@dataclass(slots=True)
class OpenPosition:
    plan: TradePlan
    quantity: Decimal
    entry_time: datetime
    rules: tuple[InvalidationRule, ...] = ()
    trail_fraction: Decimal | None = None  # e.g. 0.08 trails 8% below the highest high
    max_holding: timedelta | None = None
    flatten_at: datetime | None = None  # day trading: close at this time whatever happens
    earliest_exit: datetime | None = None  # strict profile: no sale before settlement
    highest: Decimal = field(init=False)
    stop: Decimal = field(init=False)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.trail_fraction is not None and not Decimal("0") < self.trail_fraction < Decimal(
            "1"
        ):
            raise ValueError("trail_fraction must be in (0, 1)")
        self.highest = self.plan.entry_price
        self.stop = self.plan.stop_loss


@dataclass(frozen=True, slots=True)
class ExitEvent:
    reason: ExitReason
    price: Decimal
    trade: ClosedTrade


class ExitManager:
    def on_bar(
        self, position: OpenPosition, bar: Bar, closes: tuple[Decimal, ...]
    ) -> ExitEvent | None:
        plan = position.plan
        if position.earliest_exit is not None and bar.time < position.earliest_exit:
            return None
        if bar.low <= position.stop:
            reason = (
                ExitReason.STOP_LOSS
                if position.stop <= plan.entry_price
                else ExitReason.TAKE_PROFIT
            )
            return self._exit(position, bar.time, min(position.stop, bar.high), reason)
        if any(rule.triggered(closes) for rule in position.rules):
            return self._exit(position, bar.time, bar.close, ExitReason.INVALIDATION)
        if bar.high >= plan.profit_target:
            return self._exit(position, bar.time, plan.profit_target, ExitReason.TAKE_PROFIT)
        if (
            position.max_holding is not None
            and bar.time - position.entry_time >= position.max_holding
        ):
            return self._exit(position, bar.time, bar.close, ExitReason.TIME)
        if position.flatten_at is not None and bar.time >= position.flatten_at:
            return self._exit(position, bar.time, bar.close, ExitReason.TIME)
        if position.trail_fraction is not None and bar.high > position.highest:
            position.highest = bar.high
            trailed = position.highest * (Decimal("1") - position.trail_fraction)
            position.stop = max(position.stop, trailed)  # stops only ever move up
        return None

    @staticmethod
    def _exit(
        position: OpenPosition, when: datetime, price: Decimal, reason: ExitReason
    ) -> ExitEvent:
        profit = price / position.plan.entry_price - Decimal("1")
        trade = ClosedTrade(position.plan.symbol, when, profit, reason)
        return ExitEvent(reason, price, trade)
