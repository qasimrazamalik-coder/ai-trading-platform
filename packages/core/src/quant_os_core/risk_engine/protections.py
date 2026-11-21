"""Trading protections modelled on Freqtrade's built-in Protections.

Freqtrade documents four: StoplossGuard, MaxDrawdown, LowProfitPairs and CooldownPeriod
(https://www.freqtrade.io/en/stable/plugins/). They act on *closed trades* and return
time-bounded locks, either for one symbol or for the whole account. They complement the
pre-trade :class:`~quant_os_core.risk_engine.RiskEngine`, which acts on open exposure.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Protocol


class ExitReason(StrEnum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    INVALIDATION = "invalidation"
    SIGNAL = "signal"
    TIME = "time"


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    symbol: str
    exit_time: datetime
    profit_ratio: Decimal  # e.g. Decimal("-0.02") for a 2% loss
    exit_reason: ExitReason

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol cannot be blank")
        if self.exit_time.tzinfo is None:
            raise ValueError("exit_time must be timezone-aware")


@dataclass(frozen=True, slots=True)
class Lock:
    symbol: str | None  # None locks the whole account
    until: datetime
    protection: str
    reason: str

    def applies_to(self, symbol: str, now: datetime) -> bool:
        return now < self.until and (self.symbol is None or self.symbol == symbol)


class Protection(Protocol):
    @property
    def name(self) -> str: ...

    def evaluate(self, trades: Sequence[ClosedTrade], now: datetime) -> list[Lock]: ...


def _recent(trades: Iterable[ClosedTrade], now: datetime, lookback: timedelta) -> list[ClosedTrade]:
    return [t for t in trades if now - lookback <= t.exit_time <= now]


def _positive(value: timedelta, name: str) -> None:
    if value <= timedelta(0):
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class StoplossGuard:
    """Pause trading after ``trade_limit`` losing stop-loss exits within ``lookback``."""

    trade_limit: int = 3
    lookback: timedelta = timedelta(hours=24)
    stop_duration: timedelta = timedelta(hours=12)
    only_per_symbol: bool = False
    required_profit: Decimal = Decimal("0")
    name: str = "StoplossGuard"

    def __post_init__(self) -> None:
        if self.trade_limit < 1:
            raise ValueError("trade_limit must be at least 1")
        _positive(self.lookback, "lookback")
        _positive(self.stop_duration, "stop_duration")

    def evaluate(self, trades: Sequence[ClosedTrade], now: datetime) -> list[Lock]:
        stops = [
            t
            for t in _recent(trades, now, self.lookback)
            if t.exit_reason is ExitReason.STOP_LOSS and t.profit_ratio < self.required_profit
        ]
        if not self.only_per_symbol:
            if len(stops) < self.trade_limit:
                return []
            last = max(t.exit_time for t in stops)
            return [Lock(None, last + self.stop_duration, self.name, f"{len(stops)} stop-losses")]
        locks: list[Lock] = []
        for symbol in sorted({t.symbol for t in stops}):
            hits = [t for t in stops if t.symbol == symbol]
            if len(hits) >= self.trade_limit:
                last = max(t.exit_time for t in hits)
                locks.append(
                    Lock(symbol, last + self.stop_duration, self.name, f"{len(hits)} stop-losses")
                )
        return locks


@dataclass(frozen=True, slots=True)
class MaxDrawdownProtection:
    """Pause all trading when the closed-trade equity curve falls more than the limit."""

    max_allowed_drawdown: Decimal = Decimal("0.10")
    trade_limit: int = 3
    lookback: timedelta = timedelta(days=2)
    stop_duration: timedelta = timedelta(hours=12)
    name: str = "MaxDrawdown"

    def __post_init__(self) -> None:
        if not Decimal("0") < self.max_allowed_drawdown < Decimal("1"):
            raise ValueError("max_allowed_drawdown must be in (0, 1)")
        if self.trade_limit < 1:
            raise ValueError("trade_limit must be at least 1")
        _positive(self.lookback, "lookback")
        _positive(self.stop_duration, "stop_duration")

    def evaluate(self, trades: Sequence[ClosedTrade], now: datetime) -> list[Lock]:
        recent = sorted(_recent(trades, now, self.lookback), key=lambda t: t.exit_time)
        if len(recent) < self.trade_limit:
            return []
        equity = peak = Decimal("1")
        worst = Decimal("0")
        for trade in recent:
            equity *= Decimal("1") + trade.profit_ratio
            peak = max(peak, equity)
            worst = max(worst, (peak - equity) / peak)
        if worst <= self.max_allowed_drawdown:
            return []
        until = recent[-1].exit_time + self.stop_duration
        return [Lock(None, until, self.name, f"drawdown {worst:.2%} over recent trades")]


@dataclass(frozen=True, slots=True)
class LowProfitPairs:
    """Lock a symbol whose summed profit over ``lookback`` is below ``required_profit``."""

    required_profit: Decimal = Decimal("0")
    trade_limit: int = 2
    lookback: timedelta = timedelta(hours=24)
    stop_duration: timedelta = timedelta(hours=12)
    name: str = "LowProfitPairs"

    def __post_init__(self) -> None:
        if self.trade_limit < 1:
            raise ValueError("trade_limit must be at least 1")
        _positive(self.lookback, "lookback")
        _positive(self.stop_duration, "stop_duration")

    def evaluate(self, trades: Sequence[ClosedTrade], now: datetime) -> list[Lock]:
        recent = _recent(trades, now, self.lookback)
        locks: list[Lock] = []
        for symbol in sorted({t.symbol for t in recent}):
            hits = [t for t in recent if t.symbol == symbol]
            if len(hits) < self.trade_limit:
                continue
            total = sum((t.profit_ratio for t in hits), Decimal("0"))
            if total < self.required_profit:
                last = max(t.exit_time for t in hits)
                locks.append(
                    Lock(symbol, last + self.stop_duration, self.name, f"profit {total:.2%}")
                )
        return locks


@dataclass(frozen=True, slots=True)
class CooldownPeriod:
    """Block re-entry into a symbol for ``stop_duration`` after its latest exit."""

    stop_duration: timedelta = timedelta(hours=1)
    name: str = "CooldownPeriod"

    def __post_init__(self) -> None:
        _positive(self.stop_duration, "stop_duration")

    def evaluate(self, trades: Sequence[ClosedTrade], now: datetime) -> list[Lock]:
        latest: dict[str, datetime] = {}
        for trade in trades:
            if trade.exit_time <= now:
                latest[trade.symbol] = max(
                    latest.get(trade.symbol, trade.exit_time), trade.exit_time
                )
        return [
            Lock(symbol, exit_time + self.stop_duration, self.name, "cooldown after exit")
            for symbol, exit_time in sorted(latest.items())
            if now < exit_time + self.stop_duration
        ]


class ProtectionManager:
    """Evaluates every protection and answers whether a symbol may be entered."""

    def __init__(self, protections: Sequence[Protection]) -> None:
        self.protections = tuple(protections)

    def active_locks(self, trades: Sequence[ClosedTrade], now: datetime) -> list[Lock]:
        locks: list[Lock] = []
        for protection in self.protections:
            locks.extend(lock for lock in protection.evaluate(trades, now) if now < lock.until)
        return locks

    def blocking_lock(
        self, symbol: str, trades: Sequence[ClosedTrade], now: datetime
    ) -> Lock | None:
        for lock in self.active_locks(trades, now):
            if lock.applies_to(symbol, now):
                return lock
        return None


def default_protections() -> ProtectionManager:
    """Conservative defaults for a low-frequency halal equity bot."""

    return ProtectionManager(
        [
            StoplossGuard(
                trade_limit=3, lookback=timedelta(days=2), stop_duration=timedelta(days=1)
            ),
            MaxDrawdownProtection(
                max_allowed_drawdown=Decimal("0.08"),
                trade_limit=4,
                lookback=timedelta(days=10),
                stop_duration=timedelta(days=2),
            ),
            LowProfitPairs(
                required_profit=Decimal("-0.03"),
                trade_limit=2,
                lookback=timedelta(days=14),
                stop_duration=timedelta(days=7),
            ),
            CooldownPeriod(stop_duration=timedelta(days=1)),
        ]
    )
