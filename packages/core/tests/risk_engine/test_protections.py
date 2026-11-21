from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from quant_os_core.risk_engine import (
    ClosedTrade,
    CooldownPeriod,
    ExitReason,
    LowProfitPairs,
    MaxDrawdownProtection,
    ProtectionManager,
    StoplossGuard,
    default_protections,
)

NOW = datetime(2026, 11, 21, 12, tzinfo=UTC)


def _trade(
    symbol: str, hours_ago: float, profit: str, reason: ExitReason = ExitReason.STOP_LOSS
) -> ClosedTrade:
    return ClosedTrade(symbol, NOW - timedelta(hours=hours_ago), Decimal(profit), reason)


def test_stoploss_guard_locks_account_after_limit() -> None:
    guard = StoplossGuard(
        trade_limit=2, lookback=timedelta(hours=6), stop_duration=timedelta(hours=2)
    )
    trades = [_trade("A", 3, "-0.01"), _trade("B", 1, "-0.02")]
    locks = guard.evaluate(trades, NOW)
    assert len(locks) == 1 and locks[0].symbol is None
    assert locks[0].until == NOW + timedelta(hours=1)
    assert guard.evaluate(trades[:1], NOW) == []
    profitable_stop = [_trade("A", 1, "0.01"), _trade("B", 1, "0.02")]
    assert guard.evaluate(profitable_stop, NOW) == []


def test_stoploss_guard_per_symbol() -> None:
    guard = StoplossGuard(trade_limit=2, only_per_symbol=True)
    trades = [_trade("A", 3, "-0.01"), _trade("A", 2, "-0.01"), _trade("B", 1, "-0.01")]
    locks = guard.evaluate(trades, NOW)
    assert [lock.symbol for lock in locks] == ["A"]


def test_max_drawdown_protection() -> None:
    protection = MaxDrawdownProtection(max_allowed_drawdown=Decimal("0.05"), trade_limit=3)
    losing = [
        _trade("A", 5, "0.02", ExitReason.TAKE_PROFIT),
        _trade("B", 3, "-0.04"),
        _trade("C", 1, "-0.03"),
    ]
    locks = protection.evaluate(losing, NOW)
    assert len(locks) == 1 and locks[0].symbol is None
    assert protection.evaluate(losing[:2], NOW) == []


def test_low_profit_pairs_locks_only_the_weak_symbol() -> None:
    protection = LowProfitPairs(required_profit=Decimal("0"), trade_limit=2)
    trades = [
        _trade("A", 5, "-0.02"),
        _trade("A", 2, "0.01", ExitReason.SIGNAL),
        _trade("B", 4, "0.03", ExitReason.TAKE_PROFIT),
        _trade("B", 1, "0.01", ExitReason.TAKE_PROFIT),
    ]
    assert [lock.symbol for lock in protection.evaluate(trades, NOW)] == ["A"]


def test_cooldown_blocks_reentry_only_for_that_symbol() -> None:
    cooldown = CooldownPeriod(stop_duration=timedelta(hours=2))
    trades = [_trade("A", 1, "0.01", ExitReason.TAKE_PROFIT), _trade("B", 3, "0.01")]
    manager = ProtectionManager([cooldown])
    assert manager.blocking_lock("A", trades, NOW) is not None
    assert manager.blocking_lock("B", trades, NOW) is None
    assert manager.blocking_lock("C", trades, NOW) is None


def test_default_manager_and_validation() -> None:
    manager = default_protections()
    assert len(manager.protections) == 4
    with pytest.raises(ValueError):
        StoplossGuard(trade_limit=0)
    with pytest.raises(ValueError):
        MaxDrawdownProtection(max_allowed_drawdown=Decimal("1"))
    with pytest.raises(ValueError):
        CooldownPeriod(stop_duration=timedelta(0))
    with pytest.raises(ValueError):
        ClosedTrade("A", datetime(2026, 1, 1), Decimal("0"), ExitReason.SIGNAL)
