"""Day-trading helpers: session times, settlement times and a tighter plan policy.

* US equities settle T+1 (SEC rule 15c6-1 since 28 May 2024); spot crypto settles on the
  exchange ledger at the trade and trades 24/7, so it has no session close.
* Exchange holidays are only known if passed in as ``holidays``; without them, settlement
  around a holiday is estimated one business day too early.
* The halal account is a cash account (no margin), so the FINRA pattern-day-trader margin
  rule does not apply; a cash account may only buy with settled cash, which the caller models
  through ``free_cash``.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from quant_os_core.agents.trade_plan import PlanPolicy
from quant_os_core.instruments import AssetClass

NEW_YORK = ZoneInfo("America/New_York")
US_CLOSE = time(16, 0)
SESSION_ASSETS: frozenset[AssetClass] = frozenset({AssetClass.EQUITY, AssetClass.ETF})


def _next_business_day(day: date, holidays: Collection[date]) -> date:
    candidate = day + timedelta(days=1)
    while candidate.weekday() >= 5 or candidate in holidays:
        candidate += timedelta(days=1)
    return candidate


def session_close(asset_class: AssetClass, at: datetime) -> datetime | None:
    """US close (16:00 New York) on the trade's local date; ``None`` for 24/7 assets."""

    if at.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    if asset_class not in SESSION_ASSETS:
        return None
    local_day = at.astimezone(NEW_YORK).date()
    return datetime.combine(local_day, US_CLOSE, tzinfo=NEW_YORK)


def flatten_time(
    asset_class: AssetClass, at: datetime, buffer: timedelta = timedelta(minutes=15)
) -> datetime | None:
    """When a day trade must be closed: ``buffer`` before the session close."""

    close = session_close(asset_class, at)
    return None if close is None else close - buffer


def settlement_time(
    asset_class: AssetClass, traded_at: datetime, holidays: Collection[date] = ()
) -> datetime:
    """When bought units are settled: at the trade for crypto, T+1 close for US equities."""

    if traded_at.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    if asset_class not in SESSION_ASSETS:
        return traded_at
    trade_day = traded_at.astimezone(NEW_YORK).date()
    return datetime.combine(_next_business_day(trade_day, holidays), US_CLOSE, tzinfo=NEW_YORK)


def day_trading_policy() -> PlanPolicy:
    """Intraday plan limits: tighter stops, half the risk per trade, more trades per day."""

    return PlanPolicy(
        min_confidence=Decimal("0.60"),
        min_reward_to_risk=Decimal("1.5"),
        min_stop_distance=Decimal("0.003"),
        max_stop_distance=Decimal("0.03"),
        risk_per_trade=Decimal("0.005"),
        max_position_fraction=Decimal("0.20"),
        max_trades_per_day=10,
    )
