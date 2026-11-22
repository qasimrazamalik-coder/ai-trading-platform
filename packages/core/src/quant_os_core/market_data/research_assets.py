"""Research-only daily data for oil and forex, via yfinance.

    CL=F  WTI crude oil futures, front month      BZ=F  Brent crude futures, front month
    EURUSD=X, GBPUSD=X, USDPKR=X                   spot FX rates

Why research only. Oil here is quoted from futures contracts, and futures are blocked in
every profile (GUARDRAILS R1/R2: spot ownership only; checked against IIFA Resolution 63).
Speculative forex is blocked (R10; AAOIFI Shariah Standard 1 on trading in currencies), and
Pakistani residents may not use offshore forex or CFD venues (R14). These series are still
useful as context: oil for inflation and energy-sector regimes, USD/PKR for converting a
Pakistani owner's results, EUR and GBP for the dollar's direction.

So every series is tagged ``research_only=True`` (it cannot be built otherwise), carries the
rules that block trading it, and this module has no path to orders: it imports nothing from
execution, sizing, the risk engine or the entry pipeline, and exposes no function that
produces a trade plan, signal or order.

yfinance is imported lazily; tests inject a fake ``fetch``.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

ResearchFetch = Callable[[str, str], Sequence[Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class ResearchAsset:
    symbol: str
    description: str
    blocked_by: tuple[str, ...]  # GUARDRAILS rules that forbid trading it


RESEARCH_ASSETS: dict[str, ResearchAsset] = {
    a.symbol: a
    for a in (
        ResearchAsset("CL=F", "WTI crude oil futures, front month", ("R1", "R2")),
        ResearchAsset("BZ=F", "Brent crude oil futures, front month", ("R1", "R2")),
        ResearchAsset("EURUSD=X", "Euro / US dollar spot rate", ("R10", "R14")),
        ResearchAsset("GBPUSD=X", "British pound / US dollar spot rate", ("R10", "R14")),
        ResearchAsset("USDPKR=X", "US dollar / Pakistani rupee spot rate", ("R10", "R14")),
    )
}


@dataclass(frozen=True, slots=True)
class ResearchPoint:
    time: datetime
    high: Decimal
    low: Decimal
    close: Decimal


@dataclass(frozen=True, slots=True)
class ResearchSeries:
    asset: ResearchAsset
    points: tuple[ResearchPoint, ...]
    research_only: Literal[True] = True
    note: str = "research only: not tradable under GUARDRAILS; never route to orders"

    def __post_init__(self) -> None:
        if self.research_only is not True:
            raise ValueError("research assets are always research_only")

    @property
    def symbol(self) -> str:
        return self.asset.symbol

    def closes(self) -> list[Decimal]:
        return [p.close for p in self.points]


def yfinance_fetch(symbol: str, period: str) -> list[dict[str, Any]]:
    """Daily rows ``{time, high, low, close}`` from yfinance (network; not used in tests)."""

    try:
        yf = importlib.import_module("yfinance")
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed; run `pip install yfinance`") from exc
    frame = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False)
    return [
        {"time": index, "high": row["High"], "low": row["Low"], "close": row["Close"]}
        for index, row in frame.iterrows()
    ]


def _time(value: Any) -> datetime:
    stamp = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
    if isinstance(stamp, str):
        stamp = datetime.fromisoformat(stamp)
    if not isinstance(stamp, datetime):
        raise ValueError(f"unsupported timestamp {value!r}")
    return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)


def _price(value: Any) -> Decimal | None:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return Decimal(str(number))


def load_research_series(
    symbol: str, *, period: str = "5y", fetch: ResearchFetch = yfinance_fetch
) -> ResearchSeries:
    """One research series; rows with missing, non-positive or inconsistent prices are dropped."""

    asset = RESEARCH_ASSETS.get(symbol.strip().upper())
    if asset is None:
        raise ValueError(f"unknown research symbol; choose from {sorted(RESEARCH_ASSETS)}")
    points: dict[datetime, ResearchPoint] = {}
    for row in fetch(asset.symbol, period):
        high, low, close = (_price(row.get(k, math.nan)) for k in ("high", "low", "close"))
        if high is None or low is None or close is None or not low <= close <= high:
            continue
        when = _time(row["time"])
        points[when] = ResearchPoint(when, high, low, close)
    return ResearchSeries(asset, tuple(points[t] for t in sorted(points)))


def load_all_research(
    *, period: str = "5y", fetch: ResearchFetch = yfinance_fetch
) -> dict[str, ResearchSeries]:
    return {s: load_research_series(s, period=period, fetch=fetch) for s in RESEARCH_ASSETS}
