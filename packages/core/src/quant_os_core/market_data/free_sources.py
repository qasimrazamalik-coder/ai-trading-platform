"""Free data adapters: Alpaca stock/crypto bars and SEC EDGAR fundamentals for the Shariah screen.

Network access goes through an injectable ``fetch`` callable so the adapters can be tested
offline and swapped for an async client later. Only the standard library is used.

* Alpaca market data (free Basic plan, IEX feed): https://docs.alpaca.markets/docs/about-market-data-api
* SEC EDGAR XBRL company facts (free, no key; SEC asks for a descriptive User-Agent):
  https://www.sec.gov/search-filings/edgar-application-programming-interfaces
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from quant_os_core.agents.exits import Bar
from quant_os_core.shariah import CompanyFinancials

Fetch = Callable[[str, Mapping[str, str]], bytes]


def urllib_fetch(url: str, headers: Mapping[str, str]) -> bytes:
    request = urllib.request.Request(url, headers=dict(headers))
    with urllib.request.urlopen(request, timeout=30) as response:
        body: bytes = response.read()
    return body


def _bar(raw: Mapping[str, Any]) -> Bar:
    return Bar(
        time=datetime.fromisoformat(str(raw["t"]).replace("Z", "+00:00")),
        high=Decimal(str(raw["h"])),
        low=Decimal(str(raw["l"])),
        close=Decimal(str(raw["c"])),
    )


def _check_timeframe(timeframe: str) -> str:
    allowed = {"1Min", "5Min", "15Min", "30Min", "1Hour", "4Hour", "1Day"}
    if timeframe not in allowed:
        raise ValueError(f"timeframe must be one of {sorted(allowed)}")
    return timeframe


@dataclass(frozen=True, slots=True)
class AlpacaBars:
    """US stock bars. ``timeframe`` takes intraday values for day trading."""

    key_id: str
    secret_key: str
    fetch: Fetch = urllib_fetch
    base_url: str = "https://data.alpaca.markets/v2/stocks"
    feed: str = "iex"

    def daily(self, symbol: str, start: str, end: str) -> list[Bar]:
        return self.bars(symbol, start, end, "1Day")

    def bars(self, symbol: str, start: str, end: str, timeframe: str = "1Day") -> list[Bar]:
        frame = _check_timeframe(timeframe)
        bars: list[Bar] = []
        token: str | None = None
        headers = {"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret_key}
        while True:
            url = (
                f"{self.base_url}/{symbol.upper()}/bars?timeframe={frame}&start={start}&end={end}"
                f"&adjustment=all&feed={self.feed}&limit=10000"
            )
            if token:
                url += f"&page_token={token}"
            payload: dict[str, Any] = json.loads(self.fetch(url, headers))
            bars.extend(_bar(raw) for raw in payload.get("bars") or [])
            token = payload.get("next_page_token")
            if not token:
                return bars


@dataclass(frozen=True, slots=True)
class AlpacaCryptoBars:
    """Spot crypto bars (24/7) from Alpaca's v1beta3 endpoint, e.g. ``BTC/USD``.

    https://docs.alpaca.markets/reference/cryptobars-1 . The response keys bars by symbol.
    Keys are sent when given; the free market-data plan covers crypto bars.
    """

    key_id: str = ""
    secret_key: str = ""
    fetch: Fetch = urllib_fetch
    base_url: str = "https://data.alpaca.markets/v1beta3/crypto/us/bars"

    def bars(self, symbol: str, start: str, end: str, timeframe: str = "1Hour") -> list[Bar]:
        frame = _check_timeframe(timeframe)
        pair = symbol.upper()
        if "/" not in pair:
            raise ValueError("crypto symbols use BASE/QUOTE form, e.g. BTC/USD")
        headers = (
            {"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret_key}
            if self.key_id
            else {}
        )
        bars: list[Bar] = []
        token: str | None = None
        while True:
            url = (
                f"{self.base_url}?symbols={urllib.parse.quote(pair, safe='')}&timeframe={frame}"
                f"&start={start}&end={end}&limit=10000"
            )
            if token:
                url += f"&page_token={token}"
            payload: dict[str, Any] = json.loads(self.fetch(url, headers))
            by_symbol = payload.get("bars") or {}
            bars.extend(_bar(raw) for raw in by_symbol.get(pair) or [])
            token = payload.get("next_page_token")
            if not token:
                return bars


_TAGS: dict[str, tuple[str, ...]] = {
    "debt": ("LongTermDebt", "LongTermDebtNoncurrent", "DebtCurrent", "ShortTermBorrowings"),
    "cash": (
        "CashAndCashEquivalentsAtCarryingValue",
        "ShortTermInvestments",
        "MarketableSecuritiesCurrent",
    ),
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ),
    "assets": ("Assets",),
    "intangibles": ("Goodwill", "IntangibleAssetsNetExcludingGoodwill"),
    "shares": ("EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"),
}


def _latest(facts: Mapping[str, Any], tag: str) -> Decimal | None:
    for taxonomy in ("us-gaap", "dei"):
        node = facts.get(taxonomy, {}).get(tag)
        if not node:
            continue
        values = [v for unit in node.get("units", {}).values() for v in unit if "val" in v]
        if values:
            newest = max(values, key=lambda v: str(v.get("end", "")))
            return Decimal(str(newest["val"]))
    return None


def _sum(facts: Mapping[str, Any], tags: tuple[str, ...]) -> Decimal:
    return sum((v for t in tags if (v := _latest(facts, t)) is not None), Decimal("0"))


@dataclass(frozen=True, slots=True)
class EdgarFundamentals:
    """Builds :class:`CompanyFinancials` from EDGAR company facts.

    EDGAR has no field for *non-compliant revenue* or *sector*; those must come from a
    segment review (manual or LLM-assisted) and are passed in. Interest-bearing debt and cash
    are approximated by the listed tags; review the result before relying on it.
    """

    user_agent: str  # e.g. "Quant OS research you@example.com"
    fetch: Fetch = urllib_fetch

    def financials(
        self,
        cik: int,
        *,
        market_cap: Decimal,
        sector: str,
        noncompliant_revenue: Decimal,
    ) -> CompanyFinancials:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
        facts = json.loads(self.fetch(url, {"User-Agent": self.user_agent})).get("facts", {})
        assets = _latest(facts, "Assets")
        revenue = _sum(facts, _TAGS["revenue"][:1]) or _sum(facts, _TAGS["revenue"][1:])
        shares = _latest(facts, _TAGS["shares"][0]) or _latest(facts, _TAGS["shares"][1])
        if assets is None or not revenue or shares is None:
            raise ValueError(f"EDGAR facts for CIK {cik} lack assets, revenue or shares")
        return CompanyFinancials(
            sector=sector,
            market_cap=market_cap,
            interest_bearing_debt=_sum(facts, _TAGS["debt"]),
            interest_bearing_cash_and_securities=_sum(facts, _TAGS["cash"]),
            total_revenue=revenue,
            noncompliant_revenue=noncompliant_revenue,
            total_assets=assets,
            tangible_assets=max(Decimal("0"), assets - _sum(facts, _TAGS["intangibles"])),
            shares_outstanding=shares,
        )


def load_kmi_list(path: str) -> frozenset[str]:
    """Read PSX KMI All Share constituents from a one-symbol-per-line file (from psx.com.pk)."""

    with open(path, encoding="utf-8") as handle:
        return frozenset(
            line.strip().upper() for line in handle if line.strip() and not line.startswith("#")
        )


def utc_now() -> datetime:
    return datetime.now(UTC)
