"""Fallback fundamentals from Yahoo Finance for holdings that do not file with the SEC.

Foreign companies (an Australian REIT, a Malaysian REIT, a Turkish developer...) have no
EDGAR facts, so the look-through screen would leave them unresolved. Yahoo carries their
statements. A holding is looked up by ISIN first (Yahoo maps ISINs to its own symbols), then
by its OpenFIGI listings (ticker + Bloomberg exchange code, home market first), then by the
N-PORT ticker with an exchange or country suffix.

Rules that keep it honest:

* ratios need statements and market cap in one currency: when Yahoo's ``financialCurrency``
  differs from the quote ``currency`` (a Canadian listing reporting in USD), the market cap is
  converted at Yahoo's spot FX rate; no rate means unresolved;
* missing assets, revenue or market cap means unresolved, never a guess;
* the sector comes from Yahoo's industry name, mapped onto the gate's excluded sectors.

Yahoo data is unofficial and can lag or contain errors; the screen reports which holdings
used it so the owner can review them.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from functools import partial
from typing import Any

from quant_os_core.shariah.fund_screen import CompanyData, Holding

ZERO = Decimal(0)

# Bloomberg-style exchange codes in N-PORT tickers ("CHC AU") -> Yahoo suffixes
EXCHANGE_SUFFIX = {
    "AU": ".AX", "MK": ".KL", "SP": ".SI", "TI": ".IS", "UH": ".AE", "DH": ".AE",
    "TB": ".BK", "PM": ".PS", "LN": ".L", "JP": ".T", "JT": ".T", "HK": ".HK",
    "CN": ".TO", "GR": ".DE", "GY": ".DE", "FP": ".PA", "SW": ".SW", "SE": ".SW",
    "NA": ".AS", "KS": ".KS", "TT": ".TW", "IN": ".NS", "IB": ".NS", "SS": ".ST",
    "DC": ".CO", "NO": ".OL", "FH": ".HE", "AB": ".SR", "QD": ".QA", "IJ": ".JK",
    "KQ": ".KQ", "IS": ".NS", "SM": ".MC", "IM": ".MI", "BB": ".BR", "AV": ".VI",
    "ID": ".IR", "PL": ".LS", "NZ": ".NZ", "PW": ".WA", "BZ": ".SA", "BS": ".SA",
    "MM": ".MX", "SJ": ".JO", "KK": ".KW", "CI": ".SN", "GA": ".AT", "CT": ".TO",
    "CG": ".SS", "CS": ".SZ", "IT": ".TA", "EY": ".CA",
}  # fmt: skip
# ISO country (N-PORT invCountry) -> Yahoo suffix, when the ticker has no exchange code
COUNTRY_SUFFIX = {
    "AU": ".AX", "MY": ".KL", "SG": ".SI", "TR": ".IS", "AE": ".AE", "TH": ".BK",
    "PH": ".PS", "GB": ".L", "JP": ".T", "HK": ".HK", "CA": ".TO", "DE": ".DE",
    "FR": ".PA", "CH": ".SW", "NL": ".AS", "KR": ".KS", "TW": ".TW", "IN": ".NS",
    "SE": ".ST", "DK": ".CO", "NO": ".OL", "FI": ".HE", "SA": ".SR", "QA": ".QA",
    "ID": ".JK", "ES": ".MC", "IT": ".MI", "BE": ".BR", "AT": ".VI", "IE": ".IR",
    "PT": ".LS", "NZ": ".NZ", "PL": ".WA", "BR": ".SA", "MX": ".MX", "ZA": ".JO",
    "KW": ".KW", "CL": ".SN", "GR": ".AT", "IL": ".TA", "EG": ".CA",
}  # fmt: skip
US_EXCHANGES = frozenset({"US", "UN", "UW", "UQ", "UA", "UP", "UR", "UF", "UV"})
# quotes in minor units; Yahoo's marketCap is already in the major unit
MINOR_CURRENCY = {"GBp": "GBP", "GBX": "GBP", "ZAc": "ZAR", "ILA": "ILS"}
INDUSTRY_SECTORS = (
    ("bank", "conventional_finance"),
    ("credit services", "conventional_finance"),
    ("mortgage", "conventional_finance"),
    ("capital markets", "conventional_finance"),
    ("insurance", "conventional_insurance"),
    ("tobacco", "tobacco"),
    ("brewers", "alcohol"),
    ("wineries", "alcohol"),
    ("distillers", "alcohol"),
    ("gambling", "gambling"),
    ("casinos", "gambling"),
)


def sector_for_industry(industry: str | None) -> str | None:
    name = (industry or "").lower()
    return next((sector for key, sector in INDUSTRY_SECTORS if key in name), None)


def _figi_symbols(holding: Holding) -> list[str]:
    """Yahoo symbols from the OpenFIGI listings, the holding's home market first."""

    home = COUNTRY_SUFFIX.get(holding.country or "") or COUNTRY_SUFFIX.get((holding.isin or "")[:2])
    ranked: list[tuple[int, str]] = []
    for ticker, exchange in holding.figi_listings:
        base = ticker.upper().replace("/", "-").replace(" ", "-")
        if exchange in US_EXCHANGES:
            ranked.append((1 if holding.country == "US" else 2, base))
        elif exchange in EXCHANGE_SUFFIX:
            suffix = EXCHANGE_SUFFIX[exchange]
            ranked.append((0 if suffix == home else 1, base + suffix))
    return [symbol for _, symbol in sorted(ranked, key=lambda r: r[0])]


def candidate_symbols(holding: Holding) -> list[str]:
    out: list[str] = []
    if holding.isin:
        out.append(holding.isin)
    out.extend(_figi_symbols(holding))
    if holding.ticker:
        parts = holding.ticker.split()
        base = parts[0]
        suffix = EXCHANGE_SUFFIX.get(parts[1]) if len(parts) > 1 else None
        suffix = suffix or COUNTRY_SUFFIX.get(holding.country or "")
        if suffix:
            out.append(base + suffix)
    return list(dict.fromkeys(out))


def _dec(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return d if d.is_finite() else None


def _row(frame: Any, *names: str) -> Decimal:
    """Most recent non-empty value of the first matching row in a yfinance statement."""

    index = getattr(frame, "index", ())
    for name in names:
        if name in index:
            series = frame.loc[name].dropna()
            if len(series):
                value = _dec(series.iloc[0])
                if value is not None:
                    return value
    return ZERO


@dataclass(slots=True)
class YahooFundamentals:
    ticker_factory: Callable[[str], Any] | None = None  # yfinance.Ticker by default
    used: list[str] = field(default_factory=list)  # holdings resolved through Yahoo
    attempts: int = 2  # Yahoo connections time out now and then; retry once
    retry_pause: float = 1.0
    max_parallel: int = 4  # Yahoo throttles bursts; cap concurrent lookups
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _slots: threading.BoundedSemaphore = field(init=False)
    _held: threading.local = field(default_factory=threading.local)
    _industries: dict[str, str | None] = field(default_factory=dict)
    _fx: dict[tuple[str, str], Decimal | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._slots = threading.BoundedSemaphore(max(1, self.max_parallel))

    def _ticker(self, symbol: str) -> Any:
        with self._lock:
            if self.ticker_factory is None:
                import yfinance  # type: ignore[import-untyped]

                self.ticker_factory = yfinance.Ticker
            factory = self.ticker_factory
        return factory(symbol)

    def _call(self, fn: Callable[[], Any]) -> Any:
        """Run one Yahoo lookup under the concurrency cap, retrying a failed attempt.

        Re-entrant per thread: a lookup that needs another one (a company's FX rate) runs it in
        the slot it already holds, so threads never wait on each other's slots.
        """

        nested = getattr(self._held, "slot", False)
        for attempt in range(self.attempts):
            try:
                if nested:
                    return fn()
                with self._slots:
                    self._held.slot = True
                    try:
                        return fn()
                    finally:
                        self._held.slot = False
            except Exception:
                if attempt + 1 == self.attempts:
                    raise
                time.sleep(self.retry_pause)
        return None

    def resolve(self, holding: Holding) -> CompanyData | None:
        for symbol in candidate_symbols(holding):
            try:
                found: CompanyData | None = self._call(partial(self._lookup, symbol))
            except Exception:  # unknown symbol, network error, bad ISIN
                continue
            if found is not None:
                with self._lock:
                    self.used.append(f"{holding.name} ({symbol})")
                return found
        return None

    def _lookup(self, symbol: str) -> CompanyData | None:
        return self._company(self._ticker(symbol))

    def industry(self, symbol: str) -> str | None:
        """Yahoo's industry name for a symbol (memoised), or None when unavailable."""

        if symbol not in self._industries:
            try:
                info = self._call(lambda: self._ticker(symbol.replace(".", "-")).info) or {}
                raw = info.get("industry")
                self._industries[symbol] = str(raw) if raw else None
            except Exception:
                return None  # not memoised: a later call may succeed
        return self._industries[symbol]

    def fx_rate(self, base: str, quote: str) -> Decimal | None:
        """Units of ``quote`` per one ``base`` from Yahoo's spot FX quote."""

        if base == quote:
            return Decimal(1)
        key = (base, quote)
        if key not in self._fx:
            try:
                info = self._call(lambda: self._ticker(f"{base}{quote}=X").info) or {}
            except Exception:
                return None
            rate = _dec(info.get("regularMarketPrice"))
            self._fx[key] = rate if rate and rate > 0 else None
        return self._fx[key]

    def _company(self, ticker: Any) -> CompanyData | None:
        info = ticker.info or {}
        currency, reporting = info.get("currency"), info.get("financialCurrency")
        if not currency:
            return None
        market_cap = _dec(info.get("marketCap"))
        shares = _dec(info.get("sharesOutstanding"))
        if not market_cap or not shares:
            return None
        major = MINOR_CURRENCY.get(currency, currency)
        if reporting and reporting != major:  # convert market cap into the statements' currency
            rate = self.fx_rate(major, reporting)
            if rate is None:
                return None  # ratios need one currency; never guess a rate
            market_cap = market_cap * rate
        bs, income = ticker.balance_sheet, ticker.financials
        assets = _row(bs, "Total Assets")
        revenue = _row(income, "Total Revenue", "Operating Revenue")
        if not assets or not revenue:
            return None
        industry = info.get("industry")
        sector = sector_for_industry(industry)
        return CompanyData(
            cik=0,
            sic=None,
            shares=shares,
            debt=_row(bs, "Total Debt"),
            cash=_row(
                bs, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"
            )
            + _row(bs, "Other Short Term Investments"),
            revenue=revenue,
            interest_income=_row(income, "Interest Income", "Interest Income Non Operating"),
            assets=assets,
            tangible_assets=max(ZERO, assets - _row(bs, "Goodwill And Other Intangible Assets")),
            market_cap=market_cap,
            industry_sector=sector,
            sector_reason=f"Yahoo industry '{industry}' -> {sector}" if sector else "",
        )
