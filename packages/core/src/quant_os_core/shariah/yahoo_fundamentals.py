"""Fallback fundamentals from Yahoo Finance for holdings that do not file with the SEC.

Foreign companies (an Australian REIT, a Malaysian REIT, a Turkish developer...) have no
EDGAR facts, so the look-through screen would leave them unresolved. Yahoo carries their
statements. A holding is looked up by ISIN first (Yahoo maps ISINs to its own symbols), then
by the N-PORT ticker with an exchange or country suffix.

Rules that keep it honest:

* statements and market cap must be in the same currency (``financialCurrency`` equal to
  the quote ``currency``), so the screen can use ratios without FX conversion;
* missing assets, revenue or market cap means unresolved, never a guess;
* the sector comes from Yahoo's industry name, mapped onto the gate's excluded sectors.

Yahoo data is unofficial and can lag or contain errors; the screen reports which holdings
used it so the owner can review them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
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
}  # fmt: skip
# ISO country (N-PORT invCountry) -> Yahoo suffix, when the ticker has no exchange code
COUNTRY_SUFFIX = {
    "AU": ".AX", "MY": ".KL", "SG": ".SI", "TR": ".IS", "AE": ".AE", "TH": ".BK",
    "PH": ".PS", "GB": ".L", "JP": ".T", "HK": ".HK", "CA": ".TO", "DE": ".DE",
    "FR": ".PA", "CH": ".SW", "NL": ".AS", "KR": ".KS", "TW": ".TW", "IN": ".NS",
    "SE": ".ST", "DK": ".CO", "NO": ".OL", "FI": ".HE", "SA": ".SR", "QA": ".QA",
    "ID": ".JK",
}  # fmt: skip
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


def candidate_symbols(holding: Holding) -> list[str]:
    out: list[str] = []
    if holding.isin:
        out.append(holding.isin)
    if holding.ticker:
        parts = holding.ticker.split()
        base = parts[0]
        suffix = EXCHANGE_SUFFIX.get(parts[1]) if len(parts) > 1 else None
        suffix = suffix or COUNTRY_SUFFIX.get(holding.country or "")
        if suffix:
            out.append(base + suffix)
    return out


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

    def _ticker(self, symbol: str) -> Any:
        if self.ticker_factory is None:
            import yfinance

            self.ticker_factory = yfinance.Ticker
        return self.ticker_factory(symbol)

    def resolve(self, holding: Holding) -> CompanyData | None:
        for symbol in candidate_symbols(holding):
            try:
                found = self._company(self._ticker(symbol))
            except Exception:  # unknown symbol, network error, bad ISIN
                continue
            if found is not None:
                self.used.append(f"{holding.name} ({symbol})")
                return found
        return None

    def _company(self, ticker: Any) -> CompanyData | None:
        info = ticker.info or {}
        currency, reporting = info.get("currency"), info.get("financialCurrency")
        if not currency or (reporting and reporting != currency):
            return None  # ratios need statements and market cap in one currency
        market_cap = _dec(info.get("marketCap"))
        shares = _dec(info.get("sharesOutstanding"))
        bs, income = ticker.balance_sheet, ticker.financials
        assets = _row(bs, "Total Assets")
        revenue = _row(income, "Total Revenue", "Operating Revenue")
        if not market_cap or not shares or not assets or not revenue:
            return None
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
            industry_sector=sector_for_industry(info.get("industry")),
        )
