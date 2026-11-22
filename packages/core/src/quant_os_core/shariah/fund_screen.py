"""Look-through Shariah screen for ETFs from SEC filings (N-PORT holdings + EDGAR facts).

    python -m quant_os_core.shariah.fund_screen SPUS HLAL UMMA SPRE \\
        --user-agent "Your Name you@example.com" --config paper_core.json --write

For each fund: find its latest N-PORT-P filing, read every holding, pull each company's
XBRL facts and SIC code from EDGAR, and add up the fund's *share* of each company
(value held / company market cap) into one fund-level :class:`CompanyFinancials`:

* interest-bearing debt and cash/securities: the fund's share of each company's, plus any
  conventional debt securities or cash the fund holds directly;
* non-compliant revenue: the fund's share of each company's interest income, plus all of the
  revenue of companies in an excluded sector (SIC-mapped: banks, insurers, alcohol, ...);
* market cap: the fund's equity holdings at market value.

Fails closed: if less than ``min_coverage`` of the fund's equity value can be resolved to
EDGAR data, no financials are produced. Funds that are not equity funds (sukuk funds hold
debt-like certificates; physical-gold trusts file no N-PORT) are reported as unscreenable
here and stay blocked until the owner decides how to treat them.

Holdings are matched by identifier before name: the N-PORT ticker, then the CUSIP/ISIN mapped
through OpenFIGI to exchange listings (a US listing finds the SEC filer; a foreign listing gives
Yahoo a symbol with the right exchange suffix), then the normalised company name.

SIC codes marked "review by hand" (7011 hotels, 2011 meat packing) are checked against Yahoo's
industry; REITs (SIC 6798) are checked for Yahoo's 'REIT - Mortgage'. Every sector flag keeps
its reason, and the CLI prints them.

SEC asks every client to send a User-Agent with a contact email and to stay under ten
requests per second. Company data is fetched by a thread pool that shares one rate limiter,
and responses are cached on disk to keep the request count low. Results do not depend on the
order threads finish: the pool only fills caches, and aggregation runs in holding order.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from quant_os_core.market_data.free_sources import _TAGS, _latest, _sum, urllib_fetch
from quant_os_core.shariah.rules import CompanyFinancials

Fetch = Callable[[str, Mapping[str, str]], bytes]
Post = Callable[[str, bytes, Mapping[str, str]], bytes]
FUND_SECTOR = "diversified_fund"
ZERO = Decimal(0)

# SIC code ranges -> excluded sector names used by the Shariah gate (R7)
SIC_SECTORS: tuple[tuple[int, int, str], ...] = (
    (2082, 2085, "alcohol"),  # malt beverages, wine, distilled spirits (2086 is soft drinks)
    (2100, 2199, "tobacco"),
    (2011, 2011, "pork"),  # meat packing: conservative, review by hand
    (6000, 6199, "conventional_finance"),  # banks, credit, mortgage
    (6200, 6299, "conventional_finance"),  # brokers and dealers
    (6300, 6411, "conventional_insurance"),
    # 6798 (REITs) is not mapped: it covers every REIT. Mortgage REITs are caught by the
    # interest-income share of revenue, or by Yahoo's 'REIT - Mortgage' industry.
    (7990, 7999, "gambling"),
    (7011, 7011, "gambling"),  # hotels and motels, incl. casino hotels; review by hand
)
# SIC codes whose sector flag is a guess: Yahoo's industry confirms it or, when it names an
# industry listed here, clears it. Meat packing has no clearing industry because no Yahoo
# industry rules out pork, so those flags stay until the owner reviews them.
REVIEW_SIC: dict[int, tuple[str, tuple[str, ...]]] = {
    7011: ("gambling", ("lodging",)),  # 'Lodging' = hotels; 'Resorts & Casinos' confirms
    2011: ("pork", ()),
}
REIT_SIC = 6798  # checked for Yahoo's 'REIT - Mortgage' (interest income, not rent)
INTEREST_INCOME_TAGS = (
    "InvestmentIncomeInterest",
    "InterestAndDividendIncomeOperating",
    "InterestIncomeOperating",
)
_SUFFIX = re.compile(
    r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|HOLDINGS?|GROUP|"
    r"CLASS [A-Z]|CL [A-Z]|THE|NV|SA|AG|SE)\b"
)


def sector_for_sic(sic: int | None) -> str | None:
    if sic is None:
        return None
    return next((name for lo, hi, name in SIC_SECTORS if lo <= sic <= hi), None)


def normalize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9 ]", " ", name.upper().replace("&", " AND "))
    return " ".join(_SUFFIX.sub(" ", cleaned).split())


# ---------------------------------------------------------------- N-PORT parsing


@dataclass(frozen=True, slots=True)
class Holding:
    name: str
    value_usd: Decimal
    balance: Decimal  # units held (shares for equities)
    asset_category: str  # N-PORT assetCat: EC equity common, DBT debt, ...
    ticker: str | None = None
    cusip: str | None = None
    isin: str | None = None
    country: str | None = None  # N-PORT invCountry, ISO 3166 alpha-2
    # (ticker, Bloomberg exchange code) listings from OpenFIGI for the CUSIP/ISIN
    figi_listings: tuple[tuple[str, str], ...] = ()

    @property
    def is_equity(self) -> bool:
        return self.asset_category in ("EC", "EP")

    @property
    def identifier(self) -> tuple[str, str] | None:
        """(OpenFIGI idType, value): the ISIN if valid, else a real (non-placeholder) CUSIP."""

        isin = (self.isin or "").strip().upper()
        if len(isin) == 12 and isin.isalnum():
            return ("ID_ISIN", isin)
        cusip = (self.cusip or "").strip().upper()
        if len(cusip) == 9 and cusip.isalnum() and cusip.strip("0"):
            return ("ID_CUSIP", cusip)
        return None


@dataclass(frozen=True, slots=True)
class FundHoldings:
    series_id: str
    report_date: str
    net_assets: Decimal
    holdings: tuple[Holding, ...]


def _strip(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(node: ET.Element, name: str) -> ET.Element | None:
    return next((c for c in node.iter() if _strip(c.tag) == name), None)


def _text(node: ET.Element, name: str) -> str:
    found = _child(node, name)
    return (found.text or "").strip() if found is not None else ""


def _dec(raw: str) -> Decimal:
    try:
        return Decimal(raw)
    except (InvalidOperation, ValueError):
        return ZERO


def parse_nport(xml: bytes) -> FundHoldings:
    root = ET.fromstring(xml)
    holdings: list[Holding] = []
    for node in root.iter():
        if _strip(node.tag) != "invstOrSec":
            continue
        ticker_node = next(
            (c for c in node.iter() if _strip(c.tag) == "ticker" and c.get("value")), None
        )
        holdings.append(
            Holding(
                name=_text(node, "name") or _text(node, "title"),
                value_usd=_dec(_text(node, "valUSD")),
                balance=_dec(_text(node, "balance")),
                asset_category=_text(node, "assetCat") or "UNKNOWN",
                ticker=(ticker_node.get("value") or "").upper() or None
                if ticker_node is not None
                else None,
                cusip=_text(node, "cusip") or None,
                isin=next(
                    (
                        c.get("value")
                        for c in node.iter()
                        if _strip(c.tag) == "isin" and c.get("value")
                    ),
                    None,
                ),
                country=_text(node, "invCountry") or None,
            )
        )
    return FundHoldings(
        series_id=_text(root, "seriesId"),
        report_date=_text(root, "repPdDate") or _text(root, "repPdEnd"),
        net_assets=_dec(_text(root, "netAssets")),
        holdings=tuple(holdings),
    )


# ---------------------------------------------------------------- company data


@dataclass(frozen=True, slots=True)
class CompanyData:
    cik: int
    sic: int | None
    shares: Decimal
    debt: Decimal
    cash: Decimal
    revenue: Decimal
    interest_income: Decimal
    assets: Decimal
    tangible_assets: Decimal
    # market cap in the statements' currency; None means shares x the fund's price (USD)
    market_cap: Decimal | None = None
    industry_sector: str | None = None  # excluded sector from a non-SEC source's industry
    sic_cleared: bool = False  # a review-by-hand SIC flag that Yahoo's industry ruled out
    sector_reason: str = ""  # why the sector was flagged, confirmed or cleared

    @property
    def sector(self) -> str | None:
        sic_sector = None if self.sic_cleared else sector_for_sic(self.sic)
        return sic_sector or self.industry_sector

    @property
    def reason(self) -> str:
        if self.sector_reason:
            return self.sector_reason
        if sector_for_sic(self.sic) is not None:
            return f"SIC {self.sic} -> {sector_for_sic(self.sic)}"
        return ""


def review_sector(company: CompanyData, industry: str | None) -> CompanyData:
    """Check a review-by-hand SIC flag, or a REIT, against Yahoo's industry name."""

    from quant_os_core.shariah.yahoo_fundamentals import sector_for_industry

    sic = company.sic
    named = f"Yahoo industry '{industry}'" if industry else "no Yahoo industry"
    mapped = sector_for_industry(industry)
    if sic == REIT_SIC:
        if mapped is not None:  # 'REIT - Mortgage'
            return replace(
                company,
                industry_sector=mapped,
                sector_reason=f"SIC {sic} REIT, {named} -> {mapped}",
            )
        return company
    if sic not in REVIEW_SIC:
        return company
    sector, clearing = REVIEW_SIC[sic]
    if industry is None:
        return replace(company, sector_reason=f"SIC {sic} -> {sector}; {named}, kept")
    if mapped is not None:
        return replace(company, sector_reason=f"SIC {sic} -> {sector}; confirmed by {named}")
    if any(key in industry.lower() for key in clearing):
        return replace(
            company, sic_cleared=True, sector_reason=f"SIC {sic} -> {sector}; cleared by {named}"
        )
    return replace(
        company,
        sector_reason=f"SIC {sic} -> {sector}; {named} cannot rule it out, kept (review by hand)",
    )


def _usd_only(facts: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only USD money values and share counts.

    Foreign 20-F filers report in EUR, JPY, TWD...; read as dollars against a dollar market
    cap, their debt looks tens of times too large. Without USD values the company falls
    through to the fallback source, which handles currency.
    """

    out: dict[str, Any] = {}
    for taxonomy, tags in facts.items():
        kept: dict[str, Any] = {}
        for tag, node in tags.items():
            units = {u: v for u, v in node.get("units", {}).items() if u in ("USD", "shares")}
            if units:
                kept[tag] = {**node, "units": units}
        out[taxonomy] = kept
    return out


def company_from_facts(cik: int, facts: Mapping[str, Any], sic: int | None) -> CompanyData | None:
    facts = _usd_only(facts)
    assets = _latest(facts, "Assets")
    revenue = _sum(facts, _TAGS["revenue"][:1]) or _sum(facts, _TAGS["revenue"][1:])
    shares = _latest(facts, _TAGS["shares"][0]) or _latest(facts, _TAGS["shares"][1])
    if assets is None or not revenue or not shares:
        return None
    return CompanyData(
        cik=cik,
        sic=sic,
        shares=shares,
        debt=_sum(facts, _TAGS["debt"]),
        cash=_sum(facts, _TAGS["cash"]),
        revenue=revenue,
        interest_income=_sum(facts, INTEREST_INCOME_TAGS[:1])
        or _sum(facts, INTEREST_INCOME_TAGS[1:]),
        assets=assets,
        tangible_assets=max(ZERO, assets - _sum(facts, _TAGS["intangibles"])),
    )


def frame_requests(today: date) -> list[tuple[str, str, str, list[str]]]:
    """(taxonomy, tag, unit, periods oldest-first) for the screening tags."""

    # quarter-ends filed by now: skip the current and the just-finished quarter
    q = (today.month - 1) // 3  # quarters fully past this year
    year, quarter = (today.year, q - 1) if q >= 2 else (today.year - 1, q + 3)
    instants = []
    for _ in range(3):
        instants.append(f"CY{year}Q{quarter}I")
        year, quarter = (year, quarter - 1) if quarter > 1 else (year - 1, 4)
    instants.reverse()
    annual = [f"CY{today.year - 2}", f"CY{today.year - 1}"]
    balance = ("Assets", *_TAGS["debt"], *_TAGS["cash"], *_TAGS["intangibles"])
    income = (*_TAGS["revenue"], *INTEREST_INCOME_TAGS)
    return [
        *[("us-gaap", t, "USD", instants) for t in balance],
        ("dei", "EntityCommonStockSharesOutstanding", "shares", instants),
        *[("us-gaap", t, "USD", annual) for t in income],
    ]


def company_from_frame(
    cik: int, values: Mapping[str, Decimal], sic: int | None
) -> CompanyData | None:
    def total(tags: Sequence[str]) -> Decimal:
        return sum((values.get(t, ZERO) for t in tags), ZERO)

    assets = values.get("Assets")
    revenue = values.get("Revenues") or total(_TAGS["revenue"][1:2]) or total(_TAGS["revenue"][2:])
    shares = values.get("EntityCommonStockSharesOutstanding")
    if not assets or not revenue or not shares:
        return None
    return CompanyData(
        cik=cik,
        sic=sic,
        shares=shares,
        debt=total(_TAGS["debt"]),
        cash=total(_TAGS["cash"]),
        revenue=revenue,
        interest_income=values.get(INTEREST_INCOME_TAGS[0]) or total(INTEREST_INCOME_TAGS[1:]),
        assets=assets,
        tangible_assets=max(ZERO, assets - total(_TAGS["intangibles"])),
    )


# ---------------------------------------------------------------- aggregation


@dataclass(slots=True)
class ScreenResult:
    fund: str
    report_date: str = ""
    financials: CompanyFinancials | None = None
    coverage: float = 0.0
    holdings: int = 0
    excluded_sector_weight: float = 0.0
    unresolved: list[str] = field(default_factory=list)
    flagged: list[str] = field(default_factory=list)  # sector flags and clearings, with reasons
    reason: str = ""


def look_through(
    fund: str,
    data: FundHoldings,
    resolve: Callable[[Holding], CompanyData | None],
    *,
    min_coverage: float = 0.90,
) -> ScreenResult:
    """Aggregate the fund's share of each holding into fund-level screening inputs."""

    result = ScreenResult(fund, data.report_date, holdings=len(data.holdings))
    equity = [h for h in data.holdings if h.is_equity and h.value_usd > 0]
    equity_value = sum((h.value_usd for h in equity), ZERO)
    if equity_value <= 0 or equity_value < data.net_assets * Decimal("0.5"):
        result.reason = "not an equity fund (under half its assets in common stock)"
        return result
    debt = cash = revenue = noncompliant = assets = tangible = ZERO
    covered = excluded = ZERO
    for h in equity:
        company = resolve(h)
        if company is None or h.balance <= 0:
            result.unresolved.append(h.name)
            continue
        market_cap = company.market_cap or company.shares * (h.value_usd / h.balance)
        if market_cap <= 0:
            result.unresolved.append(h.name)
            continue
        # value_usd x (item / market cap): the fund's slice of the company in USD. Item and
        # market cap share a currency, so foreign statements need no FX conversion.
        share = h.value_usd / market_cap
        if share > 1:  # a fund cannot own more than the company: the data is wrong
            result.unresolved.append(h.name)
            continue
        covered += h.value_usd
        debt += share * company.debt
        cash += share * company.cash
        revenue += share * company.revenue
        assets += share * company.assets
        tangible += share * company.tangible_assets
        weight = float(h.value_usd / equity_value)
        if company.sector is not None:
            excluded += h.value_usd
            noncompliant += share * company.revenue
            result.flagged.append(f"{h.name} ({weight:.2%}): {company.sector} - {company.reason}")
        else:
            noncompliant += share * min(company.interest_income, company.revenue)
            if company.sector_reason:  # a review flag that was cleared
                result.flagged.append(f"{h.name} ({weight:.2%}): not excluded - {company.reason}")
    # conventional debt securities and cash held directly by the fund are interest-bearing
    direct = sum((h.value_usd for h in data.holdings if not h.is_equity and h.value_usd > 0), ZERO)
    result.coverage = float(covered / equity_value)
    result.excluded_sector_weight = float(excluded / equity_value)
    if result.coverage < min_coverage or revenue <= 0 or assets <= 0:
        result.reason = (
            f"only {result.coverage:.0%} of equity value resolved to company financials "
            f"(need {min_coverage:.0%})"
        )
        return result
    # scale the covered part up to the whole equity sleeve so ratios use full market cap
    scale = equity_value / covered
    result.financials = CompanyFinancials(
        sector=FUND_SECTOR,
        market_cap=equity_value,
        interest_bearing_debt=debt * scale,
        interest_bearing_cash_and_securities=cash * scale + direct,
        total_revenue=revenue * scale,
        noncompliant_revenue=noncompliant * scale,
        total_assets=assets * scale,
        tangible_assets=tangible * scale,
        shares_outstanding=Decimal(1),  # not used by the ratio screen for a fund
    )
    result.reason = "ok"
    return result


# ---------------------------------------------------------------- rate limiting


class RateLimiter:
    """Spaces requests at least ``min_interval`` apart across every thread that shares it.

    Each caller reserves the next free slot under a lock, then sleeps outside the lock until
    that slot, so N threads together never exceed 1 / ``min_interval`` requests per second.
    """

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            slot = max(now, self._next)
            self._next = slot + self.min_interval
        if slot > now:
            self._sleep(slot - now)


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(body)
    os.replace(tmp, path)


# ---------------------------------------------------------------- OpenFIGI


def urllib_post(url: str, body: bytes, headers: Mapping[str, str]) -> bytes:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        out: bytes = response.read()
    return out


OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
US_EXCHANGES = frozenset({"US", "UN", "UW", "UQ", "UA", "UP", "UR", "UF", "UV"})


def _chunks(items: Sequence[tuple[str, str]], size: int) -> Iterable[Sequence[tuple[str, str]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


@dataclass(slots=True)
class OpenFigi:
    """Maps ISIN/CUSIP to equity exchange listings through the OpenFIGI API.

    Without an API key OpenFIGI allows 25 requests a minute of up to 10 jobs each, so requests
    are spaced 2.5 seconds apart and batched. Answers (including "not found") are cached on
    disk one file per identifier; errors are not cached, so the next run retries them.
    """

    cache_dir: Path = Path("data/sec_cache/openfigi")
    post: Post = urllib_post
    api_key: str | None = None
    batch: int = 10
    min_interval: float = 2.5
    sleep: Callable[[float], None] = time.sleep
    _memo: dict[tuple[str, str], tuple[tuple[str, str], ...]] = field(default_factory=dict)
    _limiter: RateLimiter = field(init=False)

    def __post_init__(self) -> None:
        self._limiter = RateLimiter(self.min_interval, sleep=self.sleep)

    def _path(self, ident: tuple[str, str]) -> Path:
        return self.cache_dir / f"{ident[0]}_{re.sub(r'[^A-Z0-9]', '_', ident[1])}.json"

    def _cached(self, ident: tuple[str, str]) -> tuple[tuple[str, str], ...] | None:
        if ident in self._memo:
            return self._memo[ident]
        path = self._path(ident)
        if not path.exists():
            return None
        listings = tuple((str(t), str(x)) for t, x in json.loads(path.read_text("utf-8")))
        self._memo[ident] = listings
        return listings

    def listings(self, ident: tuple[str, str] | None) -> tuple[tuple[str, str], ...]:
        if ident is None:
            return ()
        return self._cached(ident) or ()

    def map_many(self, idents: Iterable[tuple[str, str]]) -> int:
        """Look up every identifier not cached yet; return how many requests were sent."""

        todo = sorted({i for i in idents if self._cached(i) is None})
        sent = 0
        for chunk in _chunks(todo, self.batch):
            body = json.dumps([{"idType": t, "idValue": v} for t, v in chunk]).encode()
            rows = self._request(body)
            sent += 1
            if rows is None or len(rows) != len(chunk):
                continue  # network or API failure: leave unmapped, retry next run
            for ident, row in zip(chunk, rows, strict=True):
                if not isinstance(row, dict) or "error" in row:
                    continue
                listings = tuple(
                    dict.fromkeys(
                        (str(d["ticker"]), str(d["exchCode"]))
                        for d in row.get("data", [])
                        if d.get("marketSector") == "Equity"
                        and d.get("ticker")
                        and d.get("exchCode")
                    )
                )
                self._memo[ident] = listings
                _atomic_write(self._path(ident), json.dumps(listings).encode())
        return sent

    def _request(self, body: bytes) -> list[Any] | None:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-OPENFIGI-APIKEY"] = self.api_key
        for _ in range(3):
            self._limiter.wait()
            try:
                rows = json.loads(self.post(OPENFIGI_URL, body, headers))
            except urllib.error.HTTPError as exc:
                if exc.code == 429:  # over the per-minute limit: wait out the window
                    self.sleep(20)
                    continue
                return None
            except Exception:
                return None
            return rows if isinstance(rows, list) else None
        return None


# ---------------------------------------------------------------- SEC client


@dataclass(slots=True)
class SecClient:
    user_agent: str
    cache_dir: Path = Path("data/sec_cache")
    fetch: Fetch = urllib_fetch
    min_interval: float = 0.12  # stay under SEC's ten requests per second, across all threads
    _tickers: dict[str, int] | None = None
    _names: dict[str, int] | None = None
    _frames: dict[int, dict[str, Decimal]] | None = None
    fallback: Callable[[Holding], CompanyData | None] | None = (
        None  # e.g. Yahoo, for non-SEC filers
    )
    today: date = field(default_factory=date.today)
    workers: int = 8
    figi: OpenFigi | None = None
    # Yahoo industry name for a US ticker; confirms review-by-hand SICs and mortgage REITs
    industry: Callable[[str], str | None] | None = None
    _limiter: RateLimiter = field(init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _cik_ticker: dict[int, str] = field(default_factory=dict, init=False)
    _sics: dict[int, int | None] = field(default_factory=dict, init=False)
    _companies: dict[int, CompanyData | None] = field(default_factory=dict, init=False)
    _resolved: dict[Holding, CompanyData | None] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if "@" not in self.user_agent:
            raise ValueError("SEC requires a User-Agent with a contact email")
        self._limiter = RateLimiter(self.min_interval)

    def get(self, url: str) -> bytes:
        key = re.sub(r"[^A-Za-z0-9._-]", "_", url.split("://", 1)[-1])[-180:]
        path = self.cache_dir / key
        if path.exists():
            return path.read_bytes()
        self._limiter.wait()
        body = self.fetch(url, {"User-Agent": self.user_agent, "Accept-Encoding": "identity"})
        _atomic_write(path, body)
        return body

    def json(self, url: str) -> Any:
        return json.loads(self.get(url))

    def _try_get(self, url: str) -> bytes | None:
        try:
            return self.get(url)
        except Exception:  # a frame not published yet, a removed CIK
            return None

    def fund_ids(self, ticker: str) -> tuple[int, str] | None:
        mf = self.json("https://www.sec.gov/files/company_tickers_mf.json")
        fields = mf["fields"]
        for row in mf["data"]:
            item = dict(zip(fields, row, strict=False))
            if str(item.get("symbol", "")).upper() == ticker.upper():
                return int(item["cik"]), str(item["seriesId"])
        return None

    def latest_nport(self, cik: int, series_id: str, scan: int = 40) -> FundHoldings | None:
        sub = self.json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        recent = sub["filings"]["recent"]
        for i, form in enumerate(recent["form"][: scan * 4]):
            if form != "NPORT-P":
                continue
            acc = recent["accessionNumber"][i].replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/primary_doc.xml"
            parsed = parse_nport(self.get(url))
            if parsed.series_id == series_id:
                return parsed
        return None

    def _company_maps(self) -> tuple[dict[str, int], dict[str, int]]:
        with self._lock:
            if self._tickers is None or self._names is None:
                raw = self.json("https://www.sec.gov/files/company_tickers.json")
                tickers: dict[str, int] = {}
                names: dict[str, int] = {}
                for v in raw.values():
                    cik, ticker = int(v["cik_str"]), str(v["ticker"]).upper()
                    tickers[ticker] = cik
                    self._cik_ticker.setdefault(cik, ticker)  # first listed = primary class
                    names.setdefault(normalize_name(str(v["title"])), cik)
                self._tickers, self._names = tickers, names
            return self._tickers, self._names

    def frames(self) -> dict[int, dict[str, Decimal]]:
        """Every company's latest value of each screening tag, from SEC XBRL frames.

        One request returns a tag for all filers, so about 45 requests cover every holding.
        Balance-sheet tags use the latest quarter-ends (newest wins); income tags use the last
        full calendar year. Frames download in parallel and are applied in a fixed order.
        """

        with self._lock:
            if self._frames is not None:
                return self._frames
            jobs = [
                (tag, f"https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json")
                for taxonomy, tag, unit, periods in frame_requests(self.today)
                for period in periods  # oldest first, so newer values overwrite
            ]
            with ThreadPoolExecutor(max(1, self.workers)) as pool:
                bodies = list(pool.map(self._try_get, [url for _, url in jobs]))
            out: dict[int, dict[str, Decimal]] = {}
            for (tag, _), body in zip(jobs, bodies, strict=True):
                if body is None:
                    continue
                try:
                    rows = json.loads(body).get("data", [])
                except ValueError:
                    continue
                for row in rows:
                    out.setdefault(int(row["cik"]), {})[tag] = Decimal(str(row["val"]))
            self._frames = out
            return out

    def _sic(self, cik: int) -> int | None:
        if cik not in self._sics:
            try:
                sub = self.json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
                raw = str(sub.get("sic") or "")
                self._sics[cik] = int(raw) if raw.isdigit() else None
            except Exception:
                self._sics[cik] = None
        return self._sics[cik]

    def _company(self, cik: int) -> CompanyData | None:
        """Company data for one CIK (memoised: funds share many holdings)."""

        if cik in self._companies:
            return self._companies[cik]
        sic = self._sic(cik)
        found = company_from_frame(cik, self.frames().get(cik, {}), sic)
        if found is None:  # slow path: the company's full fact file, for filers missing frames
            try:
                facts = self.json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
                found = company_from_facts(cik, facts.get("facts", {}), sic)
            except Exception:  # foreign filers without XBRL facts, removed CIKs
                found = None
        if found is not None and (found.sic in REVIEW_SIC or found.sic == REIT_SIC):
            found = review_sector(found, self._industry(cik))
        self._companies[cik] = found
        return found

    def _industry(self, cik: int) -> str | None:
        ticker = self._cik_ticker.get(cik)
        if self.industry is None or ticker is None:
            return None
        try:
            return self.industry(ticker)
        except Exception:
            return None

    def enrich(self, holding: Holding) -> Holding:
        if self.figi is None or holding.figi_listings:
            return holding
        listings = self.figi.listings(holding.identifier)
        return replace(holding, figi_listings=listings) if listings else holding

    def resolve(self, holding: Holding) -> CompanyData | None:
        if holding in self._resolved:
            return self._resolved[holding]
        enriched = self.enrich(holding)
        found = self._resolve_sec(enriched)
        if found is None and self.fallback is not None:
            found = self.fallback(enriched)
        self._resolved[holding] = found
        return found

    def _resolve_sec(self, holding: Holding) -> CompanyData | None:
        tickers, names = self._company_maps()
        # A foreign holding's local ticker or name can collide with a different US company
        # (UCB, Richemont): match those only for US holdings; foreign ones need a FIGI
        # listing on a US exchange.
        domestic = holding.country in (None, "", "US")
        cik = tickers.get(holding.ticker) if holding.ticker and domestic else None
        for ticker, exchange in holding.figi_listings:  # identifier before name
            if cik is not None:
                break
            if exchange in US_EXCHANGES:
                cik = tickers.get(ticker.upper().replace("/", "-"))
        if cik is None and domestic:
            cik = names.get(normalize_name(holding.name))
        if cik is None:
            return None
        return self._company(cik)

    def prefetch(self, holdings: Sequence[Holding]) -> None:
        """Resolve every equity holding in parallel; :meth:`resolve` then answers from memory.

        Network requests still pass the shared rate limiter and the disk cache, so the pool
        only overlaps download time. The memo makes the result independent of thread order.
        """

        equity = [h for h in holdings if h.is_equity and h.value_usd > 0]
        tickers, _ = self._company_maps()
        self.frames()
        if self.figi is not None:
            idents = [
                h.identifier
                for h in equity
                if h.identifier is not None and not (h.ticker and h.ticker in tickers)
            ]
            self.figi.map_many([i for i in idents if i is not None])
        with ThreadPoolExecutor(max(1, self.workers)) as pool:
            list(pool.map(self.resolve, equity))


def screen_fund(client: SecClient, ticker: str, *, min_coverage: float = 0.90) -> ScreenResult:
    ids = client.fund_ids(ticker)
    if ids is None:
        return ScreenResult(ticker, reason="no N-PORT series on file (not a 1940-Act fund?)")
    holdings = client.latest_nport(*ids)
    if holdings is None:
        return ScreenResult(ticker, reason="no recent N-PORT-P filing found for this series")
    client.prefetch(holdings.holdings)
    return look_through(ticker, holdings, client.resolve, min_coverage=min_coverage)


def financials_json(f: CompanyFinancials) -> dict[str, Any]:
    return {
        "sector": f.sector,
        **{
            k: str(getattr(f, k).quantize(Decimal(1)))
            for k in (
                "market_cap",
                "interest_bearing_debt",
                "interest_bearing_cash_and_securities",
                "total_revenue",
                "noncompliant_revenue",
                "total_assets",
                "tangible_assets",
                "shares_outstanding",
            )
        },
    }


def write_config(
    path: Path, results: Sequence[ScreenResult], screened_at: date | None = None
) -> int:
    """Put each screened fund's financials into the config's universe; return how many."""

    config = json.loads(path.read_text("utf-8"))
    by_fund = {r.fund: r for r in results if r.financials is not None}
    stamp = (screened_at or datetime.now(UTC).date()).isoformat()
    written = 0
    for item in config.get("universe", []):
        result = by_fund.get(str(item.get("symbol", "")).upper())
        if result is not None and result.financials is not None:
            item["financials"] = financials_json(result.financials)
            item["financials_source"] = f"SEC N-PORT {result.report_date}, look-through"
            item["financials_screened_at"] = stamp
            written += 1
    path.write_text(json.dumps(config, indent=2) + "\n", "utf-8")
    return written


def stale_entries(config: Mapping[str, Any], max_age_days: int, today: date) -> list[str]:
    """Warnings for universe entries whose financials are older than ``max_age_days``."""

    out = []
    for item in config.get("universe", []):
        if "financials" not in item:
            continue
        symbol = item.get("symbol", "?")
        raw = item.get("financials_screened_at")
        try:
            screened = date.fromisoformat(str(raw)) if raw else None
        except ValueError:
            screened = None
        if screened is None:
            out.append(f"{symbol}: financials have no valid financials_screened_at date")
        elif (today - screened).days > max_age_days:
            out.append(
                f"{symbol}: financials screened {screened.isoformat()}, "
                f"{(today - screened).days} days old (max {max_age_days})"
            )
    return out


def main(argv: list[str] | None = None) -> int:
    from quant_os_core.shariah.rules import ShariahGate, ShariahProfile

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--user-agent", required=True, help='"Your Name you@example.com"')
    parser.add_argument("--config", type=Path, help="paper config to update with --write")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--cache", type=Path, default=Path("data/sec_cache"))
    parser.add_argument("--workers", type=int, default=8, help="parallel company fetches")
    parser.add_argument(
        "--max-age-days",
        type=int,
        help="warn when an existing config entry's financials are older than this",
    )
    parser.add_argument(
        "--no-yahoo", action="store_true", help="SEC data only (foreign holdings stay unresolved)"
    )
    parser.add_argument(
        "--no-openfigi", action="store_true", help="skip CUSIP/ISIN mapping through OpenFIGI"
    )
    args = parser.parse_args(argv)
    if (args.write or args.max_age_days is not None) and args.config is None:
        parser.error("--write and --max-age-days need --config")
    started = time.monotonic()
    if args.max_age_days is not None:
        existing = json.loads(args.config.read_text("utf-8"))
        today = datetime.now(UTC).date()
        for warning in stale_entries(existing, args.max_age_days, today):
            print(f"WARNING stale: {warning}")
    client = SecClient(args.user_agent, args.cache, workers=args.workers)
    if not args.no_openfigi:
        client.figi = OpenFigi(args.cache / "openfigi")
    if not args.no_yahoo:
        from quant_os_core.shariah.yahoo_fundamentals import YahooFundamentals

        yahoo = YahooFundamentals()
        client.fallback = yahoo.resolve
        client.industry = yahoo.industry
    gate = ShariahGate(ShariahProfile.spot_crypto_and_day_trading())
    results = []
    for ticker in args.tickers:
        fund_started = time.monotonic()
        r = screen_fund(client, ticker.upper(), min_coverage=args.min_coverage)
        results.append(r)
        took = f"[{time.monotonic() - fund_started:.0f}s]"
        if r.financials is None:
            print(f"{r.fund}: NOT SCREENED - {r.reason} {took}")
        else:
            f = r.financials
            verdict = gate.screen_company(f)
            outcome = "PASS" if verdict.allowed else f"FAIL {verdict.rule_id}: {verdict.reason}"
            print(
                f"{r.fund} ({r.report_date}, {r.holdings} holdings, {r.coverage:.0%} resolved): "
                f"debt/mcap {f.interest_bearing_debt / f.market_cap:.1%}, "
                f"cash/mcap {f.interest_bearing_cash_and_securities / f.market_cap:.1%}, "
                f"non-compliant revenue {f.noncompliant_revenue / f.total_revenue:.2%}, "
                f"tangible {f.tangible_assets / f.total_assets:.0%}, "
                f"excluded-sector weight {r.excluded_sector_weight:.1%} -> "
                f"{outcome} {took}"
            )
        for line in r.flagged:
            print(f"  sector: {line}")
        if r.unresolved:
            print(f"  unresolved ({len(r.unresolved)}): {', '.join(r.unresolved[:8])}")
    if not args.no_yahoo and yahoo.used:
        used = sorted(yahoo.used)
        print(f"Yahoo data used for {len(used)} holding(s); review: " + ", ".join(used[:12]))
    if args.write:
        print(f"wrote financials for {write_config(args.config, results)} fund(s) to {args.config}")
    print(f"done in {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
