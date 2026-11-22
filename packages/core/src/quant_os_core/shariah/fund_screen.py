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

SEC asks every client to send a User-Agent with a contact email and to stay under ten
requests per second; responses are cached on disk to keep the request count low.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from quant_os_core.market_data.free_sources import _TAGS, _latest, _sum, urllib_fetch
from quant_os_core.shariah.rules import CompanyFinancials

Fetch = Callable[[str, Mapping[str, str]], bytes]
FUND_SECTOR = "diversified_fund"
ZERO = Decimal(0)

# SIC code ranges -> excluded sector names used by the Shariah gate (R7)
SIC_SECTORS: tuple[tuple[int, int, str], ...] = (
    (2080, 2086, "alcohol"),
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

    @property
    def is_equity(self) -> bool:
        return self.asset_category in ("EC", "EP")


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

    @property
    def sector(self) -> str | None:
        return sector_for_sic(self.sic) or self.industry_sector


def company_from_facts(cik: int, facts: Mapping[str, Any], sic: int | None) -> CompanyData | None:
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
        if company.sector is not None:
            excluded += h.value_usd
            noncompliant += share * company.revenue
        else:
            noncompliant += share * min(company.interest_income, company.revenue)
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


# ---------------------------------------------------------------- SEC client


@dataclass(slots=True)
class SecClient:
    user_agent: str
    cache_dir: Path = Path("data/sec_cache")
    fetch: Fetch = urllib_fetch
    min_interval: float = 0.12  # stay under SEC's ten requests per second
    _last: float = 0.0
    _tickers: dict[str, int] | None = None
    _names: dict[str, int] | None = None
    _frames: dict[int, dict[str, Decimal]] | None = None
    fallback: Callable[[Holding], CompanyData | None] | None = (
        None  # e.g. Yahoo, for non-SEC filers
    )
    today: date = field(default_factory=date.today)

    def __post_init__(self) -> None:
        if "@" not in self.user_agent:
            raise ValueError("SEC requires a User-Agent with a contact email")

    def get(self, url: str) -> bytes:
        key = re.sub(r"[^A-Za-z0-9._-]", "_", url.split("://", 1)[-1])[-180:]
        path = self.cache_dir / key
        if path.exists():
            return path.read_bytes()
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        body = self.fetch(url, {"User-Agent": self.user_agent, "Accept-Encoding": "identity"})
        self._last = time.monotonic()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return body

    def json(self, url: str) -> Any:
        return json.loads(self.get(url))

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
        if self._tickers is None or self._names is None:
            raw = self.json("https://www.sec.gov/files/company_tickers.json")
            self._tickers = {str(v["ticker"]).upper(): int(v["cik_str"]) for v in raw.values()}
            self._names = {}
            for v in raw.values():
                self._names.setdefault(normalize_name(str(v["title"])), int(v["cik_str"]))
        return self._tickers, self._names

    def frames(self) -> dict[int, dict[str, Decimal]]:
        """Every company's latest value of each screening tag, from SEC XBRL frames.

        One request returns a tag for all filers, so about 45 requests cover every holding.
        Balance-sheet tags use the latest quarter-ends (newest wins); income tags use the last
        full calendar year.
        """

        if self._frames is not None:
            return self._frames
        out: dict[int, dict[str, Decimal]] = {}
        for taxonomy, tag, unit, periods in frame_requests(self.today):
            for period in periods:  # oldest first, so newer values overwrite
                url = f"https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json"
                try:
                    rows = self.json(url).get("data", [])
                except Exception:  # a frame not published yet
                    continue
                for row in rows:
                    out.setdefault(int(row["cik"]), {})[tag] = Decimal(str(row["val"]))
        self._frames = out
        return out

    def _sic(self, cik: int) -> int | None:
        try:
            sub = self.json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        except Exception:
            return None
        raw = str(sub.get("sic") or "")
        return int(raw) if raw.isdigit() else None

    def resolve(self, holding: Holding) -> CompanyData | None:
        found = self._resolve_sec(holding)
        if found is None and self.fallback is not None:
            found = self.fallback(holding)
        return found

    def _resolve_sec(self, holding: Holding) -> CompanyData | None:
        tickers, names = self._company_maps()
        cik = (tickers.get(holding.ticker) if holding.ticker else None) or names.get(
            normalize_name(holding.name)
        )
        if cik is None:
            return None
        fast = company_from_frame(cik, self.frames().get(cik, {}), self._sic(cik))
        if fast is not None:
            return fast
        try:  # slow path: the company's full fact file, for filers missing from the frames
            facts = self.json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
        except Exception:  # foreign filers without XBRL facts, removed CIKs
            return None
        return company_from_facts(cik, facts.get("facts", {}), self._sic(cik))


def screen_fund(client: SecClient, ticker: str, *, min_coverage: float = 0.90) -> ScreenResult:
    ids = client.fund_ids(ticker)
    if ids is None:
        return ScreenResult(ticker, reason="no N-PORT series on file (not a 1940-Act fund?)")
    holdings = client.latest_nport(*ids)
    if holdings is None:
        return ScreenResult(ticker, reason="no recent N-PORT-P filing found for this series")
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


def write_config(path: Path, results: Sequence[ScreenResult]) -> int:
    """Put each screened fund's financials into the config's universe; return how many."""

    config = json.loads(path.read_text("utf-8"))
    by_fund = {r.fund: r for r in results if r.financials is not None}
    written = 0
    for item in config.get("universe", []):
        result = by_fund.get(str(item.get("symbol", "")).upper())
        if result is not None and result.financials is not None:
            item["financials"] = financials_json(result.financials)
            item["financials_source"] = f"SEC N-PORT {result.report_date}, look-through"
            written += 1
    path.write_text(json.dumps(config, indent=2) + "\n", "utf-8")
    return written


def main(argv: list[str] | None = None) -> int:
    from quant_os_core.shariah.rules import ShariahGate, ShariahProfile

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--user-agent", required=True, help='"Your Name you@example.com"')
    parser.add_argument("--config", type=Path, help="paper config to update with --write")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--cache", type=Path, default=Path("data/sec_cache"))
    parser.add_argument(
        "--no-yahoo", action="store_true", help="SEC data only (foreign holdings stay unresolved)"
    )
    args = parser.parse_args(argv)
    client = SecClient(args.user_agent, args.cache)
    if not args.no_yahoo:
        from quant_os_core.shariah.yahoo_fundamentals import YahooFundamentals

        yahoo = YahooFundamentals()
        client.fallback = yahoo.resolve
    gate = ShariahGate(ShariahProfile.spot_crypto_and_day_trading())
    results = []
    for ticker in args.tickers:
        r = screen_fund(client, ticker.upper(), min_coverage=args.min_coverage)
        results.append(r)
        if r.financials is None:
            print(f"{r.fund}: NOT SCREENED - {r.reason}")
            continue
        f = r.financials
        verdict = gate.screen_company(f)
        print(
            f"{r.fund} ({r.report_date}, {r.holdings} holdings, {r.coverage:.0%} resolved): "
            f"debt/mcap {f.interest_bearing_debt / f.market_cap:.1%}, "
            f"cash/mcap {f.interest_bearing_cash_and_securities / f.market_cap:.1%}, "
            f"non-compliant revenue {f.noncompliant_revenue / f.total_revenue:.2%}, "
            f"tangible {f.tangible_assets / f.total_assets:.0%}, "
            f"excluded-sector weight {r.excluded_sector_weight:.1%} -> "
            f"{'PASS' if verdict.allowed else 'FAIL ' + verdict.rule_id + ': ' + verdict.reason}"
        )
        if r.unresolved:
            print(f"  unresolved ({len(r.unresolved)}): {', '.join(r.unresolved[:8])}")
    if not args.no_yahoo and yahoo.used:
        print(
            f"Yahoo data used for {len(yahoo.used)} holding(s); review: "
            + ", ".join(yahoo.used[:12])
        )
    if args.write:
        if args.config is None:
            parser.error("--write needs --config")
        print(f"wrote financials for {write_config(args.config, results)} fund(s) to {args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
