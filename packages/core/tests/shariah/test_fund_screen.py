import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

import pytest

from quant_os_core.shariah.fund_screen import (
    CompanyData,
    FundHoldings,
    Holding,
    SecClient,
    look_through,
    normalize_name,
    parse_nport,
    screen_fund,
    sector_for_sic,
    write_config,
)
from quant_os_core.shariah.rules import ShariahGate, ShariahProfile

NPORT = b"""<?xml version="1.0"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/nport">
 <formData>
  <genInfo><seriesId>S000099999</seriesId><repPdDate>2026-09-30</repPdDate></genInfo>
  <fundInfo><netAssets>1000000</netAssets></fundInfo>
  <invstOrSecs>
   <invstOrSec><name>Apple Inc.</name><cusip>037833100</cusip>
    <identifiers><ticker value="AAPL"/></identifiers>
    <balance>2000</balance><valUSD>600000</valUSD><assetCat>EC</assetCat></invstOrSec>
   <invstOrSec><name>Widget Corp</name><cusip>000000000</cusip>
    <balance>1000</balance><valUSD>380000</valUSD><assetCat>EC</assetCat></invstOrSec>
   <invstOrSec><name>US Treasury Bill</name><balance>20000</balance><valUSD>20000</valUSD>
    <assetCat>DBT</assetCat></invstOrSec>
  </invstOrSecs>
 </formData>
</edgarSubmission>"""


def _company(cik: int, **kw: object) -> CompanyData:
    base = dict(
        cik=cik,
        sic=3571,
        shares=Decimal("1000000"),
        debt=Decimal("10000000"),
        cash=Decimal("5000000"),
        revenue=Decimal("100000000"),
        interest_income=Decimal("1000000"),
        assets=Decimal("200000000"),
        tangible_assets=Decimal("150000000"),
    )
    base.update(kw)
    return CompanyData(**base)  # type: ignore[arg-type]


def test_parse_nport_reads_series_holdings_and_tickers() -> None:
    data = parse_nport(NPORT)
    assert data.series_id == "S000099999" and data.report_date == "2026-09-30"
    assert data.net_assets == Decimal("1000000") and len(data.holdings) == 3
    apple = data.holdings[0]
    assert apple.ticker == "AAPL" and apple.is_equity and apple.value_usd == Decimal("600000")
    assert data.holdings[1].ticker is None and not data.holdings[2].is_equity


def test_sic_mapping_and_name_normalising() -> None:
    assert sector_for_sic(6021) == "conventional_finance"
    assert sector_for_sic(2082) == "alcohol" and sector_for_sic(3571) is None
    assert normalize_name("Apple Inc.") == normalize_name("APPLE INC") == "APPLE"
    assert normalize_name("Procter & Gamble Co") == "PROCTER AND GAMBLE"


def test_look_through_aggregates_the_funds_share_of_each_company() -> None:
    data = parse_nport(NPORT)
    companies = {"Apple Inc.": _company(1), "Widget Corp": _company(2, sic=6021)}
    result = look_through("TEST", data, lambda h: companies.get(h.name))
    f = result.financials
    assert f is not None and result.coverage == pytest.approx(1.0)
    # Apple: price 300, mcap 300m, fund owns 0.2%; Widget: price 380, mcap 380m, owns 0.1%
    assert f.market_cap == Decimal("980000")
    assert f.interest_bearing_debt == pytest.approx(
        Decimal("0.002") * 10_000_000 + Decimal("0.001") * 10_000_000
    )
    assert f.interest_bearing_cash_and_securities == pytest.approx(
        Decimal("0.002") * 5_000_000 + Decimal("0.001") * 5_000_000 + 20_000  # T-bill held directly
    )
    # Widget is a bank: all of its revenue share is non-compliant; Apple adds interest income
    assert f.noncompliant_revenue == pytest.approx(
        Decimal("0.001") * 100_000_000 + Decimal("0.002") * 1_000_000
    )
    assert result.excluded_sector_weight == pytest.approx(380 / 980)
    verdict = ShariahGate(ShariahProfile.spot_crypto_and_day_trading()).screen_company(f)
    assert not verdict.allowed and verdict.rule_id == "R6"  # 39% bank weight fails revenue


def test_look_through_fails_closed_on_low_coverage_or_non_equity_funds() -> None:
    data = parse_nport(NPORT)
    partial = look_through("TEST", data, lambda h: _company(1) if h.ticker else None)
    assert partial.financials is None and "61%" in partial.reason
    assert partial.unresolved == ["Widget Corp"]
    sukuk = FundHoldings(
        "S1",
        "2026-09-30",
        Decimal("100"),
        (Holding("Sukuk A", Decimal("95"), Decimal("95"), "DBT"),),
    )
    assert "not an equity fund" in look_through("SPSK", sukuk, lambda h: None).reason


def _sec(tmp_path: Path) -> tuple[SecClient, list[Mapping[str, str]]]:
    tickers = {
        "0": {"cik_str": 1, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 2, "ticker": "WDGT", "title": "WIDGET CORP"},
    }
    facts = {
        "facts": {
            "us-gaap": {
                "Assets": {"units": {"USD": [{"val": 200000000, "end": "2026-06-30"}]}},
                "Revenues": {"units": {"USD": [{"val": 100000000, "end": "2026-06-30"}]}},
                "LongTermDebt": {"units": {"USD": [{"val": 1000000, "end": "2026-06-30"}]}},
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {"shares": [{"val": 1000000, "end": "2026-06-30"}]}
                }
            },
        }
    }
    pages: dict[str, bytes] = {
        "https://www.sec.gov/files/company_tickers_mf.json": json.dumps(
            {
                "fields": ["cik", "seriesId", "classId", "symbol"],
                "data": [[777, "S000099999", "C1", "SPUS"]],
            }
        ).encode(),
        "https://data.sec.gov/submissions/CIK0000000777.json": json.dumps(
            {
                "filings": {
                    "recent": {
                        "form": ["N-CSR", "NPORT-P"],
                        "accessionNumber": ["x", "0001-26-000001"],
                    }
                }
            }
        ).encode(),
        "https://www.sec.gov/Archives/edgar/data/777/000126000001/primary_doc.xml": NPORT,
        "https://www.sec.gov/files/company_tickers.json": json.dumps(tickers).encode(),
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json": json.dumps(facts).encode(),
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000002.json": json.dumps(facts).encode(),
        "https://data.sec.gov/submissions/CIK0000000001.json": b'{"sic": "3571"}',
        "https://data.sec.gov/submissions/CIK0000000002.json": b'{"sic": "3571"}',
    }
    seen: list[Mapping[str, str]] = []

    def fetch(url: str, headers: Mapping[str, str]) -> bytes:
        seen.append(headers)
        return pages[url]

    client = SecClient("Test test@example.com", tmp_path / "cache", fetch, min_interval=0)
    return client, seen


def test_screen_fund_end_to_end_with_cache_and_config_write(tmp_path: Path) -> None:
    client, seen = _sec(tmp_path)
    result = screen_fund(client, "SPUS")
    assert result.financials is not None and result.coverage == pytest.approx(1.0)
    assert result.report_date == "2026-09-30"
    assert (
        ShariahGate(ShariahProfile.spot_crypto_and_day_trading())
        .screen_company(result.financials)
        .allowed
    )
    assert all("@" in h["User-Agent"] for h in seen)
    calls = len(seen)
    screen_fund(client, "SPUS")
    assert len(seen) == calls  # second run is all cache
    config = tmp_path / "paper_core.json"
    config.write_text(
        json.dumps(
            {
                "universe": [
                    {"symbol": "SPUS", "asset_class": "etf"},
                    {"symbol": "AAAU", "asset_class": "etf"},
                ]
            }
        )
    )
    assert write_config(config, [result]) == 1
    written = json.loads(config.read_text())["universe"]
    assert (
        written[0]["financials"]["sector"] == "diversified_fund" and "financials" not in written[1]
    )
    assert screen_fund(client, "AAAU").financials is None  # no N-PORT series: stays blocked


def test_sec_client_requires_contact_email() -> None:
    with pytest.raises(ValueError, match="contact email"):
        SecClient("Quant OS research")


def test_frames_fast_path_builds_company_data_and_picks_filed_periods() -> None:
    from datetime import date

    from quant_os_core.shariah.fund_screen import company_from_frame, frame_requests

    requests = frame_requests(date(2026, 11, 22))
    assert requests[0][3] == ["CY2025Q4I", "CY2026Q1I", "CY2026Q2I"]
    assert requests[[r[1] for r in requests].index("EntityCommonStockSharesOutstanding")][:3] == (
        "dei",
        "EntityCommonStockSharesOutstanding",
        "shares",
    )
    assert requests[-1][3] == ["CY2024", "CY2025"]
    values = {
        "Assets": Decimal("200"),
        "Revenues": Decimal("100"),
        "EntityCommonStockSharesOutstanding": Decimal("10"),
        "LongTermDebt": Decimal("20"),
        "DebtCurrent": Decimal("5"),
        "Goodwill": Decimal("50"),
        "InvestmentIncomeInterest": Decimal("2"),
    }
    company = company_from_frame(9, values, 6021)
    assert company is not None and company.debt == 25 and company.tangible_assets == 150
    assert company.sector == "conventional_finance" and company.interest_income == 2
    assert company_from_frame(9, {"Assets": Decimal("1")}, None) is None


class _FakeFrame:
    def __init__(self, rows: dict[str, float]) -> None:
        self.index = list(rows)
        self._rows = rows

    @property
    def loc(self) -> "_FakeFrame":
        return self

    def __getitem__(self, name: str) -> "_FakeSeries":
        return _FakeSeries(self._rows[name])


class _FakeSeries:
    def __init__(self, value: float) -> None:
        self.iloc = [value]

    def dropna(self) -> "_FakeSeries":
        return self

    def __len__(self) -> int:
        return 1


class _FakeTicker:
    def __init__(self, symbol: str) -> None:
        if symbol != "MYL5227TO002":
            raise ValueError("unknown symbol")
        self.info = {
            "currency": "MYR",
            "financialCurrency": "MYR",
            "marketCap": 12_000,
            "sharesOutstanding": 3_000,
            "industry": "REIT - Retail",
        }
        self.balance_sheet = _FakeFrame(
            {"Total Assets": 6_000, "Total Debt": 1_200, "Cash And Cash Equivalents": 300}
        )
        self.financials = _FakeFrame({"Total Revenue": 600, "Interest Income": 6})


def test_yahoo_fallback_resolves_foreign_holdings_by_isin_without_fx() -> None:
    from quant_os_core.shariah.yahoo_fundamentals import (
        YahooFundamentals,
        candidate_symbols,
        sector_for_industry,
    )

    reit = Holding(
        "IGB Real Estate",
        Decimal("120"),
        Decimal("30"),
        "EC",
        ticker="IGBREIT",
        isin="MYL5227TO002",
        country="MY",
    )
    assert candidate_symbols(reit) == ["MYL5227TO002", "IGBREIT.KL"]
    assert candidate_symbols(Holding("C", Decimal(1), Decimal(1), "EC", ticker="CHC AU")) == [
        "CHC.AX"
    ]
    yahoo = YahooFundamentals(_FakeTicker)
    company = yahoo.resolve(reit)
    assert company is not None and company.market_cap == 12_000 and company.sector is None
    assert yahoo.used == ["IGB Real Estate (MYL5227TO002)"]
    # USD value 120 against a MYR 12,000 market cap: debt slice = 120 x 1,200 / 12,000 = 12 USD
    data = FundHoldings("S", "2026-09-30", Decimal("120"), (reit,))
    result = look_through("SPRE", data, yahoo.resolve)
    assert result.financials is not None
    assert result.financials.interest_bearing_debt == Decimal("12")
    assert sector_for_industry("Banks - Regional") == "conventional_finance"
    assert yahoo.resolve(Holding("X", Decimal(1), Decimal(1), "EC", ticker="ZZZ")) is None


def test_equity_reits_are_not_flagged_as_finance() -> None:
    assert sector_for_sic(6798) is None
