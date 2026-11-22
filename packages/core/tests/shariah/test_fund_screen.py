import json
from collections.abc import Mapping
from dataclasses import replace
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


def test_rate_limiter_spaces_requests_across_threads() -> None:
    import threading

    from quant_os_core.shariah.fund_screen import RateLimiter

    sleeps: list[float] = []
    limiter = RateLimiter(0.1, clock=lambda: 0.0, sleep=sleeps.append)
    threads = [threading.Thread(target=limiter.wait) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # one call goes at once; the rest each get their own later slot, never sharing one
    assert sorted(round(s, 6) for s in sleeps) == [0.1, 0.2, 0.3, 0.4]


def test_parallel_prefetch_matches_serial_and_fetches_each_url_once(tmp_path: Path) -> None:
    serial, _ = _sec(tmp_path / "a")
    serial.workers = 1
    parallel, _ = _sec(tmp_path / "b")
    parallel.workers = 8
    urls: list[str] = []
    inner = parallel.fetch

    def counting(url: str, headers: Mapping[str, str]) -> bytes:
        urls.append(url)
        return inner(url, headers)

    parallel.fetch = counting
    one, many = screen_fund(serial, "SPUS"), screen_fund(parallel, "SPUS")
    assert one.financials == many.financials and one.unresolved == many.unresolved
    assert len(urls) == len(set(urls))


class _FakeFigi:
    def __init__(self, answers: dict[str, object]) -> None:
        self.answers = answers
        self.bodies: list[list[dict[str, str]]] = []

    def __call__(self, url: str, body: bytes, headers: Mapping[str, str]) -> bytes:
        jobs = json.loads(body)
        self.bodies.append(jobs)
        answers = [self.answers.get(j["idValue"], {"warning": "none"}) for j in jobs]
        return json.dumps(answers).encode()


def test_openfigi_batches_caches_and_skips_errors(tmp_path: Path) -> None:
    from quant_os_core.shariah.fund_screen import OpenFigi

    listing = {
        "data": [
            {"ticker": "WDGT", "exchCode": "US", "marketSector": "Equity"},
            {"ticker": "WDGT", "exchCode": "UN", "marketSector": "Equity"},
            {"ticker": "WDGT 5 01/30", "exchCode": "TRACE", "marketSector": "Corp"},
        ]
    }
    post = _FakeFigi({"111111111": listing, "222222222": {"error": "busy"}})
    figi = OpenFigi(tmp_path / "figi", post, min_interval=0, sleep=lambda s: None)
    idents = [("ID_CUSIP", f"{i}" * 9) for i in range(1, 10)] + [("ID_ISIN", "US0000000001")] * 3
    assert figi.map_many(idents) == 1  # 10 distinct identifiers -> one request of 10 jobs
    assert len(post.bodies[0]) == 10
    assert figi.listings(("ID_CUSIP", "111111111")) == (("WDGT", "US"), ("WDGT", "UN"))
    assert figi.listings(("ID_CUSIP", "333333333")) == ()  # not found: cached as empty
    fresh = OpenFigi(tmp_path / "figi", post, min_interval=0, sleep=lambda s: None)
    assert fresh.map_many(idents) == 1  # only the errored identifier is asked again
    assert post.bodies[1] == [{"idType": "ID_CUSIP", "idValue": "222222222"}]


def test_openfigi_retries_after_rate_limit(tmp_path: Path) -> None:
    import urllib.error

    from quant_os_core.shariah.fund_screen import OpenFigi

    calls: list[int] = []

    def post(url: str, body: bytes, headers: Mapping[str, str]) -> bytes:
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)  # type: ignore[arg-type]
        return b'[{"data": [{"ticker": "AAPL", "exchCode": "US", "marketSector": "Equity"}]}]'

    waits: list[float] = []
    figi = OpenFigi(tmp_path, post, min_interval=0, sleep=waits.append)
    figi.map_many([("ID_CUSIP", "037833100")])
    assert len(calls) == 2 and waits and figi.listings(("ID_CUSIP", "037833100"))


def test_holding_identifier_prefers_isin_and_ignores_placeholder_cusips() -> None:
    h = Holding("A", Decimal(1), Decimal(1), "EC", cusip="000000000", isin="JP3571400005")
    assert h.identifier == ("ID_ISIN", "JP3571400005")
    assert Holding("B", Decimal(1), Decimal(1), "EC", cusip="000000000").identifier is None
    assert Holding("C", Decimal(1), Decimal(1), "EC", cusip="N/A").identifier is None
    assert Holding("D", Decimal(1), Decimal(1), "EC", cusip="037833100").identifier == (
        "ID_CUSIP",
        "037833100",
    )


def test_cusip_resolves_through_openfigi_before_the_name(tmp_path: Path) -> None:
    from quant_os_core.shariah.fund_screen import OpenFigi

    client, _ = _sec(tmp_path)
    post = _FakeFigi(
        {"123456789": {"data": [{"ticker": "WDGT", "exchCode": "US", "marketSector": "Equity"}]}}
    )
    client.figi = OpenFigi(tmp_path / "figi", post, min_interval=0, sleep=lambda s: None)
    odd_name = Holding("Widget Hldgs Cl A (unlisted name)", Decimal(10), Decimal(1), "EC")
    assert client.resolve(odd_name) is None  # no ticker, no identifier, name does not match
    by_cusip = replace(odd_name, cusip="123456789")
    client.prefetch([by_cusip])
    company = client.resolve(by_cusip)
    assert company is not None and company.cik == 2


def test_review_by_hand_sics_are_confirmed_or_cleared_by_yahoo_industry() -> None:
    from quant_os_core.shariah.fund_screen import review_sector

    hotel = _company(5, sic=7011)
    assert hotel.sector == "gambling"
    cleared = review_sector(hotel, "Lodging")
    assert cleared.sector is None and "cleared by Yahoo industry 'Lodging'" in cleared.reason
    casino = review_sector(hotel, "Resorts & Casinos")
    assert casino.sector == "gambling" and "confirmed" in casino.reason
    unknown = review_sector(hotel, None)
    assert unknown.sector == "gambling" and "kept" in unknown.reason
    meat = review_sector(_company(6, sic=2011), "Packaged Foods")
    assert meat.sector == "pork" and "cannot rule it out" in meat.reason  # fails closed
    mreit = review_sector(_company(7, sic=6798), "REIT - Mortgage")
    assert mreit.sector == "conventional_finance" and "REIT - Mortgage" in mreit.reason
    assert review_sector(_company(8, sic=6798), "REIT - Retail").sector is None


def test_sec_client_checks_industry_and_look_through_records_reasons(tmp_path: Path) -> None:
    client, _ = _sec(tmp_path)
    client.industry = {"AAPL": "Lodging", "WDGT": "REIT - Mortgage"}.get
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "data.sec.gov_submissions_CIK0000000001.json").write_bytes(b'{"sic": "7011"}')
    (cache / "data.sec.gov_submissions_CIK0000000002.json").write_bytes(b'{"sic": "6798"}')
    result = screen_fund(client, "SPUS")
    assert result.excluded_sector_weight == pytest.approx(380 / 980)
    assert any("Apple Inc." in f and "not excluded" in f and "Lodging" in f for f in result.flagged)
    assert any(
        "Widget Corp" in f and "conventional_finance" in f and "REIT - Mortgage" in f
        for f in result.flagged
    )


def test_write_config_stamps_screened_date_and_stale_entries_warn(tmp_path: Path) -> None:
    from datetime import date

    from quant_os_core.shariah.fund_screen import stale_entries

    client, _ = _sec(tmp_path)
    result = screen_fund(client, "SPUS")
    config = tmp_path / "c.json"
    config.write_text(json.dumps({"universe": [{"symbol": "SPUS"}, {"symbol": "HLAL"}]}))
    write_config(config, [result], screened_at=date(2026, 9, 1))
    data = json.loads(config.read_text())
    entry = data["universe"][0]
    assert entry["financials_screened_at"] == "2026-09-01"
    assert entry["financials_source"] == "SEC N-PORT 2026-09-30, look-through"
    assert stale_entries(data, 90, date(2026, 11, 22)) == []
    assert stale_entries(data, 30, date(2026, 11, 22)) == [
        "SPUS: financials screened 2026-09-01, 82 days old (max 30)"
    ]
    data["universe"][1]["financials"] = {"sector": "x"}
    assert "HLAL: financials have no valid" in stale_entries(data, 30, date(2026, 11, 22))[1]


def test_foreign_holdings_do_not_match_us_tickers_or_names(tmp_path: Path) -> None:
    client, _ = _sec(tmp_path)
    foreign = Holding("Widget Corp", Decimal("1"), Decimal("1"), "EC", ticker="WDGT", country="BE")
    assert client._resolve_sec(foreign) is None  # same ticker and name as a US filer
    domestic = Holding("Widget Corp", Decimal("1"), Decimal("1"), "EC", ticker="WDGT", country="US")
    assert client._resolve_sec(domestic) is not None


def test_company_facts_ignore_non_usd_money() -> None:
    from quant_os_core.shariah.fund_screen import company_from_facts

    def node(unit: str, val: int) -> dict[str, object]:
        return {"units": {unit: [{"val": val, "end": "2026-06-30"}]}}

    yen = {
        "us-gaap": {
            "Assets": node("JPY", 9_000_000),
            "Revenues": node("JPY", 5_000_000),
            "LongTermDebt": node("JPY", 4_000_000),
        },
        "dei": {"EntityCommonStockSharesOutstanding": node("shares", 100)},
    }
    assert company_from_facts(1, yen, None) is None  # yen figures are never read as dollars
    usd = {
        "us-gaap": {
            "Assets": node("USD", 900),
            "Revenues": node("USD", 500),
            "LongTermDebt": node("USD", 40),
        },
        "dei": {"EntityCommonStockSharesOutstanding": node("shares", 100)},
    }
    company = company_from_facts(1, usd, None)
    assert company is not None and company.debt == 40
