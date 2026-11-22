from decimal import Decimal
from typing import Any

from quant_os_core.shariah.fund_screen import Holding
from quant_os_core.shariah.yahoo_fundamentals import YahooFundamentals, candidate_symbols


class _Frame:
    def __init__(self, rows: dict[str, float]) -> None:
        self.index = list(rows)
        self._rows = rows

    @property
    def loc(self) -> "_Frame":
        return self

    def __getitem__(self, name: str) -> "_Series":
        return _Series(self._rows[name])


class _Series:
    def __init__(self, value: float) -> None:
        self.iloc = [value]

    def dropna(self) -> "_Series":
        return self

    def __len__(self) -> int:
        return 1


def _stock(currency: str, reporting: str, industry: str = "Software") -> Any:
    class T:
        def __init__(self) -> None:
            self.info = {
                "currency": currency,
                "financialCurrency": reporting,
                "marketCap": 1_000,
                "sharesOutstanding": 10,
                "industry": industry,
            }
            self.balance_sheet = _Frame({"Total Assets": 500, "Total Debt": 70})
            self.financials = _Frame({"Total Revenue": 200})

    return T()


def _fx(rate: float) -> Any:
    class T:
        def __init__(self) -> None:
            self.info = {"regularMarketPrice": rate}

    return T()


def test_figi_listings_give_home_market_symbols_first() -> None:
    xiaomi = Holding(
        "Xiaomi Corp",
        Decimal(1),
        Decimal(1),
        "EC",
        ticker="1810",
        isin="KYG9830T1067",
        country="CN",
        figi_listings=(("XIACF", "US"), ("1810", "HK"), ("XIA", "GR")),
    )
    assert candidate_symbols(xiaomi) == ["KYG9830T1067", "1810.HK", "XIA.DE", "XIACF"]
    nestle = Holding(
        "Nestle", Decimal(1), Decimal(1), "EC", ticker="NESN", country="CH",
        figi_listings=(("NESN", "SE"), ("NESN", "SW")),
    )  # fmt: skip
    assert candidate_symbols(nestle) == ["NESN.SW"]  # duplicates dropped, order kept


def test_market_cap_converted_to_reporting_currency_at_spot_fx() -> None:
    tickers = {"CSU.TO": _stock("CAD", "USD"), "CADUSD=X": _fx(0.7)}
    yahoo = YahooFundamentals(tickers.__getitem__, retry_pause=0)
    holding = Holding("Constellation", Decimal(1), Decimal(1), "EC", ticker="CSU", country="CA")
    company = yahoo.resolve(holding)
    assert company is not None and company.market_cap == Decimal("700.0")
    # no FX quote: unresolved, never a guessed rate
    assert (
        YahooFundamentals({"CSU.TO": _stock("CAD", "USD")}.__getitem__, retry_pause=0).resolve(
            holding
        )
        is None
    )


def test_fx_lookup_inside_a_company_lookup_does_not_deadlock_the_slot_cap() -> None:
    import threading

    tickers = {"CSU.TO": _stock("CAD", "USD"), "CADUSD=X": _fx(0.7)}
    yahoo = YahooFundamentals(tickers.__getitem__, retry_pause=0, max_parallel=1)
    holding = Holding("Constellation", Decimal(1), Decimal(1), "EC", ticker="CSU", country="CA")
    out: list[object] = []
    worker = threading.Thread(target=lambda: out.append(yahoo.resolve(holding)), daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive() and out and out[0] is not None


def test_pence_quotes_need_no_fx_and_excluded_industries_keep_a_reason() -> None:
    tickers = {"HSBA.L": _stock("GBp", "GBP", "Banks - Diversified")}
    yahoo = YahooFundamentals(tickers.__getitem__, retry_pause=0)
    company = yahoo.resolve(Holding("HSBC", Decimal(1), Decimal(1), "EC", "HSBA", country="GB"))
    assert company is not None and company.market_cap == 1_000
    assert company.sector == "conventional_finance"
    assert company.reason == "Yahoo industry 'Banks - Diversified' -> conventional_finance"


def test_transient_errors_are_retried_and_industry_is_memoised() -> None:
    calls: list[str] = []

    def factory(symbol: str) -> Any:
        calls.append(symbol)
        if len(calls) == 1:
            raise TimeoutError("connect timeout")
        return _stock("USD", "USD", "REIT - Mortgage")

    yahoo = YahooFundamentals(factory, retry_pause=0)
    assert yahoo.industry("AGNC") == "REIT - Mortgage"
    assert yahoo.industry("AGNC") == "REIT - Mortgage"
    assert calls == ["AGNC", "AGNC"]  # one failure, one retry, then memoised
