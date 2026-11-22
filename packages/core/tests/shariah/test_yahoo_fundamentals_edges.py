"""Yahoo fallback edge cases: bad numbers, currency mismatches and failing lookups fail closed."""

import sys
import types
from decimal import Decimal
from typing import Any

import pytest

from quant_os_core.shariah.fund_screen import Holding
from quant_os_core.shariah.yahoo_fundamentals import YahooFundamentals, _dec

FOREIGN = Holding(
    "Foreign Co", Decimal("10"), Decimal("5"), "EC", ticker="FCO", isin="GB0000000001", country="GB"
)


class _EmptyFrame:
    index: tuple[str, ...] = ()


class _Ticker:
    def __init__(self, info: dict[str, Any]) -> None:
        self.info = info
        self.balance_sheet = _EmptyFrame()
        self.financials = _EmptyFrame()


@pytest.mark.parametrize("raw", [None, "not-a-number", "NaN", float("inf")])
def test_dec_rejects_missing_and_non_finite_values(raw: object) -> None:
    assert _dec(raw) is None


def test_dec_accepts_numbers() -> None:
    assert _dec(12.5) == Decimal("12.5")


def test_mismatched_reporting_currency_is_not_resolved() -> None:
    info = {"currency": "GBp", "financialCurrency": "USD"}
    yahoo = YahooFundamentals(lambda symbol: _Ticker(info))
    assert yahoo.resolve(FOREIGN) is None
    assert yahoo.used == []


def test_missing_market_cap_is_not_resolved() -> None:
    yahoo = YahooFundamentals(lambda symbol: _Ticker({"currency": "GBP", "marketCap": None}))
    assert yahoo.resolve(FOREIGN) is None


def test_lookup_errors_try_every_symbol_then_give_up() -> None:
    tried: list[str] = []

    def failing(symbol: str) -> Any:
        tried.append(symbol)
        raise ConnectionError("yahoo down")

    yahoo = YahooFundamentals(failing)
    assert yahoo.resolve(FOREIGN) is None
    assert tried == ["GB0000000001", "FCO.L"]


def test_default_ticker_factory_is_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda symbol: f"ticker:{symbol}"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    yahoo = YahooFundamentals()
    assert yahoo._ticker("AAPL") == "ticker:AAPL"
    assert yahoo.ticker_factory is fake.Ticker  # type: ignore[attr-defined]
