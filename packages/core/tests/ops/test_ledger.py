import csv
import io
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from quant_os_core.ops.ledger import Ledger, noncompliant_ratios
from quant_os_core.ops.state import StateStore

T0 = datetime(2026, 3, 1, 15, tzinfo=UTC)
D = Decimal


def _ledger(tmp_path: Path) -> Ledger:
    path = tmp_path / "state.db"
    StateStore(path)  # same file as the bot's state, separate tables
    return Ledger(path)


def test_fifo_realized_pnl_with_fees(tmp_path: Path) -> None:
    led = _ledger(tmp_path)
    led.record_fill(T0, "spus", "buy", D("10"), D("40"), D("1"))
    led.record_fill(T0 + timedelta(days=1), "SPUS", "buy", D("10"), D("50"), D("1"))
    sell = led.record_fill(T0 + timedelta(days=2), "SPUS", "sell", D("15"), D("60"), D("1.5"))
    assert sell > 0
    pnl = led.realized_pnl()
    # cost: 10 x (401/10) + 5 x (501/10) = 401 + 250.5; proceeds 900 - 1.5
    assert pnl["by_symbol"] == {"SPUS": "247.00"}
    assert pnl["total"] == "247.00" and pnl["fees"] == "3.50"
    assert pnl["open"] == {"SPUS": {"quantity": "5", "cost": "250.50"}}
    assert led.fees_total() == D("3.50")
    assert led.open_quantity("spus") == D("5")
    assert [f.side for f in led.fills("SPUS")] == ["buy", "buy", "sell"]
    assert led.fills_by_id([sell])[0].quantity == D("15")


def test_selling_more_than_held_is_refused_as_short_selling(tmp_path: Path) -> None:
    led = _ledger(tmp_path)
    led.record_fill(T0, "HLAL", "buy", D("2"), D("50"))
    with pytest.raises(ValueError, match="short selling"):
        led.record_fill(T0, "HLAL", "sell", D("3"), D("55"))
    with pytest.raises(ValueError, match="short selling"):
        led.record_fill(T0, "OTHER", "sell", D("1"), D("1"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"side": "short"}, "side"),
        ({"quantity": D("0")}, "quantity"),
        ({"price": D("-1")}, "price"),
        ({"fee": D("-1")}, "fee"),
        ({"currency": "U$"}, "currency"),
        ({"symbol": " "}, "symbol"),
        ({"note": "x" * 501}, "note"),
        ({"when": datetime(2026, 1, 1)}, "timezone"),
    ],
)
def test_fill_validation(tmp_path: Path, kwargs: dict[str, object], message: str) -> None:
    led = _ledger(tmp_path)
    args: dict[str, object] = {
        "when": T0,
        "symbol": "SPUS",
        "side": "buy",
        "quantity": D("1"),
        "price": D("1"),
        "fee": D("0"),
        "currency": "USD",
        "note": "",
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        led.record_fill(**args)  # type: ignore[arg-type]


def test_cash_dividends_payments_and_csv_export(tmp_path: Path) -> None:
    led = _ledger(tmp_path)
    led.record_cash(T0, "deposit", D("1000"), "usd", "first")
    led.record_cash(T0 + timedelta(hours=1), "withdrawal", D("100"))
    led.record_fill(T0 + timedelta(hours=2), "SPUS", "buy", D("1"), D("40"), note="=cmd()")
    led.record_dividend(T0 + timedelta(days=5), "SPUS", D("12"))
    led.record_payment(T0 + timedelta(days=6), "purification", 2026, D("0.03"), symbol="SPUS")
    with pytest.raises(ValueError):
        led.record_cash(T0, "loan", D("1"))
    with pytest.raises(ValueError):
        led.record_dividend(T0, "SPUS", D("0"))
    with pytest.raises(ValueError):
        led.record_payment(T0, "charity", 2026, D("1"))
    with pytest.raises(ValueError):
        led.record_payment(T0, "zakat", 1800, D("1"))
    rows = list(csv.reader(io.StringIO(led.export_csv())))
    assert rows[0][0] == "type"
    assert [r[0] for r in rows[1:]] == ["cash", "cash", "fill", "dividend", "payment"]
    assert rows[1][8] == "USD" and rows[3][6] == "40"
    assert rows[3][10] == "'=cmd()"  # spreadsheet formula neutralised
    assert rows[5][9] == "2026"
    assert len(led.dividends(2026)) == 1 and led.dividends(2025) == []
    assert len(led.payments("purification", 2026)) == 1


def test_noncompliant_ratios_from_paper_config() -> None:
    config = {
        "universe": [
            {
                "symbol": "spus",
                "financials": {"total_revenue": "1000", "noncompliant_revenue": "25"},
            },
            {"symbol": "ZERO", "financials": {"total_revenue": "0", "noncompliant_revenue": "0"}},
            {"symbol": "BAD", "financials": {"total_revenue": "x", "noncompliant_revenue": "1"}},
            {"symbol": "NOFIN"},
            "junk",
        ]
    }
    assert noncompliant_ratios(config) == {"SPUS": D("0.025")}
    assert noncompliant_ratios({}) == {}


def test_purification_and_zakat_tracker(tmp_path: Path) -> None:
    led = _ledger(tmp_path)
    led.record_dividend(T0, "SPUS", D("1000"))
    led.record_dividend(T0 + timedelta(days=90), "SPUS", D("200"))
    led.record_dividend(T0, "MYSTERY", D("50"))
    led.record_dividend(datetime(2025, 6, 1, tzinfo=UTC), "SPUS", D("999"))
    led.record_payment(T0, "purification", 2026, D("10"), symbol="SPUS")
    led.record_payment(T0, "purification", 2026, D("1"))  # not tied to a fund
    out = led.purification(2026, {"SPUS": D("0.025")})
    spus = next(f for f in out["funds"] if f["symbol"] == "SPUS")
    assert spus == {
        "symbol": "SPUS",
        "dividends": "1200.00",
        "ratio": "0.025000",
        "purify": "30.00",
        "paid": "10.00",
        "outstanding": "20.00",
    }
    mystery = next(f for f in out["funds"] if f["symbol"] == "MYSTERY")
    assert mystery["ratio"] is None and mystery["purify"] is None
    assert out["missing_ratio"] == ["MYSTERY"]
    assert out["due"] == "30.00" and out["paid"] == "11.00" and out["outstanding"] == "19.00"

    led.record_payment(T0, "zakat", 2026, D("10000"), "PKR")
    z = led.zakat(2026, D("1500000"), nisab=D("200000"))
    assert z["due"] == "37500.00" and z["above_nisab"] is True
    assert z["paid"] == "10000.00" and z["outstanding"] == "27500.00"
    gold = led.zakat(2026, D("100"), gold_price_per_gram=D("20000"))
    assert gold["nisab"] == "1749600.00" and gold["due"] == "0.00" and gold["outstanding"] == "0.00"
    with pytest.raises(ValueError, match="nisab"):
        led.zakat(2026, D("1"))
    summary = led.summary(2026, {"SPUS": D("0.025")})
    assert summary["zakat_paid"] == "10000.00" and summary["year"] == 2026
    assert summary["deposits"] == "0.00" and summary["purification"]["due"] == "30.00"
