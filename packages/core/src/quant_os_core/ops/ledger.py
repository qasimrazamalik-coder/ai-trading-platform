"""Trade and fee ledger, plus the zakat and dividend-purification tracker.

Lives in the same SQLite file as :class:`~quant_os_core.ops.state.StateStore`, in its own
``ledger_*`` tables (created when missing). Records every fill (time, symbol, side, quantity,
price, fee, currency), cash deposits and withdrawals, dividends, and zakat or purification
payments. Money is ``Decimal`` stored as text; timestamps must be timezone-aware.

* Realized profit uses FIFO lots: a sell closes the oldest open buys first. Fees are added to
  the cost of a buy and taken from the proceeds of a sell.
* A sell larger than the open quantity is refused: short selling is blocked (GUARDRAILS 3).
* Zakat due for a year = 2.5% of zakatable wealth when it reaches the nisab
  (:func:`quant_os_core.ops.calculators.zakat`); the nisab is given in money or as gold grams x
  price per gram (:func:`~quant_os_core.ops.calculators.nisab_from_metal`).
* Purification for a fund = its dividends x non-compliant revenue / total revenue, read from the
  ``financials`` block of the paper config (:func:`noncompliant_ratios`).

This module keeps records only; it never sends an order.
"""

from __future__ import annotations

import csv
import io
import sqlite3
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from quant_os_core.ops import calculators

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger_fills (
    id INTEGER PRIMARY KEY, time TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,
    quantity TEXT NOT NULL, price TEXT NOT NULL, fee TEXT NOT NULL, currency TEXT NOT NULL,
    note TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger_cash (
    id INTEGER PRIMARY KEY, time TEXT NOT NULL, kind TEXT NOT NULL, amount TEXT NOT NULL,
    currency TEXT NOT NULL, note TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger_dividends (
    id INTEGER PRIMARY KEY, time TEXT NOT NULL, symbol TEXT NOT NULL, amount TEXT NOT NULL,
    currency TEXT NOT NULL, note TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger_payments (
    id INTEGER PRIMARY KEY, time TEXT NOT NULL, kind TEXT NOT NULL, year INTEGER NOT NULL,
    amount TEXT NOT NULL, currency TEXT NOT NULL, symbol TEXT NOT NULL, note TEXT NOT NULL
);
"""

SIDES = ("buy", "sell")
CASH_KINDS = ("deposit", "withdrawal")
PAYMENT_KINDS = ("zakat", "purification")
MAX_NOTE = 500
ZERO = Decimal("0")


def _iso(when: datetime) -> str:
    if when.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return when.astimezone(UTC).isoformat()


def _symbol(raw: str) -> str:
    symbol = raw.strip().upper()
    if not symbol or len(symbol) > 20:
        raise ValueError("symbol must be 1 to 20 characters")
    return symbol


def _currency(raw: str) -> str:
    code = raw.strip().upper()
    if not code.isalpha() or not 3 <= len(code) <= 5:
        raise ValueError("currency must be a 3 to 5 letter code")
    return code


def _note(raw: str) -> str:
    if len(raw) > MAX_NOTE:
        raise ValueError(f"note is longer than {MAX_NOTE} characters")
    return raw.strip()


def _positive(name: str, value: Decimal) -> Decimal:
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _non_negative(name: str, value: Decimal) -> Decimal:
    if not value.is_finite() or value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _year(year: int) -> int:
    if isinstance(year, bool) or not isinstance(year, int) or not 1900 <= year <= 2200:
        raise ValueError("year must be between 1900 and 2200")
    return year


def _money(value: Decimal) -> Decimal:
    return value.quantize(calculators.CENT)


def noncompliant_ratios(config: Mapping[str, Any]) -> dict[str, Decimal]:
    """Non-compliant revenue / total revenue per fund, from the paper config's ``financials``.

    Entries without financials or with a zero or missing total revenue are left out, so the
    caller can tell "unknown" from "zero".
    """

    out: dict[str, Decimal] = {}
    for row in config.get("universe") or []:
        if not isinstance(row, Mapping):
            continue
        fin = row.get("financials")
        if not isinstance(fin, Mapping):
            continue
        try:
            total = Decimal(str(fin["total_revenue"]))
            bad = Decimal(str(fin["noncompliant_revenue"]))
        except (KeyError, ArithmeticError, ValueError):
            continue
        if total.is_finite() and bad.is_finite() and total > 0 and bad >= 0:
            out[str(row.get("symbol", "")).upper()] = bad / total
    return out


@dataclass(frozen=True, slots=True)
class Fill:
    id: int
    time: datetime
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    fee: Decimal
    currency: str
    note: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "id": self.id,
            "time": self.time.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "quantity": str(self.quantity),
            "price": str(self.price),
            "fee": str(self.fee),
            "currency": self.currency,
            "note": self.note,
        }


class Ledger:
    """Append-only ledger. One connection per call, like ``StateStore``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(_SCHEMA)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    # ---- writes ----------------------------------------------------------------------------
    def record_fill(
        self,
        when: datetime,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        fee: Decimal = ZERO,
        currency: str = "USD",
        note: str = "",
    ) -> int:
        """Store one fill and return its id. A sell above the open quantity is refused."""

        sym = _symbol(symbol)
        if side not in SIDES:
            raise ValueError("side must be 'buy' or 'sell'")
        _positive("quantity", quantity)
        _positive("price", price)
        _non_negative("fee", fee)
        if side == "sell":
            held = self.open_quantity(sym)
            if quantity > held:
                raise ValueError(
                    f"cannot sell {quantity} {sym}: only {held} held (short selling is blocked)"
                )
        with self._db() as db:
            cursor = db.execute(
                "INSERT INTO ledger_fills(time, symbol, side, quantity, price, fee, currency, "
                "note) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _iso(when),
                    sym,
                    side,
                    str(quantity),
                    str(price),
                    str(fee),
                    _currency(currency),
                    _note(note),
                ),
            )
            return int(cursor.lastrowid or 0)

    def record_cash(
        self, when: datetime, kind: str, amount: Decimal, currency: str = "USD", note: str = ""
    ) -> int:
        if kind not in CASH_KINDS:
            raise ValueError("kind must be 'deposit' or 'withdrawal'")
        _positive("amount", amount)
        with self._db() as db:
            cursor = db.execute(
                "INSERT INTO ledger_cash(time, kind, amount, currency, note) "
                "VALUES (?, ?, ?, ?, ?)",
                (_iso(when), kind, str(amount), _currency(currency), _note(note)),
            )
            return int(cursor.lastrowid or 0)

    def record_dividend(
        self, when: datetime, symbol: str, amount: Decimal, currency: str = "USD", note: str = ""
    ) -> int:
        _positive("amount", amount)
        with self._db() as db:
            cursor = db.execute(
                "INSERT INTO ledger_dividends(time, symbol, amount, currency, note) "
                "VALUES (?, ?, ?, ?, ?)",
                (_iso(when), _symbol(symbol), str(amount), _currency(currency), _note(note)),
            )
            return int(cursor.lastrowid or 0)

    def record_payment(
        self,
        when: datetime,
        kind: str,
        year: int,
        amount: Decimal,
        currency: str = "USD",
        symbol: str = "",
        note: str = "",
    ) -> int:
        """A zakat or purification payment counted against ``year``'s amount due."""

        if kind not in PAYMENT_KINDS:
            raise ValueError("kind must be 'zakat' or 'purification'")
        _positive("amount", amount)
        sym = _symbol(symbol) if symbol.strip() else ""
        with self._db() as db:
            cursor = db.execute(
                "INSERT INTO ledger_payments(time, kind, year, amount, currency, symbol, note) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (_iso(when), kind, _year(year), str(amount), _currency(currency), sym, _note(note)),
            )
            return int(cursor.lastrowid or 0)

    # ---- reads -----------------------------------------------------------------------------
    def fills(self, symbol: str | None = None) -> list[Fill]:
        query = "SELECT * FROM ledger_fills"
        args: tuple[str, ...] = ()
        if symbol:
            query += " WHERE symbol=?"
            args = (_symbol(symbol),)
        with self._db() as db:
            rows = db.execute(query + " ORDER BY time, id", args).fetchall()
        return [
            Fill(
                int(i),
                datetime.fromisoformat(t),
                s,
                side,
                Decimal(q),
                Decimal(p),
                Decimal(f),
                c,
                n,
            )
            for i, t, s, side, q, p, f, c, n in rows
        ]

    def fills_by_id(self, ids: list[int]) -> list[Fill]:
        wanted = set(ids)
        return [f for f in self.fills() if f.id in wanted]

    def _rows(self, table: str) -> list[dict[str, Any]]:
        with self._db() as db:
            cursor = db.execute(f"SELECT * FROM {table} ORDER BY time, id")
            names = [c[0] for c in cursor.description]
            return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

    def cash_movements(self) -> list[dict[str, Any]]:
        return self._rows("ledger_cash")

    def dividends(self, year: int | None = None) -> list[dict[str, Any]]:
        rows = self._rows("ledger_dividends")
        if year is None:
            return rows
        return [r for r in rows if datetime.fromisoformat(r["time"]).year == _year(year)]

    def payments(self, kind: str | None = None, year: int | None = None) -> list[dict[str, Any]]:
        rows = self._rows("ledger_payments")
        return [
            r
            for r in rows
            if (kind is None or r["kind"] == kind) and (year is None or r["year"] == year)
        ]

    def open_quantity(self, symbol: str) -> Decimal:
        sym = _symbol(symbol)
        total = ZERO
        for fill in self.fills(sym):
            total += fill.quantity if fill.side == "buy" else -fill.quantity
        return total

    # ---- profit and loss -------------------------------------------------------------------
    def realized_pnl(self) -> dict[str, Any]:
        """FIFO realized profit per symbol, total fees and the open lots left."""

        lots: dict[str, deque[list[Decimal]]] = {}  # symbol -> [[qty, unit cost incl. fee]]
        realized: dict[str, Decimal] = {}
        fees = ZERO
        for fill in self.fills():
            fees += fill.fee
            book = lots.setdefault(fill.symbol, deque())
            if fill.side == "buy":
                book.append(
                    [fill.quantity, (fill.quantity * fill.price + fill.fee) / fill.quantity]
                )
                continue
            remaining = fill.quantity
            cost = ZERO
            while remaining > 0 and book:
                lot = book[0]
                take = min(lot[0], remaining)
                cost += take * lot[1]
                lot[0] -= take
                remaining -= take
                if lot[0] == 0:
                    book.popleft()
            proceeds = fill.quantity * fill.price - fill.fee
            realized[fill.symbol] = realized.get(fill.symbol, ZERO) + proceeds - cost
        open_lots = {
            sym: {
                "quantity": str(sum((lot[0] for lot in book), ZERO)),
                "cost": str(_money(sum((lot[0] * lot[1] for lot in book), ZERO))),
            }
            for sym, book in lots.items()
            if book
        }
        return {
            "by_symbol": {s: str(_money(v)) for s, v in sorted(realized.items())},
            "total": str(_money(sum(realized.values(), ZERO))),
            "fees": str(_money(fees)),
            "open": open_lots,
        }

    def fees_total(self) -> Decimal:
        return _money(sum((f.fee for f in self.fills()), ZERO))

    def export_csv(self) -> str:
        """Every fill, cash movement, dividend and payment as one CSV (oldest first)."""

        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(
            ["type", "time", "symbol", "side_or_kind", "quantity", "price", "amount", "fee",
             "currency", "year", "note"]
        )  # fmt: skip
        rows: list[tuple[str, int, list[str]]] = []
        for f in self.fills():
            amount = str(f.quantity * f.price)
            rows.append(
                (f.time.isoformat(), 0, ["fill", f.time.isoformat(), f.symbol, f.side,
                 str(f.quantity), str(f.price), amount, str(f.fee), f.currency, "", f.note])
            )  # fmt: skip
        for c in self.cash_movements():
            rows.append(
                (c["time"], 1, ["cash", c["time"], "", c["kind"], "", "", c["amount"], "",
                 c["currency"], "", c["note"]])
            )  # fmt: skip
        for d in self.dividends():
            rows.append(
                (d["time"], 2, ["dividend", d["time"], d["symbol"], "", "", "", d["amount"], "",
                 d["currency"], "", d["note"]])
            )  # fmt: skip
        for p in self.payments():
            rows.append(
                (p["time"], 3, ["payment", p["time"], p["symbol"], p["kind"], "", "", p["amount"],
                 "", p["currency"], str(p["year"]), p["note"]])
            )  # fmt: skip
        for _, _, row in sorted(rows, key=lambda r: (r[0], r[1])):
            writer.writerow([_csv_safe(v) for v in row])
        return out.getvalue()

    # ---- zakat and purification ------------------------------------------------------------
    def _paid(self, kind: str, year: int, symbol: str | None = None) -> Decimal:
        return sum(
            (
                Decimal(p["amount"])
                for p in self.payments(kind, year)
                if symbol is None or p["symbol"] == symbol
            ),
            ZERO,
        )

    def zakat(
        self,
        year: int,
        zakatable_wealth: Decimal,
        *,
        nisab: Decimal | None = None,
        gold_price_per_gram: Decimal | None = None,
        nisab_grams: Decimal = calculators.NISAB_GOLD_GRAMS,
    ) -> dict[str, Any]:
        """Zakat due for ``year``, what was paid against it and what is still outstanding.

        Give ``nisab`` in money, or ``gold_price_per_gram`` (nisab = grams x price).
        """

        _year(year)
        if nisab is None:
            if gold_price_per_gram is None:
                raise ValueError("give the nisab or the gold price per gram")
            nisab = calculators.nisab_from_metal(gold_price_per_gram, nisab_grams)
        due = calculators.zakat(zakatable_wealth, nisab)
        paid = self._paid("zakat", year)
        return {
            "year": year,
            "zakatable_wealth": str(_money(zakatable_wealth)),
            "nisab": str(due["nisab"]),
            "above_nisab": due["above_nisab"],
            "due": str(due["due"]),
            "paid": str(_money(paid)),
            "outstanding": str(_money(max(due["due"] - paid, ZERO))),
        }

    def purification(self, year: int, ratios: Mapping[str, Decimal]) -> dict[str, Any]:
        """Per fund: dividends in ``year`` x non-compliant revenue ratio, paid, outstanding.

        A fund with no ratio is listed with ``ratio: None`` and nothing computed, so a missing
        ``financials`` block shows up instead of silently counting as zero.
        """

        _year(year)
        totals: dict[str, Decimal] = {}
        for row in self.dividends(year):
            totals[row["symbol"]] = totals.get(row["symbol"], ZERO) + Decimal(row["amount"])
        funds = []
        due_total = paid_total = ZERO
        unknown: list[str] = []
        for symbol in sorted(totals):
            ratio = ratios.get(symbol)
            paid = self._paid("purification", year, symbol)
            paid_total += paid
            if ratio is None:
                unknown.append(symbol)
                funds.append(
                    {
                        "symbol": symbol,
                        "dividends": str(_money(totals[symbol])),
                        "ratio": None,
                        "purify": None,
                        "paid": str(_money(paid)),
                        "outstanding": None,
                    }
                )
                continue
            result = calculators.purification(totals[symbol], ratio * calculators.HUNDRED)
            due_total += result["purify"]
            funds.append(
                {
                    "symbol": symbol,
                    "dividends": str(_money(totals[symbol])),
                    "ratio": str(ratio.quantize(Decimal("0.000001"))),
                    "purify": str(result["purify"]),
                    "paid": str(_money(paid)),
                    "outstanding": str(_money(max(result["purify"] - paid, ZERO))),
                }
            )
        unassigned = self._paid("purification", year, "")
        paid_total += unassigned
        return {
            "year": year,
            "funds": funds,
            "due": str(_money(due_total)),
            "paid": str(_money(paid_total)),
            "outstanding": str(_money(max(due_total - paid_total, ZERO))),
            "missing_ratio": unknown,
        }

    def summary(self, year: int, ratios: Mapping[str, Decimal]) -> dict[str, Any]:
        """What ``GET /api/ledger`` returns."""

        deposits = sum(
            (Decimal(c["amount"]) for c in self.cash_movements() if c["kind"] == "deposit"), ZERO
        )
        withdrawals = sum(
            (Decimal(c["amount"]) for c in self.cash_movements() if c["kind"] == "withdrawal"),
            ZERO,
        )
        return {
            "fills": [f.as_dict() for f in reversed(self.fills())][:200],
            "cash": list(reversed(self.cash_movements()))[:100],
            "dividends": list(reversed(self.dividends()))[:100],
            "payments": list(reversed(self.payments()))[:100],
            "pnl": self.realized_pnl(),
            "deposits": str(_money(deposits)),
            "withdrawals": str(_money(withdrawals)),
            "purification": self.purification(year, ratios),
            "zakat_paid": str(_money(self._paid("zakat", year))),
            "year": year,
        }


def _csv_safe(value: str) -> str:
    """Neutralise spreadsheet formulas in free text (CSV injection)."""

    return "'" + value if value[:1] in ("=", "+", "-", "@") and not _is_number(value) else value


def _is_number(value: str) -> bool:
    try:
        Decimal(value)
    except ArithmeticError:
        return False
    return True


def record_fill_safely(
    ledger: Ledger | None,
    log: Callable[[str, str, str], None],
    when: datetime,
    symbol: str,
    side: str,
    quantity: Decimal,
    price: Decimal | None,
) -> None:
    """Record a broker fill from a trading loop; a ledger problem is logged, never raised.

    The ledger is bookkeeping: it must not stop the loop from managing real positions.
    """

    if ledger is None:
        return
    if price is None or price <= 0 or quantity <= 0:
        log("warning", "ledger", f"fill not recorded: {side} {quantity} {symbol} @ {price}")
        return
    try:
        ledger.record_fill(when, symbol, side, quantity, price, note="auto: broker fill")
    except Exception as exc:
        log("warning", "ledger", f"fill not recorded: {side} {quantity} {symbol}: {exc!r}")
