"""Education-first, step-through backtester: practise reading charts one bar at a time.

Load daily bars from a ``data/ohlc`` CSV, reveal one bar at a time, and place hypothetical
long-only buys and sells. Every buy needs a stop below and a target above the price (the bot's
own rule 9) and there is one position at a time (no pyramiding). Each buy and sell pays a cost in
basis points of the price. The stop is checked before the target when one bar touches both, the
cautious assumption. Running statistics: trades, wins, win rate, average R, costs, return and the
worst drop of the practice account.

Sessions live in server memory only, capped in number and age. Nothing here imports a broker,
the order path or the state database: practice trades can never become real orders.
"""

from __future__ import annotations

import csv
import secrets
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any

from quant_os_core.ops.analytics import SYMBOL, available_symbols
from quant_os_core.ops.calculators import parse_decimal

CENT = Decimal("0.01")
QTY = Decimal("0.0001")
BPS = Decimal("10000")
MAX_SESSIONS = 32
SESSION_TTL = timedelta(hours=6)
WINDOW = 120  # revealed bars sent to the page
MIN_HISTORY = 30


@dataclass(frozen=True, slots=True)
class PracticeBar:
    time: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal

    def as_dict(self) -> dict[str, str]:
        return {
            "time": self.time,
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
        }


def _price(raw: str) -> Decimal:
    """CSV floats such as ``28.990398406982422`` -> ``28.9904``; ``100.0`` -> ``100``."""

    value = Decimal(raw).quantize(QTY)
    return Decimal(format(value.normalize(), "f"))


def load_bars(data_dir: Path, symbol: str) -> list[PracticeBar]:
    """Bars from ``<data_dir>/<symbol>.csv`` (time, high, low, close; open optional)."""

    if not SYMBOL.match(symbol) or symbol not in available_symbols(data_dir):
        raise ValueError("unknown symbol")
    with (data_dir / f"{symbol}.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: list[PracticeBar] = []
    previous: Decimal | None = None
    for row in rows:
        close = _price(row["close"])
        high, low = _price(row["high"]), _price(row["low"])
        raw_open = row.get("open")
        opened = _price(raw_open) if raw_open else (previous if previous is not None else close)
        opened = min(max(opened, low), high)
        out.append(PracticeBar(str(row["time"])[:10], opened, high, low, close))
        previous = close
    if len(out) < MIN_HISTORY + 2:
        raise ValueError(f"need at least {MIN_HISTORY + 2} bars to practise")
    return out


@dataclass
class Position:
    quantity: Decimal
    entry: Decimal  # price paid per unit before costs
    stop: Decimal
    target: Decimal
    cost_paid: Decimal
    opened_at: str


@dataclass
class Session:
    id: str
    symbol: str
    bars: list[PracticeBar]
    cursor: int  # index of the newest revealed bar
    cash: Decimal
    start_cash: Decimal
    cost_bps: Decimal
    position: Position | None = None
    trades: list[dict[str, Any]] = field(default_factory=list)
    equity: list[Decimal] = field(default_factory=list)
    costs: Decimal = Decimal("0")
    touched: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def bar(self) -> PracticeBar:
        return self.bars[self.cursor]

    @property
    def finished(self) -> bool:
        return self.cursor >= len(self.bars) - 1

    def _cost(self, notional: Decimal) -> Decimal:
        return (notional * self.cost_bps / BPS).quantize(CENT)

    def _mark(self) -> Decimal:
        held = self.position.quantity * self.bar.close if self.position else Decimal("0")
        return (self.cash + held).quantize(CENT)

    # ---- actions -------------------------------------------------------------------------
    def buy(self, stop: Decimal, target: Decimal, fraction: Decimal) -> None:
        if self.position is not None:
            raise ValueError("already holding: one position at a time (no pyramiding)")
        if self.finished:
            raise ValueError("no more bars; start a new session")
        price = self.bar.close
        if not stop < price < target:
            raise ValueError("long only: the stop must be below the price and the target above it")
        if not Decimal("0") < fraction <= Decimal("1"):
            raise ValueError("fraction of cash must be above 0 and at most 1")
        budget = self.cash * fraction
        unit_cost = price * (1 + self.cost_bps / BPS)
        quantity = (budget / unit_cost).quantize(QTY, rounding=ROUND_DOWN)
        if quantity <= 0:
            raise ValueError("not enough practice cash for one unit")
        cost = self._cost(quantity * price)
        self.cash -= quantity * price + cost
        self.costs += cost
        self.position = Position(quantity, price, stop, target, cost, self.bar.time)

    def _close(self, price: Decimal, reason: str) -> None:
        position = self.position
        if position is None:
            raise ValueError("nothing to sell")
        cost = self._cost(position.quantity * price)
        self.cash += position.quantity * price - cost
        self.costs += cost
        pnl = (price - position.entry) * position.quantity - position.cost_paid - cost
        risk = (position.entry - position.stop) * position.quantity
        self.trades.append(
            {
                "opened": position.opened_at,
                "closed": self.bar.time,
                "quantity": str(position.quantity),
                "entry": str(position.entry),
                "exit": str(price),
                "reason": reason,
                "pnl": str(pnl.quantize(CENT)),
                "r": str((pnl / risk).quantize(Decimal("0.01"))) if risk > 0 else None,
            }
        )
        self.position = None

    def sell(self) -> None:
        self._close(self.bar.close, "manual")

    def next(self) -> None:
        """Reveal one more bar, then fire the stop or target if the new bar reached it."""

        if self.finished:
            raise ValueError("no more bars; start a new session")
        self.cursor += 1
        position = self.position
        if position is not None:
            bar = self.bar
            if bar.low <= position.stop:
                self._close(min(position.stop, bar.open), "stop")
            elif bar.high >= position.target:
                self._close(max(position.target, bar.open), "target")
        self.equity.append(self._mark())

    # ---- view ----------------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        pnls = [Decimal(t["pnl"]) for t in self.trades]
        rs = [Decimal(t["r"]) for t in self.trades if t["r"] is not None]
        wins = sum(1 for p in pnls if p > 0)
        peak, worst = self.start_cash, Decimal("0")
        for value in self.equity:
            peak = max(peak, value)
            worst = min(worst, value / peak - 1)
        equity = self._mark()
        return {
            "trades": len(pnls),
            "wins": wins,
            "win_rate": None if not pnls else str((Decimal(wins) / len(pnls)).quantize(QTY)),
            "avg_r": None if not rs else str((sum(rs, Decimal("0")) / len(rs)).quantize(CENT)),
            "net_pnl": str(sum(pnls, Decimal("0")).quantize(CENT)),
            "costs": str(self.costs.quantize(CENT)),
            "equity": str(equity),
            "return_pct": str(((equity / self.start_cash - 1) * 100).quantize(CENT)),
            "max_drawdown_pct": str((worst * 100).quantize(CENT)),
        }

    def view(self) -> dict[str, Any]:
        """Only revealed bars are ever sent: the page cannot peek at the future."""

        first = max(0, self.cursor + 1 - WINDOW)
        position = self.position
        return {
            "session": self.id,
            "symbol": self.symbol,
            "bars": [b.as_dict() for b in self.bars[first : self.cursor + 1]],
            "revealed": self.cursor + 1,
            "remaining": len(self.bars) - 1 - self.cursor,
            "finished": self.finished,
            "price": str(self.bar.close),
            "cash": str(self.cash.quantize(CENT)),
            "cost_bps": str(self.cost_bps),
            "position": None
            if position is None
            else {
                "quantity": str(position.quantity),
                "entry": str(position.entry),
                "stop": str(position.stop),
                "target": str(position.target),
                "opened": position.opened_at,
            },
            "trades": self.trades[-50:],
            "stats": self.stats(),
        }


class ManualBacktester:
    """Server-side practice sessions keyed by an unguessable id."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def _evict(self, now: datetime) -> None:
        for sid in [s for s, sess in self._sessions.items() if now - sess.touched > SESSION_TTL]:
            del self._sessions[sid]
        while len(self._sessions) >= MAX_SESSIONS:
            self._sessions.popitem(last=False)

    def start(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        symbol = str(payload.get("symbol", "")).strip()
        bars = load_bars(self.data_dir, symbol)
        cash = parse_decimal(payload, "cash", "10000")
        cost_bps = parse_decimal(payload, "cost_bps", "10")
        if not Decimal("100") <= cash <= Decimal("100000000"):
            raise ValueError("cash must be between 100 and 100,000,000")
        if not Decimal("0") <= cost_bps <= Decimal("500"):
            raise ValueError("cost_bps must be between 0 and 500")
        latest_start = len(bars) - 2
        raw_start = payload.get("start")
        if raw_start in (None, ""):
            start = secrets.randbelow(latest_start - MIN_HISTORY + 1) + MIN_HISTORY
        else:
            start = int(parse_decimal(payload, "start"))
            if not MIN_HISTORY <= start <= latest_start:
                raise ValueError(f"start must be between {MIN_HISTORY} and {latest_start}")
        session = Session(secrets.token_urlsafe(16), symbol, bars, start, cash, cash, cost_bps)
        session.equity.append(session.cash)
        now = datetime.now(UTC)
        with self._lock:
            self._evict(now)
            self._sessions[session.id] = session
        return session.view()

    def act(self, session_id: str, action: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError("practice session not found or expired; start a new one")
            session.touched = datetime.now(UTC)
            self._sessions.move_to_end(session_id)
            if action == "next":
                steps = int(parse_decimal(payload, "steps", "1"))
                if not 1 <= steps <= 50:
                    raise ValueError("steps must be between 1 and 50")
                for _ in range(steps):
                    if session.finished:
                        break
                    session.next()
            elif action == "buy":
                session.buy(
                    parse_decimal(payload, "stop"),
                    parse_decimal(payload, "target"),
                    parse_decimal(payload, "fraction", "1"),
                )
            elif action == "sell":
                session.sell()
            elif action != "view":
                raise ValueError("action must be next, buy, sell or view")
            return session.view()
