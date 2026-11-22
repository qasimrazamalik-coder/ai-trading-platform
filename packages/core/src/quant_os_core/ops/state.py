"""Durable trading state (SQLite, WAL): survives crashes and restarts exactly.

Holds open positions (with their exit plan), pending entry orders, closed trades, every
pipeline decision, equity snapshots, operator events, model versions and switches such as the
kill switch. One connection per call keeps it safe to share between the paper loop, the MCP
server and the dashboard.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from quant_os_core.agents.exits import InvalidationRule, OpenPosition, RuleKind
from quant_os_core.agents.trade_plan import TradePlan
from quant_os_core.risk_engine import ClosedTrade, ExitReason

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY, asset_class TEXT NOT NULL, plan TEXT NOT NULL,
    quantity TEXT NOT NULL, entry_time TEXT NOT NULL, stop TEXT NOT NULL, highest TEXT NOT NULL,
    rules TEXT NOT NULL, trail TEXT, max_holding_s REAL, flatten_at TEXT, earliest_exit TEXT
);
CREATE TABLE IF NOT EXISTS pending_entries (
    client_order_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, asset_class TEXT NOT NULL,
    plan TEXT NOT NULL, quantity TEXT NOT NULL, submitted TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS closed_trades (
    id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, exit_time TEXT NOT NULL,
    profit_ratio TEXT NOT NULL, exit_reason TEXT NOT NULL, quantity TEXT NOT NULL,
    entry_price TEXT NOT NULL, exit_price TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY, time TEXT NOT NULL, symbol TEXT NOT NULL, approved INTEGER NOT NULL,
    blocked_by TEXT, quantity TEXT NOT NULL, stages TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS equity (time TEXT PRIMARY KEY, equity TEXT NOT NULL, cash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, time TEXT NOT NULL, level TEXT NOT NULL, kind TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL, created TEXT NOT NULL,
    status TEXT NOT NULL, metrics TEXT NOT NULL, note TEXT NOT NULL
);
"""


def _iso(when: datetime) -> str:
    if when.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return when.astimezone(UTC).isoformat()


def _dt(raw: str | None) -> datetime | None:
    return datetime.fromisoformat(raw) if raw else None


@dataclass(frozen=True, slots=True)
class StoredPosition:
    position: OpenPosition
    asset_class: str


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    # ---- switches ------------------------------------------------------------------------
    def set_value(self, key: str, value: str) -> None:
        with self._db() as db:
            db.execute(
                "INSERT INTO kv(key, value, updated) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE "
                "SET value=excluded.value, updated=excluded.updated",
                (key, value, _iso(datetime.now(UTC))),
            )

    def get_value(self, key: str, default: str = "") -> str:
        with self._db() as db:
            row = db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else default

    def kill_switch_on(self) -> bool:
        return self.get_value("kill_switch", "off") == "on"

    def set_kill_switch(self, on: bool, reason: str) -> None:
        self.set_value("kill_switch", "on" if on else "off")
        self.log("critical" if on else "info", "kill_switch", f"{'ON' if on else 'off'}: {reason}")

    # ---- positions ---------------------------------------------------------------------
    def save_position(self, position: OpenPosition, asset_class: str) -> None:
        rules = [
            {"kind": r.kind.value, "level": str(r.level) if r.level else None, "window": r.window}
            for r in position.rules
        ]
        holding = position.max_holding.total_seconds() if position.max_holding else None
        with self._db() as db:
            db.execute(
                "INSERT OR REPLACE INTO positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    position.plan.symbol,
                    asset_class,
                    position.plan.model_dump_json(),
                    str(position.quantity),
                    _iso(position.entry_time),
                    str(position.stop),
                    str(position.highest),
                    json.dumps(rules),
                    str(position.trail_fraction) if position.trail_fraction else None,
                    holding,
                    _iso(position.flatten_at) if position.flatten_at else None,
                    _iso(position.earliest_exit) if position.earliest_exit else None,
                ),
            )

    def open_positions(self) -> list[StoredPosition]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM positions ORDER BY symbol").fetchall()
        out = []
        for row in rows:
            (_, asset, plan, qty, entry, stop, highest, rules, trail, hold, flat, earliest) = row
            position = OpenPosition(
                TradePlan.model_validate_json(plan),
                Decimal(qty),
                datetime.fromisoformat(entry),
                rules=tuple(
                    InvalidationRule(
                        RuleKind(r["kind"]),
                        Decimal(r["level"]) if r["level"] else None,
                        r["window"],
                    )
                    for r in json.loads(rules)
                ),
                trail_fraction=Decimal(trail) if trail else None,
                max_holding=timedelta(seconds=hold) if hold else None,
                flatten_at=_dt(flat),
                earliest_exit=_dt(earliest),
            )
            position.stop = Decimal(stop)  # restore a trailed stop exactly
            position.highest = Decimal(highest)
            out.append(StoredPosition(position, asset))
        return out

    def remove_position(self, symbol: str) -> None:
        with self._db() as db:
            db.execute("DELETE FROM positions WHERE symbol=?", (symbol,))

    # ---- pending entries ------------------------------------------------------------------
    def add_pending(
        self,
        client_order_id: str,
        plan: TradePlan,
        asset_class: str,
        quantity: Decimal,
        when: datetime,
    ) -> None:
        with self._db() as db:
            db.execute(
                "INSERT OR REPLACE INTO pending_entries VALUES (?, ?, ?, ?, ?, ?)",
                (
                    client_order_id,
                    plan.symbol,
                    asset_class,
                    plan.model_dump_json(),
                    str(quantity),
                    _iso(when),
                ),
            )

    def pending(self) -> list[dict[str, Any]]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM pending_entries").fetchall()
        return [
            {
                "client_order_id": cid,
                "symbol": sym,
                "asset_class": asset,
                "plan": TradePlan.model_validate_json(plan),
                "quantity": Decimal(qty),
                "submitted": datetime.fromisoformat(sub),
            }
            for cid, sym, asset, plan, qty, sub in rows
        ]

    def remove_pending(self, client_order_id: str) -> None:
        with self._db() as db:
            db.execute("DELETE FROM pending_entries WHERE client_order_id=?", (client_order_id,))

    # ---- history ---------------------------------------------------------------------------
    def record_close(
        self, trade: ClosedTrade, quantity: Decimal, entry_price: Decimal, exit_price: Decimal
    ) -> None:
        with self._db() as db:
            db.execute(
                "INSERT INTO closed_trades(symbol, exit_time, profit_ratio, exit_reason, "
                "quantity, entry_price, exit_price) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    trade.symbol,
                    _iso(trade.exit_time),
                    str(trade.profit_ratio),
                    trade.exit_reason.value,
                    str(quantity),
                    str(entry_price),
                    str(exit_price),
                ),
            )

    def closed_trades(self, limit: int = 500) -> list[ClosedTrade]:
        with self._db() as db:
            rows = db.execute(
                "SELECT symbol, exit_time, profit_ratio, exit_reason FROM closed_trades "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            ClosedTrade(s, datetime.fromisoformat(t), Decimal(p), ExitReason(r))
            for s, t, p, r in reversed(rows)
        ]

    def record_decision(
        self,
        when: datetime,
        symbol: str,
        approved: bool,
        blocked_by: str | None,
        quantity: Decimal,
        stages: list[dict[str, Any]],
    ) -> None:
        with self._db() as db:
            db.execute(
                "INSERT INTO decisions(time, symbol, approved, blocked_by, quantity, stages) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (_iso(when), symbol, int(approved), blocked_by, str(quantity), json.dumps(stages)),
            )

    def record_equity(self, when: datetime, equity: Decimal, cash: Decimal) -> None:
        with self._db() as db:
            db.execute(
                "INSERT OR REPLACE INTO equity VALUES (?, ?, ?)",
                (_iso(when), str(equity), str(cash)),
            )

    def equity_curve(self, limit: int = 2000) -> list[tuple[datetime, Decimal]]:
        with self._db() as db:
            rows = db.execute(
                "SELECT time, equity FROM equity ORDER BY time DESC LIMIT ?", (limit,)
            ).fetchall()
        return [(datetime.fromisoformat(t), Decimal(e)) for t, e in reversed(rows)]

    def log(self, level: str, kind: str, message: str) -> None:
        with self._db() as db:
            db.execute(
                "INSERT INTO events(time, level, kind, message) VALUES (?, ?, ?, ?)",
                (_iso(datetime.now(UTC)), level, kind, message[:2000]),
            )

    def recent(self, table: str, limit: int = 50) -> list[dict[str, Any]]:
        """Newest rows of a history table, for the dashboard."""

        allowed = {"events", "decisions", "closed_trades", "models"}
        if table not in allowed:
            raise ValueError(f"table must be one of {sorted(allowed)}")
        with self._db() as db:
            cursor = db.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,))
            names = [c[0] for c in cursor.description]
            return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

    # ---- model registry --------------------------------------------------------------------
    def register_model(
        self, name: str, version: str, status: str, metrics: dict[str, float], note: str
    ) -> None:
        with self._db() as db:
            if status == "champion":
                db.execute(
                    "UPDATE models SET status='retired' WHERE name=? AND status='champion'",
                    (name,),
                )
            db.execute(
                "INSERT INTO models(name, version, created, status, metrics, note) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, version, _iso(datetime.now(UTC)), status, json.dumps(metrics), note),
            )

    def champion(self, name: str) -> dict[str, Any] | None:
        with self._db() as db:
            row = db.execute(
                "SELECT version, metrics, created FROM models WHERE name=? AND status='champion' "
                "ORDER BY id DESC LIMIT 1",
                (name,),
            ).fetchone()
        if not row:
            return None
        return {"version": row[0], "metrics": json.loads(row[1]), "created": row[2]}
