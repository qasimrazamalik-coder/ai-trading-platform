"""Trading journal: why each trade was taken, how it felt and what it taught.

Entries live in the state database's ``journal_entries`` table (created when missing). Each
entry may link to ledger fills by id; a linked set whose buys and sells balance counts as a
closed trade, and its net cash flow after fees decides win or loss for the statistics:

* rule-adherence rate = entries with ``rule_followed`` / all entries;
* win rate by setup   = closed linked trades with net profit > 0 / closed linked trades.

The journal records thoughts; it never sends an order.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from quant_os_core.ops.ledger import Ledger

_SCHEMA = """
CREATE TABLE IF NOT EXISTS journal_entries (
    id INTEGER PRIMARY KEY, time TEXT NOT NULL, symbol TEXT NOT NULL, setup TEXT NOT NULL,
    reason TEXT NOT NULL, emotion TEXT NOT NULL, rule_followed INTEGER NOT NULL,
    screenshot TEXT NOT NULL, lesson TEXT NOT NULL, tags TEXT NOT NULL, fill_ids TEXT NOT NULL
);
"""

EMOTIONS = ("calm", "confident", "fearful", "greedy", "impatient", "frustrated", "unsure")
MAX_TEXT = 2000
MAX_SHORT = 60
MAX_TAGS = 10
MAX_FILLS = 50


def _short(name: str, raw: str, required: bool = False) -> str:
    text = raw.strip()
    if required and not text:
        raise ValueError(f"{name} is required")
    if len(text) > MAX_SHORT:
        raise ValueError(f"{name} is longer than {MAX_SHORT} characters")
    return text


def _long(name: str, raw: str) -> str:
    if len(raw) > MAX_TEXT:
        raise ValueError(f"{name} is longer than {MAX_TEXT} characters")
    return raw.strip()


def _screenshot(raw: str) -> str:
    path = raw.strip().replace("\\", "/")
    if len(path) > 260:
        raise ValueError("screenshot path is longer than 260 characters")
    if ".." in path.split("/") or "://" in path:
        raise ValueError("screenshot must be a local path without '..'")
    return path


def _tags(raw: Sequence[str]) -> list[str]:
    tags = sorted({_short("tag", str(t)).lower() for t in raw if str(t).strip()})
    if len(tags) > MAX_TAGS:
        raise ValueError(f"at most {MAX_TAGS} tags")
    return tags


@dataclass(frozen=True, slots=True)
class JournalEntry:
    id: int
    time: datetime
    symbol: str
    setup: str
    reason: str
    emotion: str
    rule_followed: bool
    screenshot: str
    lesson: str
    tags: tuple[str, ...]
    fill_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "time": self.time.isoformat(),
            "symbol": self.symbol,
            "setup": self.setup,
            "reason": self.reason,
            "emotion": self.emotion,
            "rule_followed": self.rule_followed,
            "screenshot": self.screenshot,
            "lesson": self.lesson,
            "tags": list(self.tags),
            "fill_ids": list(self.fill_ids),
        }


class Journal:
    def __init__(self, path: Path, ledger: Ledger | None = None) -> None:
        self.path = path
        self.ledger = ledger or Ledger(path)
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

    def add(
        self,
        when: datetime,
        symbol: str,
        setup: str,
        reason: str,
        emotion: str,
        rule_followed: bool,
        *,
        screenshot: str = "",
        lesson: str = "",
        tags: Sequence[str] = (),
        fill_ids: Sequence[int] = (),
    ) -> int:
        if when.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        sym = _short("symbol", symbol, required=True).upper()
        if emotion not in EMOTIONS:
            raise ValueError(f"emotion must be one of {', '.join(EMOTIONS)}")
        if not isinstance(rule_followed, bool):
            raise ValueError("rule_followed must be true or false")
        ids = sorted({int(i) for i in fill_ids})
        if len(ids) > MAX_FILLS:
            raise ValueError(f"at most {MAX_FILLS} linked fills")
        if ids:
            known = {f.id: f for f in self.ledger.fills_by_id(ids)}
            missing = [i for i in ids if i not in known]
            if missing:
                raise ValueError(f"unknown ledger fill ids: {missing}")
            if any(f.symbol != sym for f in known.values()):
                raise ValueError("linked fills must be for the same symbol")
        with self._db() as db:
            cursor = db.execute(
                "INSERT INTO journal_entries(time, symbol, setup, reason, emotion, rule_followed, "
                "screenshot, lesson, tags, fill_ids) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    when.astimezone(UTC).isoformat(),
                    sym,
                    _short("setup", setup, required=True).lower(),
                    _long("reason", reason),
                    emotion,
                    int(rule_followed),
                    _screenshot(screenshot),
                    _long("lesson", lesson),
                    json.dumps(_tags(tags)),
                    json.dumps(ids),
                ),
            )
            return int(cursor.lastrowid or 0)

    def entries(self, limit: int = 200) -> list[JournalEntry]:
        with self._db() as db:
            rows = db.execute(
                "SELECT * FROM journal_entries ORDER BY time DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            JournalEntry(
                int(i),
                datetime.fromisoformat(t),
                s,
                setup,
                reason,
                emotion,
                bool(ok),
                shot,
                lesson,
                tuple(json.loads(tags)),
                tuple(int(x) for x in json.loads(fills)),
            )
            for i, t, s, setup, reason, emotion, ok, shot, lesson, tags, fills in rows
        ]

    def outcome(self, entry: JournalEntry) -> Decimal | None:
        """Net profit of the linked fills, or ``None`` while they do not form a closed trade."""

        fills = self.ledger.fills_by_id(list(entry.fill_ids))
        bought = sum((f.quantity for f in fills if f.side == "buy"), Decimal("0"))
        sold = sum((f.quantity for f in fills if f.side == "sell"), Decimal("0"))
        if not fills or bought == 0 or bought != sold:
            return None
        net = Decimal("0")
        for f in fills:
            flow = f.quantity * f.price
            net += (flow if f.side == "sell" else -flow) - f.fee
        return net

    def stats(self) -> dict[str, Any]:
        entries = self.entries(10_000)
        followed = sum(1 for e in entries if e.rule_followed)
        by_setup: dict[str, dict[str, Any]] = {}
        for entry in entries:
            row = by_setup.setdefault(
                entry.setup, {"entries": 0, "closed": 0, "wins": 0, "win_rate": None}
            )
            row["entries"] += 1
            result = self.outcome(entry)
            if result is not None:
                row["closed"] += 1
                row["wins"] += int(result > 0)
        for row in by_setup.values():
            if row["closed"]:
                row["win_rate"] = row["wins"] / row["closed"]
        emotions: dict[str, int] = {}
        for entry in entries:
            emotions[entry.emotion] = emotions.get(entry.emotion, 0) + 1
        return {
            "entries": len(entries),
            "rule_adherence": followed / len(entries) if entries else None,
            "by_setup": dict(sorted(by_setup.items())),
            "emotions": dict(sorted(emotions.items())),
        }
