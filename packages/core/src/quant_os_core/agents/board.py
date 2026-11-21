"""Agent board: the shared, durable workspace between the bot and outside AI assistants.

Two directions over MCP:

* **bot -> assistants**: the bot posts *research tasks* ("review BTC/USD as news analyst").
  Claude, ChatGPT or a local model pick them up, research with their own tools, and answer.
* **assistants -> bot**: assistants submit typed *analyst reports* and paper *trade proposals*.
  Reports feed the analyst council (veto-and-shrink only). Proposals run through the full
  entry pipeline in paper mode; nothing here can reach a live broker.

Storage is one SQLite file (WAL mode, one connection per call), so the MCP server, the
paper loop and the dashboard can share it safely.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from quant_os_core.agents.council import AnalystReport, Stance
from quant_os_core.agents.trade_plan import TradePlan


class AnalystRole(StrEnum):
    FUNDAMENTALS = "fundamentals"
    NEWS_SENTIMENT = "news_sentiment"
    TECHNICAL = "technical"
    MACRO = "macro"
    RISK_OFFICER = "risk_officer"
    SHARIAH_REVIEWER = "shariah_reviewer"  # can raise red flags; the gate still decides
    BEAR_ADVOCATE = "bear_advocate"  # must argue against the trade


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, role TEXT NOT NULL, question TEXT NOT NULL,
    created TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open', claimed_by TEXT
);
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, role TEXT NOT NULL, source TEXT NOT NULL,
    stance TEXT NOT NULL, conviction TEXT NOT NULL, red_flag INTEGER NOT NULL,
    rationale TEXT NOT NULL, created TEXT NOT NULL, task_id INTEGER
);
CREATE INDEX IF NOT EXISTS reports_symbol ON reports(symbol, created);
CREATE TABLE IF NOT EXISTS grades (
    report_id INTEGER PRIMARY KEY, source TEXT NOT NULL, role TEXT NOT NULL,
    correct INTEGER NOT NULL, conviction TEXT NOT NULL, graded TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS proposals (
    id INTEGER PRIMARY KEY, source TEXT NOT NULL, plan TEXT NOT NULL, asset_class TEXT NOT NULL,
    created TEXT NOT NULL, approved INTEGER NOT NULL, blocked_by TEXT, quantity TEXT NOT NULL,
    stages TEXT NOT NULL
);
"""


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class BoardReport:
    symbol: str
    role: AnalystRole
    source: str
    report: AnalystReport
    created: datetime


class AgentBoard:
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
            with connection:  # commits on success, rolls back on error
                yield connection
        finally:
            connection.close()

    # ---- research tasks (bot -> assistants) -------------------------------------------
    def post_task(self, symbol: str, role: AnalystRole, question: str) -> int:
        if not symbol.strip() or len(question.strip()) < 10:
            raise ValueError("task needs a symbol and a question of at least 10 characters")
        with self._db() as db:
            cursor = db.execute(
                "INSERT INTO tasks(symbol, role, question, created) VALUES (?, ?, ?, ?)",
                (symbol.upper(), role.value, question.strip(), _now().isoformat()),
            )
            return int(cursor.lastrowid or 0)

    def open_tasks(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._db() as db:
            rows = db.execute(
                "SELECT id, symbol, role, question, created FROM tasks WHERE status='open' "
                "ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ("id", "symbol", "role", "question", "created")
        return [dict(zip(keys, row, strict=True)) for row in rows]

    # ---- analyst reports (assistants -> bot) -------------------------------------------
    def submit_report(
        self,
        *,
        symbol: str,
        role: AnalystRole,
        source: str,
        stance: Stance,
        conviction: Decimal,
        rationale: str,
        red_flag: bool = False,
        task_id: int | None = None,
    ) -> int:
        source = source.strip().lower()
        if not source or len(source) > 40:
            raise ValueError("source must name the assistant, e.g. 'claude' or 'chatgpt'")
        if len(rationale.strip()) < 20:
            raise ValueError("rationale must explain the view in at least 20 characters")
        report = AnalystReport(
            analyst=f"{source}:{role.value}"[:40],
            stance=stance,
            conviction=conviction,
            rationale=rationale.strip()[:800],
            red_flag=red_flag,
        )
        if role is AnalystRole.BEAR_ADVOCATE and stance is Stance.BULLISH:
            raise ValueError("the bear advocate cannot submit a bullish report")
        with self._db() as db:
            cursor = db.execute(
                "INSERT INTO reports(symbol, role, source, stance, conviction, red_flag, "
                "rationale, created, task_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    symbol.upper(),
                    role.value,
                    source,
                    report.stance.value,
                    str(report.conviction),
                    int(report.red_flag),
                    report.rationale,
                    _now().isoformat(),
                    task_id,
                ),
            )
            if task_id is not None:
                db.execute(
                    "UPDATE tasks SET status='done', claimed_by=? WHERE id=?", (source, task_id)
                )
            return int(cursor.lastrowid or 0)

    def recent_reports(self, symbol: str, max_age: timedelta) -> list[BoardReport]:
        since = (_now() - max_age).isoformat()
        with self._db() as db:
            rows = db.execute(
                "SELECT symbol, role, source, stance, conviction, red_flag, rationale, created "
                "FROM reports WHERE symbol=? AND created>=? ORDER BY created DESC",
                (symbol.upper(), since),
            ).fetchall()
        out = []
        for sym, role, source, stance, conviction, flag, rationale, created in rows:
            report = AnalystReport(
                analyst=f"{source}:{role}"[:40],
                stance=Stance(stance),
                conviction=Decimal(conviction),
                rationale=rationale,
                red_flag=bool(flag),
            )
            out.append(
                BoardReport(sym, AnalystRole(role), source, report, datetime.fromisoformat(created))
            )
        return out

    # ---- grading: turn opinions into a track record ------------------------------------
    def grade_trade(
        self,
        symbol: str,
        entry_time: datetime,
        profit_ratio: Decimal,
        window: timedelta = timedelta(hours=24),
    ) -> int:
        """Grade every directional report made in the window before entry. Returns count.

        Bullish is right if the trade made money; bearish is right if it lost or broke even.
        Neutral reports are not graded. Each report is graded once.
        """

        start, end = (entry_time - window).isoformat(), entry_time.isoformat()
        with self._db() as db:
            rows = db.execute(
                "SELECT id, source, role, stance, conviction FROM reports WHERE symbol=? AND "
                "created BETWEEN ? AND ? AND stance != 'neutral' AND id NOT IN "
                "(SELECT report_id FROM grades)",
                (symbol.upper(), start, end),
            ).fetchall()
            for rid, source, role, stance, conviction in rows:
                correct = (stance == "bullish") == (profit_ratio > 0)
                db.execute(
                    "INSERT INTO grades VALUES (?, ?, ?, ?, ?, ?)",
                    (rid, source, role, int(correct), conviction, _now().isoformat()),
                )
        return len(rows)

    def reliability(self, source: str, role: str, min_samples: int = 10) -> float:
        """Weight in [0.25, 1.0] from the hit rate; 0.75 until there are enough graded calls.

        A coin-flip analyst (50% hits) gets 0.5; 75%+ gets the full 1.0; a reliably wrong one
        is cut to 0.25. Weights only scale conviction down, never up.
        """

        with self._db() as db:
            row = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(correct), 0) FROM grades WHERE source=? AND role=?",
                (source, role),
            ).fetchone()
        total, hits = int(row[0]), int(row[1])
        if total < min_samples:
            return 0.75
        hit_rate = hits / total
        return max(0.25, min(1.0, 0.5 + (hit_rate - 0.5) * 2))

    def leaderboard(self) -> list[dict[str, Any]]:
        with self._db() as db:
            rows = db.execute(
                "SELECT source, role, COUNT(*), SUM(correct) FROM grades GROUP BY source, role "
                "ORDER BY SUM(correct) * 1.0 / COUNT(*) DESC"
            ).fetchall()
        return [
            {"source": s, "role": r, "graded": n, "hit_rate": round(h / n, 3)}
            for s, r, n, h in rows
        ]

    # ---- proposals ----------------------------------------------------------------------
    def record_proposal(
        self,
        *,
        source: str,
        plan: TradePlan,
        asset_class: str,
        approved: bool,
        blocked_by: str | None,
        quantity: Decimal,
        stages: list[dict[str, Any]],
    ) -> int:
        with self._db() as db:
            cursor = db.execute(
                "INSERT INTO proposals(source, plan, asset_class, created, approved, blocked_by, "
                "quantity, stages) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source.strip().lower(),
                    plan.model_dump_json(),
                    asset_class,
                    _now().isoformat(),
                    int(approved),
                    blocked_by,
                    str(quantity),
                    json.dumps(stages),
                ),
            )
            return int(cursor.lastrowid or 0)

    def proposals(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._db() as db:
            rows = db.execute(
                "SELECT id, source, plan, asset_class, created, approved, blocked_by, quantity, "
                "stages FROM proposals ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for pid, source, plan, asset, created, approved, blocked, qty, stages in rows:
            out.append(
                {
                    "id": pid,
                    "source": source,
                    "plan": json.loads(plan),
                    "asset_class": asset,
                    "created": created,
                    "approved": bool(approved),
                    "blocked_by": blocked,
                    "quantity": qty,
                    "stages": json.loads(stages),
                }
            )
        return out


@dataclass(frozen=True, slots=True)
class BoardAnalyst:
    """Council seat filled from the board: the latest report for one role, else neutral.

    Conviction is scaled by the source's graded track record for that role, so analysts who
    are usually wrong lose influence (red flags still veto regardless of weight).

    A stale or missing report counts as "no opinion", the same fail-closed rule used for
    local LLM analysts.
    """

    board: AgentBoard
    role: AnalystRole
    max_age: timedelta = timedelta(hours=24)

    @property
    def name(self) -> str:
        return f"board:{self.role.value}"

    def review(self, plan: TradePlan) -> AnalystReport:
        for item in self.board.recent_reports(plan.symbol, self.max_age):
            if item.role is self.role:
                weight = Decimal(str(self.board.reliability(item.source, item.role.value)))
                scaled = (item.report.conviction * weight).quantize(Decimal("0.0001"))
                return item.report.model_copy(update={"conviction": scaled})
        return AnalystReport(
            analyst=self.name,
            stance=Stance.NEUTRAL,
            conviction=Decimal("0"),
            rationale="no recent report on the board",
        )


def distinct_sources(board: AgentBoard, symbol: str, max_age: timedelta) -> set[str]:
    return {r.source for r in board.recent_reports(symbol, max_age)}
