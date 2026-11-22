"""Live-readiness check over the paper record. Advisory only: it unlocks nothing.

    python -m quant_os_core.ops.readiness --config paper_core.json
    python -m quant_os_core.ops.readiness --config paper_core.json --mark-resolved "fixed AAPL qty"

PASS needs, from the state store alone:

* at least ``min_days`` (30) of paper equity snapshots, the newest no older than 4 days;
* no unresolved position-mismatch events (resolve one with ``--mark-resolved`` after fixing it);
* no critical events in the last 7 days;
* worst drawdown of the paper equity within the risk dial's limit;
* at least ``min_rebalances`` completed core rebalances (calendar months with an approved
  cost-guard order; 0 for the breakout loop);
* reconciliation clean: kill switch off, no core order stuck pending, no mismatch,
  unexpected-account or broker-unavailable event in the last day of activity.

Live trading stays behind the two-key interlock (``Settings.enforce_live_trading_interlock``:
production mode plus the written risk acknowledgement). Nothing here reads or changes it; a
PASS is one input to the owner's decision, not permission to trade real money.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from quant_os_core.ops.state import StateStore

# Worst paper drawdown accepted per risk dial. Balanced matches halal_risk_limits() (15%).
DIAL_MAX_DRAWDOWN = {
    "conservative": Decimal("0.10"),
    "balanced": Decimal("0.15"),
    "growth": Decimal("0.25"),
}
_MISMATCH = ("position mismatch", "unexpected account state")
_RECONCILE_BAD = (*_MISMATCH, "broker unavailable")
_RESOLVED = "mismatch resolved"
STALE_AFTER = timedelta(days=4)


@dataclass(slots=True)
class Readiness:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(ok for _, ok, _ in self.checks)

    @property
    def reasons(self) -> list[str]:
        return [f"{name}: {detail}" for name, ok, detail in self.checks if not ok]

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append((name, ok, detail))

    def text(self) -> str:
        lines = [f"Live readiness: {'PASS' if self.passed else 'FAIL'} (advisory only)"]
        lines += [
            f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}" for name, ok, detail in self.checks
        ]
        lines.append("Live trading still needs the two-key interlock; this check unlocks nothing.")
        return "\n".join(lines)


def mark_resolved(store: StateStore, note: str) -> None:
    """Record that the owner fixed every mismatch reported so far."""

    if not note.strip():
        raise ValueError("say what was fixed")
    store.log("info", "readiness", f"{_RESOLVED}: {note.strip()}")


def _events(store: StateStore) -> list[tuple[int, datetime, str, str, str]]:
    """(id, time, level, kind, message), oldest first; the id orders same-time events."""

    rows = store.recent("events", 1_000_000)
    return sorted(
        (int(r["id"]), datetime.fromisoformat(r["time"]), r["level"], r["kind"], r["message"])
        for r in rows
    )


def check_readiness(
    store: StateStore,
    *,
    dial: str = "balanced",
    min_days: int = 30,
    min_rebalances: int = 2,
    now: datetime | None = None,
) -> Readiness:
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if dial not in DIAL_MAX_DRAWDOWN:
        raise ValueError(f"dial must be one of {sorted(DIAL_MAX_DRAWDOWN)}")
    result = Readiness()

    curve = [(t, e) for t, e in store.equity_curve(1_000_000) if t <= now]
    span = curve[-1][0] - curve[0][0] if curve else timedelta(0)
    result.add(
        "history",
        span >= timedelta(days=min_days),
        f"{span.days} days of paper equity (need {min_days})",
    )
    fresh = bool(curve) and now - curve[-1][0] <= STALE_AFTER
    result.add(
        "fresh",
        fresh,
        f"last snapshot {curve[-1][0]:%Y-%m-%d}" if curve else "no equity snapshots",
    )

    events = [e for e in _events(store) if e[1] <= now]
    resolved_id = max(
        (i for i, _, _, kind, msg in events if kind == "readiness" and msg.startswith(_RESOLVED)),
        default=0,
    )
    open_mismatches = [
        (t, msg)
        for i, t, _, _, msg in events
        if i > resolved_id and any(m in msg for m in _MISMATCH)
    ]
    result.add(
        "mismatches",
        not open_mismatches,
        (
            f"{len(open_mismatches)} unresolved, latest {open_mismatches[-1][0]:%Y-%m-%d}: "
            f"{open_mismatches[-1][1][:120]}"
            if open_mismatches
            else "none unresolved"
        ),
    )
    week_ago = now - timedelta(days=7)
    critical = [(t, m) for _, t, level, _, m in events if level == "critical" and t >= week_ago]
    result.add(
        "critical events",
        not critical,
        f"{len(critical)} in 7 days, latest: {critical[-1][1][:120]}"
        if critical
        else "none in 7 days",
    )

    peak, worst = Decimal(0), Decimal(0)
    for _, equity in curve:
        peak = max(peak, equity)
        if peak > 0:
            worst = max(worst, (peak - equity) / peak)
    limit = DIAL_MAX_DRAWDOWN[dial]
    result.add("drawdown", worst <= limit, f"worst {worst:.2%} (limit {limit:.0%}, {dial} dial)")

    months: set[str] = set()
    for row in store.recent("decisions", 1_000_000):
        stages = json.loads(row["stages"])
        if row["approved"] and any(s.get("stage") == "cost_guard" for s in stages):
            months.add(str(row["time"])[:7])
    result.add(
        "rebalances",
        len(months) >= min_rebalances,
        f"{len(months)} completed (need {min_rebalances})",
    )

    problems: list[str] = []
    if store.kill_switch_on():
        problems.append("kill switch on")
    if store.get_value("core:pending"):
        problems.append("core order still pending")
    if curve:
        recent_bad = [
            msg
            for _, t, _, _, msg in events
            if t >= curve[-1][0] - timedelta(days=1) and any(b in msg for b in _RECONCILE_BAD)
        ]
        if recent_bad:
            problems.append(f"last step: {recent_bad[-1][:120]}")
    result.add("reconciliation", not problems, "; ".join(problems) or "clean")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Advisory live-readiness check (paper record)")
    parser.add_argument("--config", type=Path, help="paper config (state file, dial, strategy)")
    parser.add_argument("--state", type=Path, help="state file (default: the config's)")
    parser.add_argument("--dial", choices=sorted(DIAL_MAX_DRAWDOWN))
    parser.add_argument("--min-days", type=int, default=30)
    parser.add_argument("--min-rebalances", type=int, help="default 2 for the core, else 0")
    parser.add_argument("--mark-resolved", metavar="NOTE", help="record that mismatches are fixed")
    args = parser.parse_args(argv)
    config: dict[str, Any] = json.loads(args.config.read_text("utf-8")) if args.config else {}
    store = StateStore(args.state or Path(config.get("state", "state/trading.db")))
    if args.mark_resolved:
        mark_resolved(store, args.mark_resolved)
    core = (config.get("core") or {}).get("gate", "cost_guard") == "cost_guard"
    is_core = config.get("strategy") == "core" and core
    result = check_readiness(
        store,
        dial=args.dial or str(config.get("dial", "balanced")),
        min_days=args.min_days,
        min_rebalances=(
            args.min_rebalances if args.min_rebalances is not None else (2 if is_core else 0)
        ),
    )
    print(result.text(), file=sys.stdout)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
