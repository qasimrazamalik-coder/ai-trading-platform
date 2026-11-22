"""Weekly paper-trading report, sent through the configured alert sinks.

    python -m quant_os_core.ops.report --config paper_core.json          # build and send
    python -m quant_os_core.ops.report --config paper_core.json --dry-run  # print only

Reads the state store only (no broker or market-data calls): equity change over 7 and 30
days, worst drawdown, orders approved and blocked (with blocked-by counts), the core's drift
from its last target weights when the core loop stored them, and a warning when a fund's
``financials_source`` date in the config is older than 120 days (re-run the fund screen).
Alerts go through ntfy, webhooks or the state log (``QUANT_OS_NTFY_TOPIC``,
``QUANT_OS_WEBHOOK_URL``); Telegram is not used.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from quant_os_core.ops.alerts import AlertSink
from quant_os_core.ops.core_loop import TARGETS_KEY, WEIGHTS_KEY
from quant_os_core.ops.state import StateStore

SCREEN_MAX_AGE_DAYS = 120
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass(slots=True)
class WeeklyReport:
    now: datetime
    equity: Decimal | None = None
    change_week: Decimal | None = None  # fraction, e.g. 0.012 = +1.2%
    change_month: Decimal | None = None
    worst_drawdown_month: Decimal = Decimal(0)  # fraction below the running peak, >= 0
    worst_drawdown_all: Decimal = Decimal(0)
    approved: int = 0
    blocked: Counter[str] = field(default_factory=Counter)
    closed_trades: int = 0
    drift: dict[str, Decimal] = field(default_factory=dict)  # weight - target
    warnings: list[str] = field(default_factory=list)

    @property
    def level(self) -> str:
        return "warning" if self.warnings else "info"

    def text(self) -> str:
        def pct(value: Decimal | None) -> str:
            return "n/a" if value is None else f"{value:+.2%}"

        lines = [
            f"Week to {self.now:%Y-%m-%d}",
            f"Equity: {'n/a' if self.equity is None else f'${self.equity:,.2f}'}",
            f"Change: week {pct(self.change_week)}, month {pct(self.change_month)}",
            f"Worst drawdown: 30 days {self.worst_drawdown_month:.2%}, "
            f"all time {self.worst_drawdown_all:.2%}",
            f"Orders this week: {self.approved} approved, {sum(self.blocked.values())} blocked; "
            f"closed trades {self.closed_trades}",
        ]
        if self.blocked:
            lines.append(
                "Blocked by: " + ", ".join(f"{k} {v}" for k, v in self.blocked.most_common())
            )
        if self.drift:
            lines.append(
                "Core drift vs target: "
                + ", ".join(f"{s} {d:+.1%}" for s, d in sorted(self.drift.items()))
            )
        lines.extend(f"WARNING: {w}" for w in self.warnings)
        return "\n".join(lines)


def _value_at(curve: Sequence[tuple[datetime, Decimal]], when: datetime) -> Decimal | None:
    """Last equity at or before ``when`` (None when the curve starts later)."""

    before = [e for t, e in curve if t <= when]
    return before[-1] if before else None


def _drawdown(values: Sequence[Decimal]) -> Decimal:
    peak, worst = Decimal(0), Decimal(0)
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


def _change(now_value: Decimal, then: Decimal | None) -> Decimal | None:
    return None if then is None or then <= 0 else now_value / then - 1


def screening_warnings(config: Mapping[str, Any], today: date) -> list[str]:
    """Funds whose screening data is missing or older than ``SCREEN_MAX_AGE_DAYS``."""

    out: list[str] = []
    for row in config.get("universe") or []:
        symbol = str(row.get("symbol", "?")).upper()
        if row.get("asset_class") not in ("etf", "equity"):  # crypto uses token screens
            continue
        if not row.get("financials"):
            out.append(f"{symbol} has no financials: the Shariah gate blocks it (R6)")
            continue
        match = _DATE.search(str(row.get("financials_source") or ""))
        if not match:
            out.append(f"{symbol} financials have no dated source; re-run the fund screen")
            continue
        age = (today - date.fromisoformat(match.group(1))).days
        if age > SCREEN_MAX_AGE_DAYS:
            out.append(
                f"{symbol} screening data is {age} days old (> {SCREEN_MAX_AGE_DAYS}); "
                "re-run the fund screen"
            )
    return out


def build_report(
    store: StateStore, config: Mapping[str, Any], now: datetime | None = None
) -> WeeklyReport:
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    report = WeeklyReport(now)
    week_ago, month_ago = now - timedelta(days=7), now - timedelta(days=30)
    curve = [(t, e) for t, e in store.equity_curve(100_000) if t <= now]
    if curve:
        report.equity = curve[-1][1]
        report.change_week = _change(report.equity, _value_at(curve, week_ago))
        report.change_month = _change(report.equity, _value_at(curve, month_ago))
        report.worst_drawdown_all = _drawdown([e for _, e in curve])
        report.worst_drawdown_month = _drawdown([e for t, e in curve if t >= month_ago])
    else:
        report.warnings.append("no equity snapshots yet: is the paper loop running?")
    for row in store.recent("decisions", 100_000):
        if not week_ago <= datetime.fromisoformat(row["time"]) <= now:
            continue
        if row["approved"]:
            report.approved += 1
        else:
            report.blocked[str(row["blocked_by"] or "unknown")] += 1
    report.closed_trades = sum(
        1 for t in store.closed_trades(100_000) if week_ago <= t.exit_time <= now
    )
    targets_raw, weights_raw = store.get_value(TARGETS_KEY), store.get_value(WEIGHTS_KEY)
    if targets_raw:
        targets = {s: Decimal(w) for s, w in json.loads(targets_raw).items()}
        weights = {s: Decimal(w) for s, w in json.loads(weights_raw or "{}").items()}
        report.drift = {s: weights.get(s, Decimal(0)) - t for s, t in targets.items()}
    report.warnings.extend(screening_warnings(config, now.date()))
    if store.kill_switch_on():
        report.warnings.append("kill switch is ON: no new trades")
    return report


def send_report(report: WeeklyReport, sink: AlertSink) -> None:
    sink.send(report.level, "weekly paper report", report.text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Weekly paper-trading report")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, help="state file (default: the config's)")
    parser.add_argument("--dry-run", action="store_true", help="print, do not send")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text("utf-8"))
    store = StateStore(args.state or Path(config.get("state", "state/trading.db")))
    report = build_report(store, config)
    print(report.text(), file=sys.stdout)
    if not args.dry_run:
        from quant_os_core.ops.run_paper import _alerts

        send_report(report, _alerts(config, store))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
