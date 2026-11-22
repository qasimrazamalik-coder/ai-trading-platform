"""Signal agent: deterministic notices about the diversified core. Information only.

It reads the core's target weights (``strategies.diversified_core.target_weights``) from
daily bars and the current holdings from a snapshot passed in (a broker account or the
``StateStore``), and produces typed :class:`Signal` records:

* ``REBALANCE_DUE``: the monthly rebalance has not run this calendar month;
* ``DRIFT_ALERT``: a fund is more than ``band`` off its target weight;
* ``TREND_EXIT`` / ``TREND_ENTRY``: in defensive mode, a fund's close crossed below / above
  its 200-day average on the latest bar;
* ``SCREEN_STALE``: a fund's Shariah screening financials are older than 120 days or missing.

Signals go to the analyst board (as a research task for the risk officer) and to alert sinks,
with a short summary in English, Urdu or Arabic. Nothing executes from a signal: this module
has no broker and places, approves or changes no order. The core loop and its gates decide.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from quant_os_core.agents.board import AgentBoard, AnalystRole
from quant_os_core.agents.exits import Bar
from quant_os_core.execution.models import AccountSnapshot
from quant_os_core.ops.alerts import AlertSink
from quant_os_core.ops.state import StateStore
from quant_os_core.strategies.diversified_core import (
    TREND_WINDOW,
    CoreMode,
    is_rebalance_day,
    target_weights,
)

SIGNAL_TITLE = "Core signals"
BOARD_SYMBOL = "CORE"
LAST_REBALANCE_KEY = "core:last_rebalance"  # written by ops.core_loop
MAX_SCREEN_AGE = timedelta(days=120)
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_PCT = Decimal("0.01")


class SignalKind(StrEnum):
    REBALANCE_DUE = "REBALANCE_DUE"
    DRIFT_ALERT = "DRIFT_ALERT"
    TREND_EXIT = "TREND_EXIT"
    TREND_ENTRY = "TREND_ENTRY"
    SCREEN_STALE = "SCREEN_STALE"


@dataclass(frozen=True, slots=True)
class Signal:
    kind: SignalKind
    symbol: str  # BOARD_SYMBOL for book-level signals
    level: str  # "info" or "warning", as used by the alert sinks
    detail: str  # English, with the numbers
    data: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "symbol": self.symbol,
            "level": self.level,
            "detail": self.detail,
            "data": dict(self.data),
        }


@dataclass(frozen=True, slots=True)
class HoldingsSnapshot:
    """Market value per symbol, cash and the date of the last completed rebalance."""

    cash: Decimal
    values: Mapping[str, Decimal]
    last_rebalance: date | None = None

    @property
    def equity(self) -> Decimal:
        return self.cash + sum(self.values.values(), Decimal(0))


@dataclass(frozen=True, slots=True)
class ScreenRecord:
    symbol: str
    financials: Mapping[str, str] | None
    source: str
    as_of: date | None


def _last_rebalance(store: StateStore | None) -> date | None:
    raw = store.get_value(LAST_REBALANCE_KEY) if store is not None else ""
    return date.fromisoformat(raw) if raw else None


def holdings_from_account(
    account: AccountSnapshot, store: StateStore | None = None
) -> HoldingsSnapshot:
    """Snapshot from a broker account (read only); the rebalance date comes from the store."""

    values: dict[str, Decimal] = {}
    for position in account.positions:
        symbol = position.instrument_id.symbol
        values[symbol] = values.get(symbol, Decimal(0)) + position.notional
    return HoldingsSnapshot(account.cash, values, _last_rebalance(store))


def holdings_from_store(
    store: StateStore, prices: Mapping[str, Decimal], cash: Decimal
) -> HoldingsSnapshot:
    """Snapshot from the StateStore's open positions marked at ``prices`` (read only)."""

    values: dict[str, Decimal] = {}
    for stored in store.open_positions():
        position = stored.position
        symbol = position.plan.symbol
        price = prices.get(symbol, position.plan.entry_price)
        values[symbol] = values.get(symbol, Decimal(0)) + position.quantity * price
    return HoldingsSnapshot(cash, values, _last_rebalance(store))


def load_screening(path: Path) -> dict[str, ScreenRecord]:
    """Screening records from a paper config (``universe`` entries with ``financials``).

    The as-of date is ``financials_date`` if present, else the first ISO date found in
    ``financials_source`` (e.g. "SEC N-PORT 2026-08-31, look-through").
    """

    config = json.loads(path.read_text("utf-8"))
    records: dict[str, ScreenRecord] = {}
    for item in config.get("universe", []):
        symbol = str(item.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        raw = item.get("financials")
        financials = {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else None
        source = str(item.get("financials_source", ""))
        stamp = str(item.get("financials_date", "")) or source
        match = _DATE.search(stamp)
        as_of = date.fromisoformat(match.group(0)) if match and financials else None
        records[symbol] = ScreenRecord(symbol, financials, source, as_of)
    return records


def _pct(value: Decimal) -> str:
    return f"{(value * 100).quantize(_PCT)}%"


def _average(closes: Sequence[Decimal], end: int, window: int) -> Decimal:
    tail = closes[end - window : end]
    return sum(tail, Decimal(0)) / Decimal(window)


def compute_signals(
    bars: Mapping[str, Sequence[Bar]],
    holdings: HoldingsSnapshot,
    today: date,
    *,
    mode: CoreMode = CoreMode.DEFENSIVE,
    band: Decimal = Decimal("0.02"),
    screening: Mapping[str, ScreenRecord] | None = None,
    max_screen_age: timedelta = MAX_SCREEN_AGE,
) -> list[Signal]:
    """Deterministic signals for the core universe (the keys of ``bars``)."""

    signals: list[Signal] = []
    if is_rebalance_day(today, holdings.last_rebalance):
        last = holdings.last_rebalance.isoformat() if holdings.last_rebalance else "never"
        signals.append(
            Signal(
                SignalKind.REBALANCE_DUE,
                BOARD_SYMBOL,
                "info",
                f"monthly rebalance due for {today:%Y-%m} (last rebalance: {last})",
                {"last_rebalance": last},
            )
        )

    closes = {s: [b.close for b in series] for s, series in bars.items() if series}
    try:
        targets = target_weights({s: [float(c) for c in v] for s, v in closes.items()}, mode)
    except ValueError as exc:  # too little history: say so instead of guessing weights
        targets = {}
        signals.append(
            Signal(SignalKind.DRIFT_ALERT, BOARD_SYMBOL, "warning", f"weights unavailable: {exc}")
        )
    equity = holdings.equity
    if targets and equity > 0:
        for symbol in sorted(set(targets) | set(holdings.values)):
            held = holdings.values.get(symbol, Decimal(0)) / equity
            goal = targets.get(symbol, Decimal(0))
            if abs(held - goal) > band:
                signals.append(
                    Signal(
                        SignalKind.DRIFT_ALERT,
                        symbol,
                        "warning",
                        f"{symbol} weight {_pct(held)} vs target {_pct(goal)} (band {_pct(band)})",
                        {"weight": str(held.quantize(Decimal("0.0001"))), "target": str(goal)},
                    )
                )

    if mode is CoreMode.DEFENSIVE:
        for symbol in sorted(closes):
            series = closes[symbol]
            if len(series) < TREND_WINDOW + 1:
                continue
            now_avg = _average(series, len(series), TREND_WINDOW)
            prev_avg = _average(series, len(series) - 1, TREND_WINDOW)
            above_now = series[-1] > now_avg
            above_before = series[-2] > prev_avg
            if above_now == above_before:
                continue
            kind = SignalKind.TREND_ENTRY if above_now else SignalKind.TREND_EXIT
            side = "above" if above_now else "below"
            signals.append(
                Signal(
                    kind,
                    symbol,
                    "info" if above_now else "warning",
                    f"{symbol} closed {side} its {TREND_WINDOW}-day average "
                    f"({series[-1]:.2f} vs {now_avg:.2f})",
                    {"close": str(series[-1]), "average": str(now_avg.quantize(_PCT))},
                )
            )

    for symbol, record in sorted((screening or {}).items()):
        if record.financials is None or record.as_of is None:
            signals.append(
                Signal(
                    SignalKind.SCREEN_STALE,
                    symbol,
                    "warning",
                    f"{symbol} has no dated screening financials; the Shariah gate blocks it",
                    {"as_of": "none"},
                )
            )
            continue
        age = (today - record.as_of).days
        if age > max_screen_age.days:
            signals.append(
                Signal(
                    SignalKind.SCREEN_STALE,
                    symbol,
                    "warning",
                    f"{symbol} screening financials are {age} days old "
                    f"(as of {record.as_of.isoformat()}, limit {max_screen_age.days})",
                    {"as_of": record.as_of.isoformat(), "age_days": str(age)},
                )
            )
    return signals


# ---- summaries in English, Urdu and Arabic ----------------------------------------------

_HEADER = {
    "en": "Core signals (information only, nothing is executed):",
    "ur": "کور سگنلز (صرف معلومات کے لیے، کوئی آرڈر نہیں دیا جاتا):",
    "ar": "إشارات المحفظة الأساسية (للعلم فقط، لا يتم تنفيذ أي أمر):",
}
_NONE = {
    "en": "No core signals today.",
    "ur": "آج کوئی کور سگنل نہیں\u06d4",
    "ar": "لا توجد إشارات اليوم.",
}
_LINES: dict[str, dict[SignalKind, str]] = {
    "en": {
        SignalKind.REBALANCE_DUE: "Monthly rebalance is due.",
        SignalKind.DRIFT_ALERT: "{symbol} has drifted from its target weight.",
        SignalKind.TREND_EXIT: "{symbol} fell below its 200-day average.",
        SignalKind.TREND_ENTRY: "{symbol} rose above its 200-day average.",
        SignalKind.SCREEN_STALE: "{symbol} needs a fresh Shariah screen.",
    },
    "ur": {
        SignalKind.REBALANCE_DUE: "ماہانہ ری بیلنس کا وقت ہو گیا ہے\u06d4",
        SignalKind.DRIFT_ALERT: "{symbol} اپنے ہدف وزن سے ہٹ گیا ہے\u06d4",
        SignalKind.TREND_EXIT: "{symbol} اپنی 200 دن کی اوسط سے نیچے آ گیا ہے\u06d4",
        SignalKind.TREND_ENTRY: "{symbol} اپنی 200 دن کی اوسط سے اوپر چلا گیا ہے\u06d4",
        SignalKind.SCREEN_STALE: "{symbol} کی شرعی اسکریننگ تازہ کرنے کی ضرورت ہے\u06d4",
    },
    "ar": {
        SignalKind.REBALANCE_DUE: "حان موعد إعادة التوازن الشهرية.",
        SignalKind.DRIFT_ALERT: "انحرف {symbol} عن وزنه المستهدف.",
        SignalKind.TREND_EXIT: "انخفض {symbol} تحت متوسط 200 يوم.",
        SignalKind.TREND_ENTRY: "ارتفع {symbol} فوق متوسط 200 يوم.",
        SignalKind.SCREEN_STALE: "يحتاج {symbol} إلى فحص شرعي محدث.",
    },
}


def summarize(signals: Sequence[Signal], lang: str = "en") -> str:
    """Human summary; the English detail (with numbers) follows each localized line."""

    if lang not in _LINES:
        raise ValueError("lang must be 'en', 'ur' or 'ar'")
    if not signals:
        return _NONE[lang]
    lines = [_HEADER[lang]]
    for signal in signals:
        line = _LINES[lang][signal.kind].format(symbol=signal.symbol)
        lines.append(f"- {line} ({signal.detail})" if lang != "en" else f"- {signal.detail}")
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class PublishResult:
    summary: str
    task_id: int | None
    alerted: bool


def publish_signals(
    signals: Sequence[Signal],
    *,
    board: AgentBoard | None = None,
    alerts: AlertSink | None = None,
    lang: str = "en",
) -> PublishResult:
    """Post the summary to the board (risk-officer task) and the alert sinks.

    Nothing is published when there are no signals. The board task asks analysts to review;
    it carries no order and no approval.
    """

    summary = summarize(signals, lang)
    if not signals:
        return PublishResult(summary, None, False)
    task_id = None
    if board is not None:
        task_id = board.post_task(
            BOARD_SYMBOL,
            AnalystRole.RISK_OFFICER,
            f"Review these core signals (information only, nothing executes):\n{summary}",
        )
    alerted = False
    if alerts is not None:
        level = "warning" if any(s.level == "warning" for s in signals) else "info"
        alerts.send(level, SIGNAL_TITLE, summary)
        alerted = True
    return PublishResult(summary, task_id, alerted)
