"""Weekly market note: signals and metrics turned into a short advisory note for the owner.

With an LLM callable the note is prose written under a strict prompt (advisory only, no trade
instructions, only the numbers given). The reply is checked in code: if it contains a trade
instruction or any number that is not in the facts, the template note is used instead. The
note is posted to the analyst board; it never reaches the order path and approves nothing.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from quant_os_core.agents.board import AgentBoard, AnalystRole
from quant_os_core.agents.signals import BOARD_SYMBOL, Signal
from quant_os_core.ops.state import StateStore

Generate = Callable[[Sequence[dict[str, str]]], str]
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_INSTRUCTION = re.compile(
    r"\b(you|we|i)\s+(should|must|recommend|suggest)\b[^.]*\b(buy|sell|short|enter|exit|"
    r"trade|order|leverage)\b"
    r"|\b(buy|sell|short)\s+(now|today|tomorrow|immediately|more)\b"
    r"|\bplace (an? )?order\b|\bgo (long|short)\b|\btake profits?\b",
    re.IGNORECASE,
)
NOTE_PROMPT = (
    "You write a short weekly note for the owner of a halal, long-only, paper-trading "
    "portfolio. Rules: the note is advisory information only; never tell anyone to buy, sell, "
    "enter, exit or place an order; never promise or forecast returns; use only the numbers in "
    "the facts and quote them exactly; do not add any other number; if something is unknown, "
    "say so. At most 150 words, plain language, no headings."
)
DISCLAIMER = "Advisory only: nothing in this note is an order or a recommendation to trade."


@dataclass(frozen=True, slots=True)
class MarketNote:
    text: str
    used_llm: bool
    facts: str


def metrics_from_store(store: StateStore) -> dict[str, Decimal]:
    """Equity, peak, drawdown and the change over the last 7 days, from equity snapshots."""

    curve = store.equity_curve()
    if not curve:
        return {}
    last_time, equity = curve[-1]
    peak = max(e for _, e in curve)
    metrics = {"equity": equity, "peak_equity": peak}
    if peak > 0:
        metrics["drawdown_pct"] = ((peak - equity) / peak * 100).quantize(Decimal("0.01"))
    week = [e for t, e in curve if (last_time - t).days >= 7]
    if week and week[-1] > 0:
        change = (equity / week[-1] - 1) * 100
        metrics["week_change_pct"] = change.quantize(Decimal("0.01"))
    return metrics


def _facts(signals: Sequence[Signal], metrics: Mapping[str, Decimal | str]) -> str:
    lines = [f"{key}: {value}" for key, value in sorted(metrics.items())]
    lines += [f"signal {s.kind.value} {s.symbol}: {s.detail}" for s in signals]
    return "\n".join(lines) if lines else "no metrics and no signals this week"


def template_note(signals: Sequence[Signal], metrics: Mapping[str, Decimal | str]) -> str:
    parts = ["Weekly note."]
    if "equity" in metrics:
        parts.append(f"Paper equity is {metrics['equity']}.")
    if "drawdown_pct" in metrics:
        parts.append(f"Drawdown from the peak is {metrics['drawdown_pct']}%.")
    if "week_change_pct" in metrics:
        parts.append(f"Change over the week: {metrics['week_change_pct']}%.")
    for key, value in sorted(metrics.items()):
        if key not in {"equity", "drawdown_pct", "week_change_pct", "peak_equity"}:
            parts.append(f"{key}: {value}.")
    if signals:
        parts.append("Signals: " + "; ".join(s.detail for s in signals) + ".")
    else:
        parts.append("No signals this week.")
    parts.append(DISCLAIMER)
    return " ".join(parts)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def _acceptable(reply: str, facts: str) -> bool:
    if not reply or _INSTRUCTION.search(reply):
        return False
    return _numbers(reply) <= _numbers(facts)


def build_market_note(
    signals: Sequence[Signal],
    metrics: Mapping[str, Decimal | str],
    *,
    llm: Generate | None = None,
    now: datetime | None = None,
) -> MarketNote:
    """Prose from ``llm`` when it passes the checks, else the template note."""

    facts = _facts(signals, metrics)
    if now is not None:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        facts = f"week ending: {now.date().isoformat()}\n{facts}"
    if llm is not None:
        messages = [
            {"role": "system", "content": NOTE_PROMPT},
            {"role": "user", "content": f"Facts:\n{facts}\n\nWrite the note."},
        ]
        try:
            reply = _THINK.sub("", llm(messages)).strip()
        except Exception:  # model offline: the template still informs the owner
            reply = ""
        if _acceptable(reply, facts):
            return MarketNote(f"{reply}\n{DISCLAIMER}", True, facts)
    return MarketNote(template_note(signals, metrics), False, facts)


def post_market_note(note: MarketNote, board: AgentBoard) -> int:
    """Post the note to the board as a macro-analyst task. Never touches orders."""

    return board.post_task(
        BOARD_SYMBOL,
        AnalystRole.MACRO,
        f"Weekly market note (advisory, information only):\n{note.text}",
    )
