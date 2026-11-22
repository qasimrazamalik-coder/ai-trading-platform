from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from quant_os_core.agents.board import AgentBoard
from quant_os_core.agents.market_note import (
    DISCLAIMER,
    NOTE_PROMPT,
    build_market_note,
    metrics_from_store,
    post_market_note,
    template_note,
)
from quant_os_core.agents.signals import BOARD_SYMBOL, Signal, SignalKind
from quant_os_core.ops.state import StateStore

NOW = datetime(2026, 11, 22, tzinfo=UTC)
SIGNALS = [
    Signal(SignalKind.DRIFT_ALERT, "SPUS", "warning", "SPUS weight 30.00% vs target 25.00%"),
    Signal(SignalKind.SCREEN_STALE, "UMMA", "warning", "UMMA has no dated screening financials"),
]
METRICS = {"equity": Decimal("52.40"), "drawdown_pct": Decimal("3.10")}


def _llm(reply: str):  # type: ignore[no-untyped-def]
    seen: list[Sequence[dict[str, str]]] = []

    def generate(messages: Sequence[dict[str, str]]) -> str:
        seen.append(messages)
        return reply

    return generate, seen


def test_template_without_llm_quotes_the_numbers_and_disclaims() -> None:
    note = build_market_note(SIGNALS, METRICS, now=NOW)
    assert not note.used_llm
    assert "52.40" in note.text and "3.10%" in note.text and "SPUS weight 30.00%" in note.text
    assert note.text.endswith(DISCLAIMER)
    assert "No signals this week." in template_note([], {})


def test_llm_prose_is_used_when_it_is_advisory_and_cites_given_numbers() -> None:
    generate, seen = _llm(
        "<think>draft</think>Paper equity stands at 52.40 with a 3.10% drawdown. SPUS sits "
        "at 30.00% against a 25.00% target, and UMMA still lacks screening data."
    )
    note = build_market_note(SIGNALS, METRICS, llm=generate, now=NOW)
    assert note.used_llm and note.text.startswith("Paper equity stands at 52.40")
    assert note.text.endswith(DISCLAIMER) and "<think>" not in note.text
    system, user = seen[0]
    assert system["content"] == NOTE_PROMPT and "never tell anyone to buy" in NOTE_PROMPT
    assert "equity: 52.40" in user["content"] and "week ending: 2026-11-22" in user["content"]


@pytest.mark.parametrize(
    "reply",
    [
        "Equity is 52.40. You should buy more SPUS now.",
        "Drawdown is 3.10%. Sell today before it falls.",
        "Equity is 52.40; place an order to rebalance.",
        "Equity could reach 80.00 next month.",  # number not in the facts
        "",
    ],
)
def test_llm_reply_with_instructions_or_invented_numbers_falls_back(reply: str) -> None:
    generate, _ = _llm(reply)
    note = build_market_note(SIGNALS, METRICS, llm=generate, now=NOW)
    assert not note.used_llm and note.text == template_note(SIGNALS, METRICS)


def test_offline_llm_falls_back_and_naive_time_is_rejected() -> None:
    def offline(_m: Sequence[dict[str, str]]) -> str:
        raise ConnectionError("ollama down")

    assert not build_market_note(SIGNALS, METRICS, llm=offline).used_llm
    with pytest.raises(ValueError):
        build_market_note(SIGNALS, METRICS, now=datetime(2026, 11, 22))


def test_metrics_from_store_and_post_to_board_only(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "core.db")
    assert metrics_from_store(store) == {}
    for days, equity in ((14, "50"), (7, "55"), (0, "52.25")):
        store.record_equity(NOW - timedelta(days=days), Decimal(equity), Decimal("1"))
    metrics = metrics_from_store(store)
    assert metrics["equity"] == Decimal("52.25") and metrics["peak_equity"] == Decimal("55")
    assert metrics["drawdown_pct"] == Decimal("5.00")
    assert metrics["week_change_pct"] == Decimal("-5.00")

    board = AgentBoard(tmp_path / "board.db")
    note = build_market_note(SIGNALS, metrics, now=NOW)
    task_id = post_market_note(note, board)
    task = board.open_tasks()[0]
    assert task["id"] == task_id and task["symbol"] == BOARD_SYMBOL and task["role"] == "macro"
    assert task["question"].startswith("Weekly market note (advisory")
    assert not board.proposals() and store.open_positions() == [] and store.pending() == []
