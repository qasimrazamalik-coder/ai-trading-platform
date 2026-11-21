"""Local analyst worker: a free, offline model (Qwen3 via Ollama) answers board tasks.

This is the third council voice next to Claude and ChatGPT, so proposals can reach the
two-source consensus even when only one cloud assistant is in use. Replies must parse into
an analyst report (JSON schema, temperature 0); anything else is skipped, never guessed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from quant_os_core.agents.board import AgentBoard, AnalystRole
from quant_os_core.agents.council import parse_analyst_json

Generate = Callable[[Sequence[dict[str, str]]], str]
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

ROLE_BRIEFS: dict[AnalystRole, str] = {
    AnalystRole.FUNDAMENTALS: "Judge valuation, earnings quality, debt and cash flow.",
    AnalystRole.NEWS_SENTIMENT: "Judge recent news and sentiment; red_flag for fraud, halts, "
    "lawsuits or delisting risk.",
    AnalystRole.TECHNICAL: "Judge trend, momentum, volatility and where a stop belongs.",
    AnalystRole.MACRO: "Judge rates, currency, sector and market regime effects.",
    AnalystRole.RISK_OFFICER: "Judge liquidity, gap risk, event risk and correlation with "
    "existing holdings. Prefer caution.",
    AnalystRole.SHARIAH_REVIEWER: "Judge whether the business or token could breach Shariah "
    "screens (interest, gambling, alcohol, derivatives); red_flag if likely.",
    AnalystRole.BEAR_ADVOCATE: "Argue the strongest honest case AGAINST buying. Never bullish.",
}


@dataclass(frozen=True, slots=True)
class ShiftResult:
    answered: int
    skipped: int


def run_local_shift(
    board: AgentBoard,
    generate: Generate,
    *,
    source: str = "qwen-local",
    max_tasks: int = 10,
    context: Callable[[str], str] | None = None,
) -> ShiftResult:
    """Answer up to ``max_tasks`` open tasks. ``context(symbol)`` may supply local data."""

    answered = skipped = 0
    for task in board.open_tasks(max_tasks):
        role = AnalystRole(task["role"])
        facts = context(task["symbol"]) if context else "No extra data supplied."
        messages = [
            {
                "role": "system",
                "content": f"You are the {role.value} analyst for a halal, long-only, spot-only "
                f"trading system. {ROLE_BRIEFS[role]} Reply only with JSON: analyst, stance "
                "(bullish|bearish|neutral), conviction (0-1), rationale (20-800 chars), "
                "red_flag (true/false). If you lack information, say neutral with low conviction.",
            },
            {
                "role": "user",
                "content": f"Symbol: {task['symbol']}\nTask: {task['question']}\nData:\n{facts}",
            },
        ]
        try:
            raw = _THINK.sub("", generate(messages)).strip()
        except Exception:
            skipped += 1
            continue
        report = parse_analyst_json(raw)
        if report is None or len(report.rationale) < 20:
            skipped += 1
            continue
        try:
            board.submit_report(
                symbol=task["symbol"],
                role=role,
                source=source,
                stance=report.stance,
                conviction=report.conviction,
                rationale=report.rationale,
                red_flag=report.red_flag,
                task_id=int(task["id"]),
            )
        except ValueError:  # e.g. a bullish "bear advocate": rejected, not coerced
            skipped += 1
            continue
        answered += 1
    return ShiftResult(answered, skipped)


def ollama_json_generate(
    model: str = "qwen3:8b", url: str = "http://localhost:11434/api/chat"
) -> Generate:
    """Ollama chat with the analyst-report JSON schema in ``format`` and temperature 0."""

    import urllib.request

    from quant_os_core.agents.council import ANALYST_JSON_SCHEMA

    def call(messages: Sequence[dict[str, str]]) -> str:
        body = json.dumps(
            {
                "model": model,
                "messages": list(messages),
                "stream": False,
                "format": ANALYST_JSON_SCHEMA,
                "options": {"temperature": 0},
            }
        ).encode()
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read())
        return str(payload.get("message", {}).get("content", ""))

    return call
