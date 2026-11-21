"""Local LLM analyst through Ollama's chat API with schema-enforced JSON output.

Ollama accepts a JSON schema in the ``format`` field and recommends temperature 0 for more
deterministic output (https://docs.ollama.com/capabilities/structured-outputs). Replies that do
not parse into :class:`AnalystReport` are turned into a zero-conviction neutral report, so a
broken or hallucinating model can never push a trade through (fail closed).
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from quant_os_core.agents.council import (
    ANALYST_JSON_SCHEMA,
    AnalystReport,
    Stance,
    parse_analyst_json,
)
from quant_os_core.agents.trade_plan import TradePlan

Post = Callable[[str, bytes], bytes]


def urllib_post(url: str, body: bytes) -> bytes:
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        reply: bytes = response.read()
    return reply


ROLE_PROMPTS: dict[str, str] = {
    "technical": "You are a technical analyst. Judge trend, momentum and the stop placement.",
    "news": (
        "You are a news analyst. Flag material news; set red_flag for fraud, halts or lawsuits."
    ),
    "fundamental": "You are a fundamentals analyst. Judge valuation, earnings quality and debt.",
    "bear": "You are the designated bear. Argue the strongest honest case AGAINST this trade.",
}


@dataclass(frozen=True, slots=True)
class OllamaAnalyst:
    name: str
    model: str = "qwen3:8b"
    context: str = ""  # news headlines, indicator values, filing excerpts supplied by the caller
    post: Post = urllib_post
    url: str = "http://localhost:11434/api/chat"

    def _request(self, plan: TradePlan) -> bytes:
        role = ROLE_PROMPTS.get(self.name, f"You are the {self.name} analyst.")
        user = (
            f"Proposed long trade in {plan.symbol}: entry {plan.entry_price}, "
            f"stop {plan.stop_loss}, "
            f"target {plan.profit_target}. Invalidation: {plan.invalidation_condition}. "
            f"Rationale: {plan.justification}\nContext:\n{self.context or 'none'}\n"
            f'Reply as JSON with analyst="{self.name}".'
        )
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "format": ANALYST_JSON_SCHEMA,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": role + " You never give financial advice."},
                {"role": "user", "content": user},
            ],
        }
        return json.dumps(body).encode()

    def review(self, plan: TradePlan) -> AnalystReport:
        try:
            raw = json.loads(self.post(self.url, self._request(plan)))
            content = str(raw.get("message", {}).get("content", ""))
        except (OSError, ValueError):
            content = ""
        report = parse_analyst_json(content)
        if report is None or report.analyst != self.name:
            return AnalystReport(
                analyst=self.name,
                stance=Stance.NEUTRAL,
                conviction=Decimal("0"),
                rationale="no valid reply from the model; ignored",
            )
        return report
