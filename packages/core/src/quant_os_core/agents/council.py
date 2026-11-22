"""TradingAgents-style analyst council, restricted to a veto-and-shrink role.

TradingAgents (TauricResearch) runs fundamentals, sentiment, news and technical analysts,
then a bull/bear debate, a trader, a risk team and a portfolio manager. Here the analysts are
pluggable (rule-based today, local Ollama LLMs later) and their output is a typed report.

Safety rules that differ from the original framework on purpose:

* the council can only *reduce* size or veto; it can never raise size or open a trade itself;
* the bear case is always argued, so confirmation bias cannot pass unchallenged;
* LLM output must parse into :class:`AnalystReport` or it is ignored (fail closed).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from quant_os_core.agents.trade_plan import TradePlan


class Stance(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class AnalystReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    analyst: str = Field(min_length=1, max_length=40)
    stance: Stance
    conviction: Decimal = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=800)
    red_flag: bool = False  # e.g. fraud news, trading halt, Shariah status change


class Analyst(Protocol):
    @property
    def name(self) -> str: ...

    def review(self, plan: TradePlan) -> AnalystReport: ...


ANALYST_JSON_SCHEMA: dict[str, object] = AnalystReport.model_json_schema()


def parse_analyst_json(raw: str) -> AnalystReport | None:
    """Parse an LLM reply (e.g. Ollama ``format=schema`` output); invalid replies are dropped."""

    try:
        return AnalystReport.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError):
        return None


@dataclass(frozen=True, slots=True)
class CouncilPolicy:
    veto_on_red_flag: bool = True
    veto_bear_conviction: Decimal = Decimal("0.75")  # one strong bear vetoes
    min_size_multiplier: Decimal = Decimal("0.25")  # below this the trade is skipped
    require_bear_case: bool = True


@dataclass(frozen=True, slots=True)
class CouncilDecision:
    approved: bool
    size_multiplier: Decimal  # always in [0, 1]
    bull_case: tuple[str, ...]
    bear_case: tuple[str, ...]
    reason: str


class AnalystCouncil:
    def __init__(self, analysts: Sequence[Analyst], policy: CouncilPolicy | None = None) -> None:
        if not analysts:
            raise ValueError("the council needs at least one analyst")
        self.analysts = tuple(analysts)
        self.policy = policy or CouncilPolicy()

    def deliberate(self, plan: TradePlan) -> CouncilDecision:
        reports = [analyst.review(plan) for analyst in self.analysts]
        bull = tuple(f"{r.analyst}: {r.rationale}" for r in reports if r.stance is Stance.BULLISH)
        bear = tuple(f"{r.analyst}: {r.rationale}" for r in reports if r.stance is Stance.BEARISH)
        if self.policy.require_bear_case and not bear:
            bear = ("devil's advocate: no analyst argued against; size reduced for safety",)
        if self.policy.veto_on_red_flag and any(r.red_flag for r in reports):
            return CouncilDecision(False, Decimal("0"), bull, bear, "red flag raised")
        strongest_bear = max(
            (r.conviction for r in reports if r.stance is Stance.BEARISH), default=Decimal("0")
        )
        if strongest_bear >= self.policy.veto_bear_conviction:
            return CouncilDecision(False, Decimal("0"), bull, bear, "strong bear case vetoed")
        net = sum(
            (
                r.conviction if r.stance is Stance.BULLISH else -r.conviction
                for r in reports
                if r.stance is not Stance.NEUTRAL
            ),
            Decimal("0"),
        ) / Decimal(len(reports))
        multiplier = max(Decimal("0"), min(Decimal("1"), Decimal("0.5") + net))
        if not any(r.stance is Stance.BEARISH for r in reports) and self.policy.require_bear_case:
            multiplier = min(multiplier, Decimal("0.9"))
        if multiplier < self.policy.min_size_multiplier:
            return CouncilDecision(False, multiplier, bull, bear, "council too negative")
        return CouncilDecision(True, multiplier, bull, bear, "approved with size multiplier")


@dataclass(frozen=True, slots=True)
class RuleAnalyst:
    """A deterministic analyst driven by one numeric signal in [-1, 1]."""

    name: str
    signal: Decimal
    rationale_up: str = "signal supports the trade"
    rationale_down: str = "signal argues against the trade"
    flag: bool = False

    def review(self, plan: TradePlan) -> AnalystReport:
        if self.signal > Decimal("0.1"):
            stance, text = Stance.BULLISH, self.rationale_up
        elif self.signal < Decimal("-0.1"):
            stance, text = Stance.BEARISH, self.rationale_down
        else:
            stance, text = Stance.NEUTRAL, "no clear signal"
        return AnalystReport(
            analyst=self.name,
            stance=stance,
            conviction=min(Decimal("1"), abs(self.signal)),
            rationale=f"{plan.symbol}: {text}",
            red_flag=self.flag,
        )
