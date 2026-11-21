"""Pre-registered trade plans, as used by the Alpha Arena (nof1) trading agents.

Season 1 of Alpha Arena required every model action to carry a confidence score and an
exit plan (profit target, stop loss and an objective *invalidation condition*); nof1 reported
that these fields improved performance. The winner, Qwen3-Max, traded rarely (about 43 trades
in 17 days) and cut losers. This module encodes those habits as hard rules:

* every entry has a stop, a target and a written invalidation condition;
* reward-to-risk and minimum stop distance are enforced;
* size comes from a fixed risk budget scaled by confidence (never by an LLM directly);
* the governor caps trade frequency and forbids pyramiding (one position per symbol).

Leverage fields from the original format are intentionally absent: this system is unlevered.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TradePlan(BaseModel):
    """A long-only spot entry with its exit plan registered up front."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1, max_length=64)
    entry_price: Decimal = Field(gt=0)
    profit_target: Decimal = Field(gt=0)
    stop_loss: Decimal = Field(gt=0)
    invalidation_condition: str = Field(min_length=8, max_length=300)
    confidence: Decimal = Field(ge=0, le=1)
    justification: str = Field(min_length=1, max_length=1000)

    @field_validator("entry_price", "profit_target", "stop_loss", "confidence", mode="before")
    @classmethod
    def reject_binary_float(cls, value: object) -> object:
        if isinstance(value, float):
            raise ValueError("prices and confidence must use Decimal, int or string")
        return value

    @model_validator(mode="after")
    def check_geometry(self) -> TradePlan:
        if not self.stop_loss < self.entry_price < self.profit_target:
            raise ValueError("long plan requires stop_loss < entry_price < profit_target")
        return self

    @property
    def risk_per_unit(self) -> Decimal:
        return self.entry_price - self.stop_loss

    @property
    def reward_per_unit(self) -> Decimal:
        return self.profit_target - self.entry_price

    @property
    def reward_to_risk(self) -> Decimal:
        return self.reward_per_unit / self.risk_per_unit

    @property
    def stop_distance(self) -> Decimal:
        return self.risk_per_unit / self.entry_price


@dataclass(frozen=True, slots=True)
class PlanPolicy:
    min_confidence: Decimal = Decimal("0.55")
    min_reward_to_risk: Decimal = Decimal("1.5")
    min_stop_distance: Decimal = Decimal("0.01")
    max_stop_distance: Decimal = Decimal("0.15")
    risk_per_trade: Decimal = Decimal("0.01")  # fraction of equity lost if the stop is hit
    max_position_fraction: Decimal = Decimal("0.20")
    max_trades_per_day: int = 3
    quantity_step: Decimal = Decimal("0.0001")  # fractional shares

    def __post_init__(self) -> None:
        if not Decimal("0") < self.risk_per_trade <= Decimal("0.05"):
            raise ValueError("risk_per_trade must be in (0, 0.05]")
        if not Decimal("0") < self.max_position_fraction <= Decimal("1"):
            raise ValueError("max_position_fraction must be in (0, 1]")
        if self.max_trades_per_day < 1:
            raise ValueError("max_trades_per_day must be at least 1")
        if self.min_stop_distance >= self.max_stop_distance:
            raise ValueError("min_stop_distance must be below max_stop_distance")


@dataclass(frozen=True, slots=True)
class PlanVerdict:
    accepted: bool
    reason: str
    quantity: Decimal = Decimal("0")


def validate_plan(plan: TradePlan, policy: PlanPolicy) -> PlanVerdict:
    if plan.confidence < policy.min_confidence:
        return PlanVerdict(False, f"confidence {plan.confidence} below {policy.min_confidence}")
    if plan.reward_to_risk < policy.min_reward_to_risk:
        return PlanVerdict(False, f"reward/risk {plan.reward_to_risk:.2f} too low")
    if plan.stop_distance < policy.min_stop_distance:
        return PlanVerdict(False, "stop too tight; noise would trigger it")
    if plan.stop_distance > policy.max_stop_distance:
        return PlanVerdict(False, "stop too wide for the risk budget")
    return PlanVerdict(True, "plan meets policy")


def size_plan(plan: TradePlan, policy: PlanPolicy, equity: Decimal, free_cash: Decimal) -> Decimal:
    """Fixed-fractional sizing scaled by confidence above the minimum, capped, unlevered."""

    if equity <= 0 or free_cash < 0:
        raise ValueError("equity must be positive and free_cash non-negative")
    span = Decimal("1") - policy.min_confidence
    scale = Decimal("1") if span == 0 else (plan.confidence - policy.min_confidence) / span
    scale = Decimal("0.5") + Decimal("0.5") * max(Decimal("0"), min(Decimal("1"), scale))
    risk_quantity = equity * policy.risk_per_trade * scale / plan.risk_per_unit
    cap_quantity = equity * policy.max_position_fraction / plan.entry_price
    cash_quantity = free_cash / plan.entry_price
    raw = min(risk_quantity, cap_quantity, cash_quantity)
    return (raw / policy.quantity_step).to_integral_value(ROUND_DOWN) * policy.quantity_step


@dataclass(slots=True)
class PlanGovernor:
    """Low trade frequency and no pyramiding: the Alpha Arena winner's habits."""

    policy: PlanPolicy = field(default_factory=PlanPolicy)
    _entries: deque[datetime] = field(default_factory=deque)

    def check(self, plan: TradePlan, open_symbols: set[str], now: datetime) -> PlanVerdict:
        if plan.symbol.upper() in {s.upper() for s in open_symbols}:
            return PlanVerdict(False, "already holding this symbol; pyramiding is not allowed")
        while self._entries and now - self._entries[0] >= timedelta(days=1):
            self._entries.popleft()
        if len(self._entries) >= self.policy.max_trades_per_day:
            return PlanVerdict(False, "daily trade limit reached; waiting for a better setup")
        return PlanVerdict(True, "within frequency limits")

    def record_entry(self, now: datetime) -> None:
        self._entries.append(now)
