"""Deterministic local alert rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Comparison(StrEnum):
    ABOVE = "above"
    BELOW = "below"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"


@dataclass(frozen=True, slots=True)
class AlertRule:
    rule_id: str
    metric: str
    comparison: Comparison
    threshold: float

    def __post_init__(self) -> None:
        if not self.rule_id.strip() or not self.metric.strip():
            raise ValueError("rule_id and metric cannot be blank")

    def triggered(self, current: float, previous: float | None = None) -> bool:
        if self.comparison is Comparison.ABOVE:
            return current > self.threshold
        if self.comparison is Comparison.BELOW:
            return current < self.threshold
        if previous is None:
            return False
        if self.comparison is Comparison.CROSSES_ABOVE:
            return previous <= self.threshold < current
        return previous >= self.threshold > current
