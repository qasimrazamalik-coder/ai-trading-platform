"""Core regime labels and immutable outputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class TrendRegime(StrEnum):
    BULL = "bull_trend"
    BEAR = "bear_trend"
    SIDEWAYS = "sideways"


class VolatilityRegime(StrEnum):
    LOW = "low_volatility"
    NORMAL = "normal_volatility"
    HIGH = "high_volatility"


class RiskRegime(StrEnum):
    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"


class LiquidityRegime(StrEnum):
    ABUNDANT = "abundant_liquidity"
    NORMAL = "normal_liquidity"
    TIGHT = "tight_liquidity"


class CorrelationRegime(StrEnum):
    LOW = "low_correlation"
    NORMAL = "normal_correlation"
    HIGH = "high_correlation"


class MacroRegime(StrEnum):
    GOLDILOCKS = "growth_up_inflation_down"
    REFLATION = "growth_up_inflation_up"
    STAGFLATION = "growth_down_inflation_up"
    DEFLATION = "growth_down_inflation_down"
    MIXED = "mixed_macro"


@dataclass(frozen=True, slots=True)
class RegimeEstimate:
    label: str
    confidence: float
    score: float

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("regime label cannot be blank")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not np.isfinite(self.score):
            raise ValueError("score must be finite")


@dataclass(frozen=True, slots=True)
class RegimeSnapshot:
    trend: RegimeEstimate
    volatility: RegimeEstimate
    risk: RegimeEstimate
    liquidity: RegimeEstimate | None = None
    correlation: RegimeEstimate | None = None
    macro: RegimeEstimate | None = None
