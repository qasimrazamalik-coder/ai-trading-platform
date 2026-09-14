"""Fundamental and technical screening primitives inspired by public trader methodologies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MinerviniInputs:
    price: float
    sma_50: float
    sma_150: float
    sma_200: float
    sma_200_20d_ago: float
    low_52w: float
    high_52w: float
    relative_strength_percentile: float


@dataclass(frozen=True, slots=True)
class MinerviniScreenResult:
    passed: bool
    score: float
    checks: dict[str, bool]


def minervini_trend_template(inputs: MinerviniInputs) -> MinerviniScreenResult:
    values = np.asarray(
        [
            inputs.price,
            inputs.sma_50,
            inputs.sma_150,
            inputs.sma_200,
            inputs.sma_200_20d_ago,
            inputs.low_52w,
            inputs.high_52w,
            inputs.relative_strength_percentile,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("Minervini inputs must be finite")
    if np.any(values[:7] <= 0.0):
        raise ValueError("price and moving-average inputs must be > 0")
    if not 0.0 <= inputs.relative_strength_percentile <= 100.0:
        raise ValueError("relative_strength_percentile must be between 0 and 100")
    checks = {
        "price_above_150_200": inputs.price > inputs.sma_150 and inputs.price > inputs.sma_200,
        "sma_150_above_200": inputs.sma_150 > inputs.sma_200,
        "sma_200_rising": inputs.sma_200 > inputs.sma_200_20d_ago,
        "sma_50_above_150_200": inputs.sma_50 > inputs.sma_150 and inputs.sma_50 > inputs.sma_200,
        "price_above_50": inputs.price > inputs.sma_50,
        "price_30pct_above_52w_low": inputs.price >= 1.30 * inputs.low_52w,
        "price_within_25pct_of_52w_high": inputs.price >= 0.75 * inputs.high_52w,
        "relative_strength_leader": inputs.relative_strength_percentile >= 70.0,
    }
    score = float(np.mean(list(checks.values())))
    return MinerviniScreenResult(passed=all(checks.values()), score=score, checks=checks)


@dataclass(frozen=True, slots=True)
class CanSlimInputs:
    quarterly_eps_growth: float
    annual_eps_growth: float
    revenue_growth: float
    catalyst_score: float
    supply_demand_score: float
    relative_strength_percentile: float
    institutional_sponsorship_score: float
    market_regime_score: float


@dataclass(frozen=True, slots=True)
class CanSlimScore:
    total: float
    components: dict[str, float]


def canslim_score(inputs: CanSlimInputs) -> CanSlimScore:
    raw = {
        "current_earnings": inputs.quarterly_eps_growth,
        "annual_earnings": inputs.annual_eps_growth,
        "new_catalyst": inputs.catalyst_score,
        "supply_demand": inputs.supply_demand_score,
        "relative_strength": inputs.relative_strength_percentile / 100.0,
        "institutional": inputs.institutional_sponsorship_score,
        "market": inputs.market_regime_score,
        "revenue": inputs.revenue_growth,
    }
    if not all(np.isfinite(value) for value in raw.values()):
        raise ValueError("CAN SLIM inputs must be finite")
    components = {
        "current_earnings": float(np.clip(raw["current_earnings"] / 0.25, 0.0, 1.0)),
        "annual_earnings": float(np.clip(raw["annual_earnings"] / 0.25, 0.0, 1.0)),
        "new_catalyst": float(np.clip(raw["new_catalyst"], 0.0, 1.0)),
        "supply_demand": float(np.clip(raw["supply_demand"], 0.0, 1.0)),
        "relative_strength": float(np.clip(raw["relative_strength"], 0.0, 1.0)),
        "institutional": float(np.clip(raw["institutional"], 0.0, 1.0)),
        "market": float(np.clip(raw["market"], 0.0, 1.0)),
        "revenue": float(np.clip(raw["revenue"] / 0.20, 0.0, 1.0)),
    }
    weights = {
        "current_earnings": 0.16,
        "annual_earnings": 0.16,
        "new_catalyst": 0.11,
        "supply_demand": 0.12,
        "relative_strength": 0.16,
        "institutional": 0.10,
        "market": 0.12,
        "revenue": 0.07,
    }
    total = sum(components[key] * weights[key] for key in components)
    return CanSlimScore(total=float(total), components=components)
