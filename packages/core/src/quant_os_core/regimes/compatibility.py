"""Regime-conditioned strategy compatibility and dynamic allocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class StrategyRegimeScore:
    strategy_id: str
    regime: str
    compatibility: float

    def __post_init__(self) -> None:
        if not self.strategy_id.strip() or not self.regime.strip():
            raise ValueError("strategy_id and regime cannot be blank")
        if not np.isfinite(self.compatibility) or not 0.0 <= self.compatibility <= 1.0:
            raise ValueError("compatibility must be between 0 and 1")


class StrategyRegimeMatrix:
    def __init__(self, scores: tuple[StrategyRegimeScore, ...] = ()) -> None:
        self._scores: dict[tuple[str, str], float] = {}
        for score in scores:
            self.set(score.strategy_id, score.regime, score.compatibility)

    def set(self, strategy_id: str, regime: str, compatibility: float) -> None:
        score = StrategyRegimeScore(strategy_id, regime, compatibility)
        self._scores[(score.strategy_id, score.regime)] = score.compatibility

    def get(self, strategy_id: str, regime: str, *, default: float = 0.5) -> float:
        if not 0.0 <= default <= 1.0:
            raise ValueError("default compatibility must be between 0 and 1")
        return self._scores.get((strategy_id, regime), default)

    def strategies(self) -> tuple[str, ...]:
        return tuple(sorted({strategy for strategy, _ in self._scores}))

    def regimes(self) -> tuple[str, ...]:
        return tuple(sorted({regime for _, regime in self._scores}))

    def heatmap(self) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
        strategies = self.strategies()
        regimes = self.regimes()
        matrix = np.empty((len(strategies), len(regimes)), dtype=np.float64)
        for i, strategy in enumerate(strategies):
            for j, regime in enumerate(regimes):
                matrix[i, j] = self.get(strategy, regime)
        return strategies, regimes, matrix


def dynamic_strategy_weights(
    base_weights: dict[str, float],
    *,
    regime: str,
    compatibility: StrategyRegimeMatrix,
    performance_scores: dict[str, float] | None = None,
    confidence: float = 1.0,
    concentration_power: float = 1.0,
) -> dict[str, float]:
    if not base_weights:
        raise ValueError("base_weights cannot be empty")
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not np.isfinite(concentration_power) or concentration_power <= 0.0:
        raise ValueError("concentration_power must be positive")
    performance = performance_scores or {}
    raw: dict[str, float] = {}
    for strategy, base in base_weights.items():
        if not np.isfinite(base) or base < 0.0:
            raise ValueError("base strategy weights must be finite and non-negative")
        perf = performance.get(strategy, 1.0)
        if not np.isfinite(perf) or perf < 0.0:
            raise ValueError("performance scores must be finite and non-negative")
        regime_score = compatibility.get(strategy, regime)
        blended = (1.0 - confidence) + confidence * regime_score
        raw[strategy] = float(base * perf * blended) ** concentration_power
    total = float(sum(raw.values()))
    if total <= 0.0:
        raise ValueError("strategy weights have zero total mass")
    return {strategy: weight / total for strategy, weight in raw.items()}


def composite_strategy_weights(
    base_weights: dict[str, float],
    *,
    active_regimes: dict[str, float],
    compatibility: StrategyRegimeMatrix,
    performance_scores: dict[str, float] | None = None,
) -> dict[str, float]:
    """Blend multiple regime axes using confidence-weighted geometric compatibility."""
    if not active_regimes:
        raise ValueError("active_regimes cannot be empty")
    for regime, confidence in active_regimes.items():
        if not regime.strip():
            raise ValueError("regime labels cannot be blank")
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("regime confidence must be between 0 and 1")
    performance = performance_scores or {}
    raw: dict[str, float] = {}
    for strategy, base in base_weights.items():
        if not np.isfinite(base) or base < 0.0:
            raise ValueError("base strategy weights must be finite and non-negative")
        perf = performance.get(strategy, 1.0)
        if not np.isfinite(perf) or perf < 0.0:
            raise ValueError("performance scores must be finite and non-negative")
        multiplier = 1.0
        for regime, confidence in active_regimes.items():
            score = max(compatibility.get(strategy, regime), 1e-12)
            multiplier *= score**confidence
        raw[strategy] = float(base * perf * multiplier)
    total = float(sum(raw.values()))
    if total <= 0.0:
        raise ValueError("strategy weights have zero total mass")
    return {strategy: value / total for strategy, value in raw.items()}
