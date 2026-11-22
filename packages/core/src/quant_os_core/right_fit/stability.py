"""Generalization and parameter-surface stability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class ParameterStabilityResult:
    score: float
    plateau_fraction: float
    local_stability: float
    global_smoothness: float
    best_index: tuple[int, int]
    best_value: float


def relative_performance_degradation(train_score: float, test_score: float) -> float:
    """Positive fraction means test performance degraded relative to training."""
    train = float(train_score)
    test = float(test_score)
    if not np.isfinite(train) or not np.isfinite(test):
        raise ValueError("scores must be finite")
    scale = max(abs(train), np.finfo(np.float64).eps)
    return float((train - test) / scale)


def complexity_penalized_score(
    performance: float,
    *,
    complexity: float,
    penalty_strength: float,
) -> float:
    perf = float(performance)
    comp = float(complexity)
    penalty = float(penalty_strength)
    if not all(np.isfinite(value) for value in (perf, comp, penalty)):
        raise ValueError("inputs must be finite")
    if comp < 0.0 or penalty < 0.0:
        raise ValueError("complexity and penalty_strength must be non-negative")
    return perf - comp * penalty


def parameter_surface_stability(
    surface: ArrayLike,
    *,
    tolerance: float = 0.10,
    target_plateau_fraction: float = 0.15,
) -> ParameterStabilityResult:
    """Score whether a 2-D parameter surface has a robust plateau instead of a spike."""
    values = np.asarray(surface, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("surface must be a finite 2-D array of at least 2x2")
    if not np.isfinite(values).all():
        raise ValueError("surface must contain only finite values")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be strictly between 0 and 1")
    if not 0.0 < target_plateau_fraction <= 1.0:
        raise ValueError("target_plateau_fraction must be in (0, 1]")

    best_flat = int(np.argmax(values))
    best_row, best_col = np.unravel_index(best_flat, values.shape)
    best_index = (int(best_row), int(best_col))
    best_value = float(values[best_index])
    scale = max(float(np.ptp(values)), abs(best_value), np.finfo(np.float64).eps)
    threshold = best_value - tolerance * max(abs(best_value), np.finfo(np.float64).eps)
    plateau_fraction = float(np.mean(values >= threshold))
    plateau_score = min(1.0, plateau_fraction / target_plateau_fraction)

    row, col = best_index
    r0, r1 = max(0, row - 1), min(values.shape[0], row + 2)
    c0, c1 = max(0, col - 1), min(values.shape[1], col + 2)
    neighborhood = values[r0:r1, c0:c1]
    local_gap = float(np.mean(np.abs(neighborhood - best_value))) / scale
    local_stability = float(np.clip(1.0 - local_gap, 0.0, 1.0))

    horizontal = np.abs(np.diff(values, axis=1)).ravel()
    vertical = np.abs(np.diff(values, axis=0)).ravel()
    mean_gradient = float(np.mean(np.concatenate([horizontal, vertical]))) / scale
    global_smoothness = float(np.exp(-3.0 * mean_gradient))

    score = float(
        np.clip(
            0.40 * plateau_score + 0.35 * local_stability + 0.25 * global_smoothness,
            0.0,
            1.0,
        )
    )
    return ParameterStabilityResult(
        score=score,
        plateau_fraction=plateau_fraction,
        local_stability=local_stability,
        global_smoothness=global_smoothness,
        best_index=best_index,
        best_value=best_value,
    )
