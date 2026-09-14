import numpy as np
import pytest

from quant_os_core.right_fit import (
    complexity_penalized_score,
    parameter_surface_stability,
    relative_performance_degradation,
)


def test_relative_degradation() -> None:
    assert relative_performance_degradation(2.0, 1.5) == pytest.approx(0.25)
    assert relative_performance_degradation(1.0, 1.2) == pytest.approx(-0.2)


def test_complexity_penalty() -> None:
    score = complexity_penalized_score(1.5, complexity=4.0, penalty_strength=0.1)
    assert score == pytest.approx(1.1)


def test_broad_plateau_scores_above_isolated_spike() -> None:
    broad = np.array(
        [
            [0.90, 0.95, 0.96, 0.94],
            [0.93, 1.00, 0.99, 0.95],
            [0.92, 0.98, 0.97, 0.94],
            [0.88, 0.91, 0.92, 0.90],
        ]
    )
    spike = np.full((4, 4), 0.2)
    spike[2, 2] = 1.0
    assert parameter_surface_stability(broad).score > parameter_surface_stability(spike).score


def test_stability_identifies_best_index() -> None:
    surface = [[0.1, 0.2], [0.3, 0.9]]
    result = parameter_surface_stability(surface)
    assert result.best_index == (1, 1)
    assert result.best_value == pytest.approx(0.9)


def test_stability_rejects_non_matrix() -> None:
    with pytest.raises(ValueError):
        parameter_surface_stability([1.0, 2.0])
