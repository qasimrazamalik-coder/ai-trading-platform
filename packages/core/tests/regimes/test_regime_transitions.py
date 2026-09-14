import numpy as np
import pytest

from quant_os_core.regimes import (
    confidence_from_probabilities,
    normalized_entropy,
    stationary_distribution,
    transition_matrix,
)


def test_transition_matrix_rows_sum_to_one() -> None:
    matrix = transition_matrix([0, 0, 1, 1, 1, 0], n_states=2)
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert matrix[1, 1] > matrix[1, 0]


def test_transition_matrix_handles_unobserved_state() -> None:
    matrix = transition_matrix([0, 0, 0], n_states=3, smoothing=0.0)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert np.allclose(matrix[1], np.full(3, 1 / 3))


def test_transition_guards() -> None:
    with pytest.raises(ValueError):
        transition_matrix([0])
    with pytest.raises(ValueError):
        transition_matrix([0, -1])
    with pytest.raises(ValueError):
        transition_matrix([0, 2], n_states=2)


def test_stationary_distribution_known_chain() -> None:
    matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    stationary = stationary_distribution(matrix)
    assert stationary.sum() == pytest.approx(1.0)
    assert stationary == pytest.approx([2 / 3, 1 / 3], abs=1e-6)


def test_stationary_distribution_guards() -> None:
    with pytest.raises(ValueError):
        stationary_distribution(np.ones((2, 3)))
    with pytest.raises(ValueError):
        stationary_distribution(np.array([[0.5, 0.6], [0.2, 0.8]]))


def test_entropy_and_confidence() -> None:
    assert normalized_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert normalized_entropy([1.0, 1.0]) == pytest.approx(1.0)
    assert confidence_from_probabilities([1.0, 0.0]) == pytest.approx(1.0)
    assert confidence_from_probabilities([0.5, 0.5]) == pytest.approx(0.0)


def test_entropy_guards() -> None:
    with pytest.raises(ValueError):
        normalized_entropy([])
    with pytest.raises(ValueError):
        normalized_entropy([-1.0, 2.0])
    with pytest.raises(ValueError):
        normalized_entropy([0.0, 0.0])
