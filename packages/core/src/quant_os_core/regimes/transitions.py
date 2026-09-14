"""Regime transition statistics and uncertainty measures."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def transition_matrix(
    states: npt.ArrayLike,
    *,
    n_states: int | None = None,
    smoothing: float = 1e-6,
) -> np.ndarray:
    values = np.asarray(states, dtype=np.int64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("states must contain at least two observations")
    if np.any(values < 0):
        raise ValueError("states must be non-negative integers")
    inferred = int(values.max()) + 1
    total_states = inferred if n_states is None else int(n_states)
    if total_states < inferred or total_states < 1:
        raise ValueError("n_states cannot be smaller than observed state ids")
    if smoothing < 0.0 or not np.isfinite(smoothing):
        raise ValueError("smoothing must be finite and non-negative")

    counts = np.full((total_states, total_states), smoothing, dtype=np.float64)
    for current, nxt in zip(values[:-1], values[1:], strict=True):
        counts[int(current), int(nxt)] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] <= 0.0
    if np.any(zero_rows):
        counts[zero_rows] = 1.0
        row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums


def stationary_distribution(matrix: npt.ArrayLike) -> np.ndarray:
    transition = np.asarray(matrix, dtype=np.float64)
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition matrix must be square")
    if transition.shape[0] == 0 or not np.isfinite(transition).all():
        raise ValueError("transition matrix must be non-empty and finite")
    if np.any(transition < 0.0):
        raise ValueError("transition probabilities cannot be negative")
    row_sums = transition.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        raise ValueError("transition rows must sum to one")

    values, vectors = np.linalg.eig(transition.T)
    index = int(np.argmin(np.abs(values - 1.0)))
    vector = np.real(vectors[:, index])
    if np.sum(vector) < 0.0:
        vector = -vector
    vector = np.maximum(vector, 0.0)
    total = float(np.sum(vector))
    if total <= 1e-15:
        return np.full(transition.shape[0], 1.0 / transition.shape[0])
    return vector / total


def normalized_entropy(probabilities: npt.ArrayLike) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probabilities must be a non-empty vector")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("probabilities must have positive mass")
    p = values / total
    if p.size == 1:
        return 0.0
    positive = p[p > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return float(entropy / np.log(p.size))


def confidence_from_probabilities(probabilities: npt.ArrayLike) -> float:
    """Return 1 - normalized entropy, a compact regime-certainty score."""
    return float(np.clip(1.0 - normalized_entropy(probabilities), 0.0, 1.0))
