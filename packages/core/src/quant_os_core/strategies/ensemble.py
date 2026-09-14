"""Strategy-ensemble combination functions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def weighted_ensemble(
    signals: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
    clip: bool = True,
) -> npt.NDArray[np.float64]:
    """Combine strategy x time signals into one target-exposure series."""
    matrix = np.asarray(signals, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("signals must be a 2-D strategy x time matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("signals must contain only finite values")
    if np.any(np.abs(matrix) > 1.0 + 1e-12):
        raise ValueError("individual signals must stay inside [-1, 1]")
    if weights is None:
        normalized = np.full(matrix.shape[0], 1.0 / matrix.shape[0], dtype=np.float64)
    else:
        raw_weights = np.asarray(weights, dtype=np.float64)
        if raw_weights.ndim != 1 or raw_weights.size != matrix.shape[0]:
            raise ValueError("weights must match the number of strategies")
        if not np.isfinite(raw_weights).all() or np.any(raw_weights < 0.0):
            raise ValueError("weights must be finite and non-negative")
        total = float(np.sum(raw_weights))
        if total <= np.finfo(np.float64).eps:
            raise ValueError("weights must have positive total")
        normalized = raw_weights / total
    combined = normalized @ matrix
    return np.clip(combined, -1.0, 1.0) if clip else combined


def majority_vote(
    signals: npt.ArrayLike,
    *,
    minimum_agreement: float = 0.5,
) -> npt.NDArray[np.float64]:
    matrix = np.asarray(signals, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("signals must be a 2-D strategy x time matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("signals must contain only finite values")
    threshold = float(minimum_agreement)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("minimum_agreement must be in (0, 1]")
    bullish = np.mean(matrix > 0.0, axis=0)
    bearish = np.mean(matrix < 0.0, axis=0)
    result = np.zeros(matrix.shape[1], dtype=np.float64)
    result[bullish >= threshold] = 1.0
    result[bearish >= threshold] = -1.0
    conflicts = (bullish >= threshold) & (bearish >= threshold)
    result[conflicts] = 0.0
    return result
