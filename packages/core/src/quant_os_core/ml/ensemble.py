"""Prediction ensembles with validated weights."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def weighted_predictions(predictions: npt.ArrayLike, weights: npt.ArrayLike) -> np.ndarray:
    matrix = np.asarray(predictions, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or w.ndim != 1 or matrix.shape[0] != w.size or w.size == 0:
        raise ValueError("predictions must be [models, samples] with one weight per model")
    if not np.isfinite(matrix).all() or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("predictions and weights must be finite; weights non-negative")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return np.average(matrix, axis=0, weights=w)
