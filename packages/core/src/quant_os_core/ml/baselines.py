"""Simple baselines that advanced models must beat out of sample."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def zero_return_forecast(length: int) -> np.ndarray:
    if length < 1:
        raise ValueError("length must be positive")
    return np.zeros(length, dtype=np.float64)


def historical_mean_forecast(train_targets: npt.ArrayLike, length: int) -> np.ndarray:
    y = np.asarray(train_targets, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or length < 1 or not np.isfinite(y).all():
        raise ValueError("invalid inputs")
    return np.full(length, float(np.mean(y)), dtype=np.float64)
