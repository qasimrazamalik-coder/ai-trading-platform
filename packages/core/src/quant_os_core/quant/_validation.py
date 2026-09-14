"""Numerical validation helpers for quantitative calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def as_float_array(
    values: ArrayLike,
    *,
    name: str = "values",
    min_size: int = 1,
    allow_infinite: bool = False,
) -> FloatArray:
    """Return a copied 1-D float64 array after strict validation."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} observations")
    if np.isnan(array).any():
        raise ValueError(f"{name} must not contain NaN")
    if not allow_infinite and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def paired_float_arrays(
    left: ArrayLike,
    right: ArrayLike,
    *,
    left_name: str = "left",
    right_name: str = "right",
    min_size: int = 2,
) -> tuple[FloatArray, FloatArray]:
    x = as_float_array(left, name=left_name, min_size=min_size)
    y = as_float_array(right, name=right_name, min_size=min_size)
    if x.size != y.size:
        raise ValueError(f"{left_name} and {right_name} must have equal length")
    return x, y


def require_positive(value: float, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return parsed


def require_probability(value: float, *, name: str = "probability") -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return parsed


def require_confidence(value: float) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    return parsed
