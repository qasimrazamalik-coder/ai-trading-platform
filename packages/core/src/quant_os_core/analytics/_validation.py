"""Internal validation helpers for analytics payloads."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def finite_1d(values: ArrayLike, *, name: str, min_size: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < min_size:
        raise ValueError(f"{name} must be a 1-D array with at least {min_size} values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def finite_2d(values: ArrayLike, *, name: str, min_rows: int = 1, min_cols: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < min_rows or array.shape[1] < min_cols:
        raise ValueError(
            f"{name} must be a 2-D array with at least {min_rows} rows and {min_cols} columns"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def labels_tuple(labels: Sequence[str], *, expected: int, name: str) -> tuple[str, ...]:
    result = tuple(str(label).strip() for label in labels)
    if len(result) != expected or any(not label for label in result):
        raise ValueError(f"{name} must contain {expected} non-empty labels")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result
