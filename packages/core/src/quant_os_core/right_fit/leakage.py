"""Temporal leakage guards for labels and validation boundaries."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


def _as_index_array(values: ArrayLike, *, name: str) -> IntArray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.issubdtype(raw.dtype, np.integer):
        numeric = np.asarray(values, dtype=np.float64)
        if not np.isfinite(numeric).all() or not np.all(numeric == np.floor(numeric)):
            raise ValueError(f"{name} must contain integer indices")
    parsed = np.asarray(values, dtype=np.int64)
    if np.any(parsed < 0):
        raise ValueError(f"{name} must contain non-negative indices")
    return parsed


def assert_monotonic_unique_timestamps(timestamps: ArrayLike) -> None:
    """Reject duplicated or non-increasing timestamps without losing integer precision."""
    raw = np.asarray(timestamps)
    if raw.ndim != 1 or raw.size < 2:
        raise ValueError("timestamps must be one-dimensional with at least 2 observations")
    if np.issubdtype(raw.dtype, np.datetime64):
        values = raw.astype("datetime64[ns]").astype(np.int64)
    elif np.issubdtype(raw.dtype, np.integer):
        values = raw.astype(np.int64, copy=False)
    else:
        values = np.asarray(timestamps, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("timestamps must contain only finite values")
    if np.any(np.diff(values) <= 0):
        raise ValueError("timestamps must be strictly increasing and unique")


def train_test_overlap(train_indices: ArrayLike, test_indices: ArrayLike) -> IntArray:
    train = _as_index_array(train_indices, name="train_indices")
    test = _as_index_array(test_indices, name="test_indices")
    return np.intersect1d(train, test).astype(np.int64)


def interval_overlap_mask(
    starts: ArrayLike,
    ends: ArrayLike,
    *,
    test_start: int,
    test_end: int,
) -> BoolArray:
    """Return intervals that overlap the inclusive test interval."""
    start_values = _as_index_array(starts, name="starts")
    end_values = _as_index_array(ends, name="ends")
    if start_values.size != end_values.size:
        raise ValueError("starts and ends must have equal length")
    if np.any(end_values < start_values):
        raise ValueError("interval ends must be >= starts")
    if test_start < 0 or test_end < test_start:
        raise ValueError("test interval must be non-negative and ordered")
    return (start_values <= test_end) & (end_values >= test_start)


def purge_label_overlap(
    train_indices: ArrayLike,
    label_end_indices: ArrayLike,
    test_indices: ArrayLike,
    *,
    embargo: int = 0,
) -> IntArray:
    """Remove train samples whose label horizons overlap the test/embargo interval.

    ``label_end_indices[i]`` is the last index used to determine the label for
    observation ``i``. This is the financial-label equivalent of purging samples
    whose future information crosses a validation boundary.
    """
    train = _as_index_array(train_indices, name="train_indices")
    test = _as_index_array(test_indices, name="test_indices")
    label_ends = _as_index_array(label_end_indices, name="label_end_indices")
    embargo_size = int(embargo)
    if embargo_size != embargo or embargo_size < 0:
        raise ValueError("embargo must be a non-negative integer")
    if np.any(train >= label_ends.size):
        raise ValueError("label_end_indices must cover every train index")

    test_start = int(np.min(test))
    test_end = int(np.max(test)) + embargo_size
    starts = train
    ends = label_ends[train]
    keep = ~interval_overlap_mask(starts, ends, test_start=test_start, test_end=test_end)
    return train[keep].astype(np.int64)
