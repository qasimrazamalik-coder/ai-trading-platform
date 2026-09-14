"""Leakage-aware time-series splitters for financial model validation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class TimeSeriesSplit:
    """Indices for one time-ordered train/test evaluation."""

    train: IntArray
    test: IntArray

    def __post_init__(self) -> None:
        if self.train.ndim != 1 or self.test.ndim != 1:
            raise ValueError("train and test indices must be one-dimensional")
        if self.train.size == 0 or self.test.size == 0:
            raise ValueError("train and test indices must be non-empty")
        if np.intersect1d(self.train, self.test).size:
            raise ValueError("train and test indices must not overlap")


def _require_non_negative_int(value: int, *, name: str) -> int:
    parsed = int(value)
    if parsed != value or parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _require_positive_int(value: int, *, name: str) -> int:
    parsed = int(value)
    if parsed != value or parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def purged_kfold(
    n_samples: int,
    *,
    n_splits: int = 5,
    purge: int = 0,
    embargo: int = 0,
) -> list[TimeSeriesSplit]:
    """Create contiguous K-fold splits with purge and post-test embargo zones.

    Purge removes observations immediately before each test fold. Embargo removes
    observations immediately after it. These gaps reduce leakage from overlapping
    labels or features with temporal persistence.
    """
    size = _require_positive_int(n_samples, name="n_samples")
    folds = _require_positive_int(n_splits, name="n_splits")
    purge_size = _require_non_negative_int(purge, name="purge")
    embargo_size = _require_non_negative_int(embargo, name="embargo")
    if folds < 2 or folds > size:
        raise ValueError("n_splits must be between 2 and n_samples")

    all_indices = np.arange(size, dtype=np.int64)
    test_folds = np.array_split(all_indices, folds)
    result: list[TimeSeriesSplit] = []
    for test in test_folds:
        start = int(test[0])
        stop = int(test[-1]) + 1
        excluded_start = max(0, start - purge_size)
        excluded_stop = min(size, stop + embargo_size)
        mask = np.ones(size, dtype=bool)
        mask[excluded_start:excluded_stop] = False
        train = all_indices[mask]
        if train.size == 0:
            raise ValueError("purge/embargo settings leave an empty training set")
        result.append(TimeSeriesSplit(train=train, test=test.copy()))
    return result


def walk_forward_splits(
    n_samples: int,
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    gap: int = 0,
    expanding: bool = True,
) -> list[TimeSeriesSplit]:
    """Create expanding- or rolling-window walk-forward splits."""
    size = _require_positive_int(n_samples, name="n_samples")
    train_window = _require_positive_int(train_size, name="train_size")
    test_window = _require_positive_int(test_size, name="test_size")
    step = test_window if step_size is None else _require_positive_int(step_size, name="step_size")
    gap_size = _require_non_negative_int(gap, name="gap")
    if train_window + gap_size + test_window > size:
        raise ValueError("not enough observations for one walk-forward split")

    result: list[TimeSeriesSplit] = []
    test_start = train_window + gap_size
    while test_start + test_window <= size:
        train_end = test_start - gap_size
        train_start = 0 if expanding else train_end - train_window
        train = np.arange(train_start, train_end, dtype=np.int64)
        test = np.arange(test_start, test_start + test_window, dtype=np.int64)
        result.append(TimeSeriesSplit(train=train, test=test))
        test_start += step
    return result


def combinatorial_purged_splits(
    n_samples: int,
    *,
    n_groups: int = 6,
    test_groups: int = 2,
    purge: int = 0,
    embargo: int = 0,
    max_combinations: int = 1_000,
) -> list[TimeSeriesSplit]:
    """Create combinatorial purged CV splits from contiguous temporal groups."""
    size = _require_positive_int(n_samples, name="n_samples")
    groups_count = _require_positive_int(n_groups, name="n_groups")
    held_out = _require_positive_int(test_groups, name="test_groups")
    purge_size = _require_non_negative_int(purge, name="purge")
    embargo_size = _require_non_negative_int(embargo, name="embargo")
    max_count = _require_positive_int(max_combinations, name="max_combinations")
    if groups_count < 2 or groups_count > size:
        raise ValueError("n_groups must be between 2 and n_samples")
    if held_out >= groups_count:
        raise ValueError("test_groups must be smaller than n_groups")

    all_indices = np.arange(size, dtype=np.int64)
    groups = [group.astype(np.int64) for group in np.array_split(all_indices, groups_count)]
    choices = list(combinations(range(groups_count), held_out))
    if len(choices) > max_count:
        raise ValueError("requested CPCV configuration exceeds max_combinations")

    result: list[TimeSeriesSplit] = []
    for choice in choices:
        test = np.sort(np.concatenate([groups[index] for index in choice]))
        excluded = np.zeros(size, dtype=bool)
        for index in choice:
            group = groups[index]
            start = max(0, int(group[0]) - purge_size)
            stop = min(size, int(group[-1]) + 1 + embargo_size)
            excluded[start:stop] = True
        train = all_indices[~excluded]
        if train.size == 0:
            raise ValueError("purge/embargo settings leave an empty CPCV training set")
        result.append(TimeSeriesSplit(train=train, test=test))
    return result
