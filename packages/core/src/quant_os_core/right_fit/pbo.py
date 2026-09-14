"""Probability-of-backtest-overfitting diagnostics using CSCV-style splits."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PBOResult:
    probability: float
    logits: FloatArray
    oos_rank_percentiles: FloatArray
    selected_strategy_indices: IntArray
    split_count: int


def _score_columns(matrix: FloatArray) -> FloatArray:
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0, ddof=1)
    scores = np.full(means.shape, -np.inf, dtype=np.float64)
    valid = stds > np.finfo(np.float64).eps
    scores[valid] = means[valid] / stds[valid]
    return scores


def probability_backtest_overfitting(
    returns_matrix: ArrayLike,
    *,
    n_slices: int = 8,
    max_combinations: int = 5_000,
) -> PBOResult:
    """Estimate PBO from strategy returns with combinatorially symmetric CV.

    Rows are time observations and columns are candidate strategies/configurations.
    In each split, the best in-sample strategy is ranked out of sample. PBO is the
    fraction of selections whose OOS rank falls in the lower half of candidates.
    """
    matrix = np.asarray(returns_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("returns_matrix must be two-dimensional")
    if matrix.shape[0] < 8 or matrix.shape[1] < 2:
        raise ValueError("PBO requires at least 8 observations and 2 strategies")
    if not np.isfinite(matrix).all():
        raise ValueError("returns_matrix must contain only finite values")
    slices = int(n_slices)
    if slices != n_slices or slices < 4 or slices % 2 != 0:
        raise ValueError("n_slices must be an even integer >= 4")
    if slices > matrix.shape[0]:
        raise ValueError("n_slices cannot exceed the observation count")

    blocks = [block for block in np.array_split(np.arange(matrix.shape[0]), slices)]
    choices = list(combinations(range(slices), slices // 2))
    if len(choices) > max_combinations:
        raise ValueError("PBO split count exceeds max_combinations")

    logits: list[float] = []
    rank_percentiles: list[float] = []
    selected_indices: list[int] = []
    strategy_count = matrix.shape[1]

    for train_blocks in choices:
        train_set = set(train_blocks)
        test_blocks = [index for index in range(slices) if index not in train_set]
        train_idx = np.concatenate([blocks[index] for index in train_blocks])
        test_idx = np.concatenate([blocks[index] for index in test_blocks])
        train_scores = _score_columns(matrix[train_idx])
        test_scores = _score_columns(matrix[test_idx])
        selected = int(np.argmax(train_scores))
        ranks = rankdata(test_scores, method="average")
        percentile = float(ranks[selected] / (strategy_count + 1.0))
        percentile = float(np.clip(percentile, 1e-12, 1.0 - 1e-12))
        logit = float(np.log(percentile / (1.0 - percentile)))
        selected_indices.append(selected)
        rank_percentiles.append(percentile)
        logits.append(logit)

    logits_array = np.asarray(logits, dtype=np.float64)
    ranks_array = np.asarray(rank_percentiles, dtype=np.float64)
    selected_array = np.asarray(selected_indices, dtype=np.int64)
    return PBOResult(
        probability=float(np.mean(logits_array <= 0.0)),
        logits=logits_array,
        oos_rank_percentiles=ranks_array,
        selected_strategy_indices=selected_array,
        split_count=len(choices),
    )
