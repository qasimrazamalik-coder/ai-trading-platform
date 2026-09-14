"""Walk-forward model/parameter selection and fit diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quant_os_core.right_fit.splits import TimeSeriesSplit

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class FitDiagnosis(StrEnum):
    UNDERFIT = "underfit"
    RIGHT_FIT = "right_fit"
    OVERFIT = "overfit"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class WalkForwardSelectionResult:
    selected_candidate_indices: IntArray
    train_scores: FloatArray
    test_scores: FloatArray
    median_test_score: float
    positive_test_fraction: float


def _sharpe_like(matrix: FloatArray) -> FloatArray:
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0, ddof=1)
    result = np.full(means.shape, -np.inf, dtype=np.float64)
    valid = stds > np.finfo(np.float64).eps
    result[valid] = means[valid] / stds[valid]
    return result


def walk_forward_parameter_selection(
    returns_matrix: ArrayLike,
    splits: list[TimeSeriesSplit],
) -> WalkForwardSelectionResult:
    """Select the best candidate in each train window and score it on unseen data.

    Rows represent time observations and columns represent candidate parameter sets
    or models. Selection is performed only on each split's training observations.
    """
    matrix = np.asarray(returns_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("returns_matrix must be two-dimensional with candidates")
    if matrix.shape[0] < 3 or not np.isfinite(matrix).all():
        raise ValueError("returns_matrix must contain at least 3 finite observations")
    if not splits:
        raise ValueError("splits must be non-empty")

    selected: list[int] = []
    train_scores: list[float] = []
    test_scores: list[float] = []
    for split in splits:
        if np.max(split.train) >= matrix.shape[0] or np.max(split.test) >= matrix.shape[0]:
            raise ValueError("split indices exceed returns_matrix rows")
        if split.train.size < 2 or split.test.size < 2:
            raise ValueError("each split needs at least 2 train and test observations")
        train_all = _sharpe_like(matrix[split.train])
        candidate = int(np.argmax(train_all))
        test_all = _sharpe_like(matrix[split.test])
        selected.append(candidate)
        train_scores.append(float(train_all[candidate]))
        test_scores.append(float(test_all[candidate]))

    selected_array = np.asarray(selected, dtype=np.int64)
    train_array = np.asarray(train_scores, dtype=np.float64)
    test_array = np.asarray(test_scores, dtype=np.float64)
    return WalkForwardSelectionResult(
        selected_candidate_indices=selected_array,
        train_scores=train_array,
        test_scores=test_array,
        median_test_score=float(np.median(test_array)),
        positive_test_fraction=float(np.mean(test_array > 0.0)),
    )


def diagnose_fit(
    *,
    train_score: float,
    validation_score: float,
    min_train_score: float = 0.20,
    max_relative_degradation: float = 0.35,
) -> FitDiagnosis:
    """Classify a simple train/validation pattern as underfit/right-fit/overfit.

    This is a diagnostic label, not proof. More demanding Right-Fit gates such as
    PBO, DSR, stability and leakage checks should still be applied.
    """
    train = float(train_score)
    validation = float(validation_score)
    minimum = float(min_train_score)
    max_gap = float(max_relative_degradation)
    if not all(np.isfinite(value) for value in (train, validation, minimum, max_gap)):
        raise ValueError("fit diagnostic inputs must be finite")
    if minimum < 0.0 or max_gap < 0.0:
        raise ValueError("fit diagnostic thresholds must be non-negative")

    if train < minimum and validation < minimum:
        return FitDiagnosis.UNDERFIT
    scale = max(abs(train), np.finfo(np.float64).eps)
    degradation = (train - validation) / scale
    if train >= minimum and (validation <= 0.0 or degradation > max_gap):
        return FitDiagnosis.OVERFIT
    if train >= minimum and validation > 0.0:
        return FitDiagnosis.RIGHT_FIT
    return FitDiagnosis.INCONCLUSIVE
