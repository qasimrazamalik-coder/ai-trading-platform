"""Prediction metrics with finance-specific diagnostics."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _pair(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.size == 0 or y.size != p.size:
        raise ValueError("inputs must be matching non-empty 1D arrays")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("inputs must be finite")
    return y, p


def mean_squared_error(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    y, p = _pair(y_true, y_pred)
    return float(np.mean((y - p) ** 2))


def mean_absolute_error(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    y, p = _pair(y_true, y_pred)
    return float(np.mean(np.abs(y - p)))


def directional_accuracy(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    y, p = _pair(y_true, y_pred)
    return float(np.mean(np.sign(y) == np.sign(p)))


def information_coefficient(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    y, p = _pair(y_true, y_pred)
    if np.std(y) <= 1e-15 or np.std(p) <= 1e-15:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1])


def brier_score(y_true: npt.ArrayLike, probabilities: npt.ArrayLike) -> float:
    y, p = _pair(y_true, probabilities)
    if np.any((y != 0.0) & (y != 1.0)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Brier score expects binary targets and probabilities in [0, 1]")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    y_true: npt.ArrayLike, probabilities: npt.ArrayLike, *, bins: int = 10
) -> float:
    y, p = _pair(y_true, probabilities)
    if bins < 2 or np.any((y != 0.0) & (y != 1.0)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("invalid calibration inputs")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = y.size
    error = 0.0
    for idx in range(bins):
        if idx == bins - 1:
            mask = (p >= edges[idx]) & (p <= edges[idx + 1])
        else:
            mask = (p >= edges[idx]) & (p < edges[idx + 1])
        if not np.any(mask):
            continue
        error += (np.sum(mask) / total) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(error)
