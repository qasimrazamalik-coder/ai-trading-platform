"""Probability calibration utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class PlattCalibrator:
    """Logistic (Platt) calibration fitted only on held-out calibration data."""

    _model: object | None = None

    def fit(self, scores: npt.ArrayLike, targets: npt.ArrayLike) -> PlattCalibrator:
        from sklearn.linear_model import LogisticRegression

        x = np.asarray(scores, dtype=np.float64)
        y = np.asarray(targets, dtype=np.int64)
        if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 4:
            raise ValueError("scores and targets must be matching 1D arrays")
        valid_classes = set(np.unique(y)).issubset({0, 1}) and np.unique(y).size >= 2
        if not np.isfinite(x).all() or not valid_classes:
            raise ValueError("calibration targets must contain both binary classes")
        model = LogisticRegression(solver="lbfgs")
        model.fit(x.reshape(-1, 1), y)
        self._model = model
        return self

    def predict_proba(self, scores: npt.ArrayLike) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("calibrator must be fit before prediction")
        x = np.asarray(scores, dtype=np.float64)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError("scores must be a finite 1D array")
        return np.asarray(self._model.predict_proba(x.reshape(-1, 1))[:, 1], dtype=np.float64)
