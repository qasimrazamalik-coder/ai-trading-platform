"""Model evaluation helpers that force comparison against simple baselines."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quant_os_core.ml.baselines import zero_return_forecast
from quant_os_core.ml.metrics import (
    directional_accuracy,
    information_coefficient,
    mean_squared_error,
)


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    mse: float
    baseline_mse: float
    mse_improvement: float
    directional_accuracy: float
    information_coefficient: float
    beats_baseline: bool


def evaluate_regression_predictions(
    y_true: npt.ArrayLike,
    predictions: npt.ArrayLike,
) -> ModelEvaluation:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    mse = mean_squared_error(y, p)
    baseline = mean_squared_error(y, zero_return_forecast(y.size))
    improvement = 0.0 if baseline <= 1e-18 else (baseline - mse) / baseline
    return ModelEvaluation(
        mse=mse,
        baseline_mse=baseline,
        mse_improvement=float(improvement),
        directional_accuracy=directional_accuracy(y, p),
        information_coefficient=information_coefficient(y, p),
        beats_baseline=mse < baseline,
    )
