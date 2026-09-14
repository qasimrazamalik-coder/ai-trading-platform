"""Model-agnostic permutation feature importance."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from quant_os_core.ml.metrics import mean_squared_error


def permutation_importance(
    predict: object,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    random_state: int = 7,
) -> np.ndarray:
    features = np.asarray(x, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.size:
        raise ValueError("invalid feature/target arrays")
    prediction_fn = getattr(predict, "predict", None)
    if prediction_fn is None:
        raise TypeError("predict object must expose predict(x)")
    baseline = mean_squared_error(targets, prediction_fn(features))
    rng = np.random.default_rng(random_state)
    result = np.zeros(features.shape[1], dtype=np.float64)
    for col in range(features.shape[1]):
        shuffled = features.copy()
        shuffled[:, col] = rng.permutation(shuffled[:, col])
        result[col] = mean_squared_error(targets, prediction_fn(shuffled)) - baseline
    return result
