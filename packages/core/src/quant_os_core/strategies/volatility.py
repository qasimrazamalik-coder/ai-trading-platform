"""Volatility-aware strategy primitives."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from quant_os_core.quant._validation import as_float_array


def volatility_breakout_signal(
    returns: npt.ArrayLike,
    *,
    window: int = 20,
    threshold_std: float = 2.0,
) -> npt.NDArray[np.float64]:
    values = as_float_array(returns, name="returns", min_size=3)
    size = int(window)
    threshold = float(threshold_std)
    if size != window or size < 2 or size >= values.size:
        raise ValueError("window must be an integer in [2, len(returns)-1]")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold_std must be finite and > 0")
    result = np.zeros(values.size, dtype=np.float64)
    for index in range(size, values.size):
        history = values[index - size : index]
        mean = float(np.mean(history))
        std = float(np.std(history, ddof=1))
        if std <= np.finfo(np.float64).eps:
            difference = values[index] - mean
            if difference > np.finfo(np.float64).eps:
                result[index] = 1.0
            elif difference < -np.finfo(np.float64).eps:
                result[index] = -1.0
            continue
        z = (values[index] - mean) / std
        if z > threshold:
            result[index] = 1.0
        elif z < -threshold:
            result[index] = -1.0
    return result


def inverse_volatility_weights(volatilities: npt.ArrayLike) -> npt.NDArray[np.float64]:
    values = as_float_array(volatilities, name="volatilities")
    if np.any(values <= 0.0):
        raise ValueError("volatilities must be > 0")
    inverse = 1.0 / values
    return inverse / np.sum(inverse)
