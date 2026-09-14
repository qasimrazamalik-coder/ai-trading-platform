"""Core statistics used across research, risk and portfolio modules."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from quant_os_core.quant._validation import FloatArray, as_float_array, paired_float_arrays


def arithmetic_mean(values: ArrayLike) -> float:
    x = as_float_array(values, min_size=1)
    return float(np.mean(x))


def sample_variance(values: ArrayLike) -> float:
    x = as_float_array(values, min_size=2)
    return float(np.var(x, ddof=1))


def sample_std(values: ArrayLike) -> float:
    return float(np.sqrt(sample_variance(values)))


def covariance(left: ArrayLike, right: ArrayLike) -> float:
    x, y = paired_float_arrays(left, right)
    return float(np.cov(x, y, ddof=1)[0, 1])


def correlation(left: ArrayLike, right: ArrayLike) -> float:
    x, y = paired_float_arrays(left, right)
    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)
    if sx == 0.0 or sy == 0.0:
        raise ValueError("correlation is undefined for a constant series")
    return float(np.corrcoef(x, y)[0, 1])


def zscore(values: ArrayLike, *, ddof: int = 0) -> FloatArray:
    x = as_float_array(values, min_size=2)
    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1")
    std = float(np.std(x, ddof=ddof))
    if std == 0.0:
        raise ValueError("z-score is undefined for a constant series")
    return (x - np.mean(x)) / std


def rolling_mean(values: ArrayLike, window: int) -> FloatArray:
    x = as_float_array(values, min_size=1)
    if window <= 0 or window > x.size:
        raise ValueError("window must be in [1, len(values)]")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="valid")


def rolling_std(values: ArrayLike, window: int, *, ddof: int = 1) -> FloatArray:
    x = as_float_array(values, min_size=2)
    if window <= ddof or window > x.size:
        raise ValueError("window must be greater than ddof and <= len(values)")
    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1")
    windows = np.lib.stride_tricks.sliding_window_view(x, window)
    return np.std(windows, axis=1, ddof=ddof)


def autocorrelation(values: ArrayLike, lag: int = 1) -> float:
    x = as_float_array(values, min_size=3)
    if lag <= 0 or lag >= x.size:
        raise ValueError("lag must be > 0 and < len(values)")
    return correlation(x[:-lag], x[lag:])
