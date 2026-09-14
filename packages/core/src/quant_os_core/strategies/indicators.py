"""Reusable technical indicators with explicit warm-up semantics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from quant_os_core.quant._validation import as_float_array, paired_float_arrays

FloatArray = npt.NDArray[np.float64]


def _positive_window(window: int, *, name: str = "window") -> int:
    parsed = int(window)
    if parsed != window or parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def sma(values: npt.ArrayLike, window: int) -> FloatArray:
    data = as_float_array(values, name="values")
    size = _positive_window(window)
    if size > data.size:
        raise ValueError("window cannot exceed number of observations")
    result = np.full(data.size, np.nan, dtype=np.float64)
    csum = np.cumsum(np.insert(data, 0, 0.0))
    result[size - 1 :] = (csum[size:] - csum[:-size]) / size
    return result


def ema(values: npt.ArrayLike, window: int) -> FloatArray:
    data = as_float_array(values, name="values")
    size = _positive_window(window)
    if size > data.size:
        raise ValueError("window cannot exceed number of observations")
    result = np.full(data.size, np.nan, dtype=np.float64)
    seed = float(np.mean(data[:size]))
    result[size - 1] = seed
    alpha = 2.0 / (size + 1.0)
    for index in range(size, data.size):
        result[index] = alpha * data[index] + (1.0 - alpha) * result[index - 1]
    return result


def true_range(
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    close: npt.ArrayLike,
) -> FloatArray:
    highs, lows = paired_float_arrays(high, low, left_name="high", right_name="low", min_size=1)
    closes = as_float_array(close, name="close")
    if closes.size != highs.size:
        raise ValueError("high, low and close must have equal length")
    if np.any(highs < lows):
        raise ValueError("high cannot be below low")
    result = np.empty(highs.size, dtype=np.float64)
    result[0] = highs[0] - lows[0]
    if highs.size > 1:
        result[1:] = np.maximum.reduce(
            [
                highs[1:] - lows[1:],
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ]
        )
    return result


def atr(
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    close: npt.ArrayLike,
    window: int = 14,
) -> FloatArray:
    ranges = true_range(high, low, close)
    size = _positive_window(window)
    if size > ranges.size:
        raise ValueError("window cannot exceed number of observations")
    result = np.full(ranges.size, np.nan, dtype=np.float64)
    result[size - 1] = float(np.mean(ranges[:size]))
    for index in range(size, ranges.size):
        result[index] = ((size - 1) * result[index - 1] + ranges[index]) / size
    return result


def rsi(values: npt.ArrayLike, window: int = 14) -> FloatArray:
    data = as_float_array(values, name="values", min_size=2)
    size = _positive_window(window)
    if size >= data.size:
        raise ValueError("window must be smaller than number of observations")
    delta = np.diff(data)
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    result = np.full(data.size, np.nan, dtype=np.float64)
    avg_gain = float(np.mean(gains[:size]))
    avg_loss = float(np.mean(losses[:size]))

    def score(gain: float, loss: float) -> float:
        if loss <= np.finfo(np.float64).eps:
            return 100.0 if gain > 0.0 else 50.0
        rs = gain / loss
        return 100.0 - 100.0 / (1.0 + rs)

    result[size] = score(avg_gain, avg_loss)
    for index in range(size + 1, data.size):
        avg_gain = ((size - 1) * avg_gain + gains[index - 1]) / size
        avg_loss = ((size - 1) * avg_loss + losses[index - 1]) / size
        result[index] = score(avg_gain, avg_loss)
    return result


def rolling_zscore(values: npt.ArrayLike, window: int) -> FloatArray:
    data = as_float_array(values, name="values")
    size = _positive_window(window)
    if size > data.size:
        raise ValueError("window cannot exceed number of observations")
    result = np.full(data.size, np.nan, dtype=np.float64)
    for index in range(size - 1, data.size):
        sample = data[index - size + 1 : index + 1]
        std = float(np.std(sample, ddof=1)) if size > 1 else 0.0
        result[index] = (
            0.0
            if std <= np.finfo(np.float64).eps
            else (data[index] - np.mean(sample)) / std
        )
    return result


def bollinger_bands(
    values: npt.ArrayLike,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    data = as_float_array(values, name="values")
    size = _positive_window(window)
    multiplier = float(num_std)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("num_std must be finite and > 0")
    middle = sma(data, size)
    std = np.full(data.size, np.nan, dtype=np.float64)
    for index in range(size - 1, data.size):
        std[index] = float(np.std(data[index - size + 1 : index + 1], ddof=1))
    return middle - multiplier * std, middle, middle + multiplier * std


def donchian_channels(
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    window: int,
    *,
    exclude_current: bool = True,
) -> tuple[FloatArray, FloatArray]:
    highs, lows = paired_float_arrays(high, low, left_name="high", right_name="low", min_size=2)
    size = _positive_window(window)
    if size >= highs.size and exclude_current:
        raise ValueError("window must be smaller than observations when excluding current bar")
    if size > highs.size:
        raise ValueError("window cannot exceed observations")
    upper = np.full(highs.size, np.nan, dtype=np.float64)
    lower = np.full(highs.size, np.nan, dtype=np.float64)
    start = size if exclude_current else size - 1
    for index in range(start, highs.size):
        end = index if exclude_current else index + 1
        begin = end - size
        upper[index] = float(np.max(highs[begin:end]))
        lower[index] = float(np.min(lows[begin:end]))
    return upper, lower


def rate_of_change(values: npt.ArrayLike, window: int) -> FloatArray:
    data = as_float_array(values, name="values")
    size = _positive_window(window)
    if size >= data.size:
        raise ValueError("window must be smaller than observations")
    if np.any(data <= 0.0):
        raise ValueError("values must be > 0 for rate of change")
    result = np.full(data.size, np.nan, dtype=np.float64)
    result[size:] = data[size:] / data[:-size] - 1.0
    return result


def cumulative_vwap(
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    close: npt.ArrayLike,
    volume: npt.ArrayLike,
) -> FloatArray:
    highs, lows = paired_float_arrays(high, low, left_name="high", right_name="low", min_size=1)
    closes = as_float_array(close, name="close")
    volumes = as_float_array(volume, name="volume")
    if closes.size != highs.size or volumes.size != highs.size:
        raise ValueError("high, low, close and volume must have equal length")
    if np.any(highs < lows):
        raise ValueError("high cannot be below low")
    if np.any(volumes < 0.0):
        raise ValueError("volume cannot be negative")
    typical = (highs + lows + closes) / 3.0
    cumulative_volume = np.cumsum(volumes)
    numerator = np.cumsum(typical * volumes)
    result = np.full(highs.size, np.nan, dtype=np.float64)
    valid = cumulative_volume > 0.0
    result[valid] = numerator[valid] / cumulative_volume[valid]
    return result
