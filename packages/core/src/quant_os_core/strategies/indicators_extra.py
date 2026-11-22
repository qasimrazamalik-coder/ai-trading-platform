"""Extra technical indicators: numpy only, causal, NaN during warm-up.

Every function here uses only bars up to and including ``t`` to produce the value at ``t``
(Ichimoku's leading spans are shifted forward, so their value at ``t`` is built from bars up
to ``t - displacement``; the lagging span is omitted because plotting it needs the future).
Inputs must be finite; outputs are ``NaN`` until enough bars exist. Formulas follow the
textbook definitions (Wilder for ATR/ADX/Parabolic SAR, Lambert for CCI, Kaufman for KAMA,
Hull for HMA, Hosoda for Ichimoku). These are research indicators, not trade signals: any
strategy built on them still goes through the Shariah gate, the risk engine and the
edge-over-cost filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quant_os_core.quant._validation import as_float_array
from quant_os_core.strategies.indicators import atr, cumulative_vwap, ema

FloatArray = npt.NDArray[np.float64]

__all__ = [
    "Ichimoku",
    "adx",
    "atr",
    "cci",
    "donchian",
    "hull_ma",
    "ichimoku",
    "kama",
    "keltner",
    "obv",
    "parabolic_sar",
    "stochastic",
    "supertrend",
    "vwap",
    "williams_r",
    "wma",
    "zscore",
]


def _window(window: int, *, name: str = "window") -> int:
    parsed = int(window)
    if parsed != window or parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _hlc(
    high: npt.ArrayLike, low: npt.ArrayLike, close: npt.ArrayLike | None = None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    highs = as_float_array(high, name="high")
    lows = as_float_array(low, name="low")
    closes = as_float_array(close if close is not None else (highs + lows) / 2.0, name="close")
    if not highs.size == lows.size == closes.size:
        raise ValueError("high, low and close must have equal length")
    if np.any(highs < lows):
        raise ValueError("high cannot be below low")
    return highs, lows, closes


def _rolling_max(values: FloatArray, window: int) -> FloatArray:
    out = np.full(values.size, np.nan)
    if window <= values.size:
        view = np.lib.stride_tricks.sliding_window_view(values, window)
        out[window - 1 :] = view.max(axis=1)
    return out


def _rolling_min(values: FloatArray, window: int) -> FloatArray:
    out = np.full(values.size, np.nan)
    if window <= values.size:
        view = np.lib.stride_tricks.sliding_window_view(values, window)
        out[window - 1 :] = view.min(axis=1)
    return out


def _wilder_sum(values: FloatArray, window: int, start: int) -> FloatArray:
    """Wilder running sum seeded with the plain sum of ``values[start:start+window]``.

    Callers guarantee ``start + window <= len(values)``.
    """

    out = np.full(values.size, np.nan)
    first = start + window - 1
    out[first] = float(np.sum(values[start : first + 1]))
    for i in range(first + 1, values.size):
        out[i] = out[i - 1] - out[i - 1] / window + values[i]
    return out


def wma(values: npt.ArrayLike, window: int) -> FloatArray:
    """Linearly weighted moving average (newest bar weight ``window``)."""

    data = as_float_array(values, name="values")
    size = _window(window)
    out = np.full(data.size, np.nan)
    if size > data.size:
        return out
    weights = np.arange(1, size + 1, dtype=np.float64)
    view = np.lib.stride_tricks.sliding_window_view(data, size)
    out[size - 1 :] = view @ weights / weights.sum()
    return out


def hull_ma(values: npt.ArrayLike, window: int) -> FloatArray:
    """Hull moving average: ``WMA(2*WMA(n/2) - WMA(n), round(sqrt(n)))``."""

    data = as_float_array(values, name="values")
    size = _window(window)
    if size < 2:
        raise ValueError("window must be at least 2")
    half = max(1, size // 2)
    root = max(1, round(np.sqrt(size)))
    raw = 2.0 * wma(data, half) - wma(data, size)
    out = np.full(data.size, np.nan)
    valid = np.flatnonzero(np.isfinite(raw))
    if valid.size >= root:
        out[valid[0] :] = wma(raw[valid[0] :], root)
    return out


def kama(values: npt.ArrayLike, window: int = 10, fast: int = 2, slow: int = 30) -> FloatArray:
    """Kaufman adaptive moving average, seeded with the close at bar ``window``."""

    data = as_float_array(values, name="values")
    size = _window(window)
    fast_sc = 2.0 / (_window(fast, name="fast") + 1.0)
    slow_sc = 2.0 / (_window(slow, name="slow") + 1.0)
    out = np.full(data.size, np.nan)
    if size >= data.size:
        return out
    out[size] = data[size]
    for i in range(size + 1, data.size):
        change = abs(data[i] - data[i - size])
        volatility = float(np.sum(np.abs(np.diff(data[i - size : i + 1]))))
        er = 0.0 if volatility <= 0.0 else change / volatility
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        out[i] = out[i - 1] + sc * (data[i] - out[i - 1])
    return out


def zscore(values: npt.ArrayLike, window: int, *, ddof: int = 0) -> FloatArray:
    """Rolling z-score of the last value against its own window; 0 when the window is flat."""

    data = as_float_array(values, name="values")
    size = _window(window)
    if ddof not in (0, 1) or size - ddof <= 0:
        raise ValueError("ddof must be 0 or 1 and smaller than window")
    out = np.full(data.size, np.nan)
    if size > data.size:
        return out
    view = np.lib.stride_tricks.sliding_window_view(data, size)
    mean = view.mean(axis=1)
    std = view.std(axis=1, ddof=ddof)
    last = data[size - 1 :]
    flat = std <= np.finfo(np.float64).eps * np.maximum(1.0, np.abs(mean))
    out[size - 1 :] = np.where(flat, 0.0, (last - mean) / np.where(flat, 1.0, std))
    return out


def stochastic(
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    close: npt.ArrayLike,
    k_window: int = 14,
    d_window: int = 3,
) -> tuple[FloatArray, FloatArray]:
    """Fast stochastic %K (0-100) and %D = SMA(%K, d_window); 50 when the range is flat."""

    highs, lows, closes = _hlc(high, low, close)
    k_size, d_size = _window(k_window, name="k_window"), _window(d_window, name="d_window")
    hh, ll = _rolling_max(highs, k_size), _rolling_min(lows, k_size)
    rng = hh - ll
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(rng > 0, 100.0 * (closes - ll) / np.where(rng > 0, rng, 1.0), 50.0)
    k[np.isnan(hh)] = np.nan
    d = np.full(k.size, np.nan)
    valid = np.flatnonzero(np.isfinite(k))
    if valid.size >= d_size:
        tail = k[valid[0] :]
        view = np.lib.stride_tricks.sliding_window_view(tail, d_size)
        d[valid[0] + d_size - 1 :] = view.mean(axis=1)
    return k, d


def williams_r(
    high: npt.ArrayLike, low: npt.ArrayLike, close: npt.ArrayLike, window: int = 14
) -> FloatArray:
    """Williams %R in [-100, 0]; -50 when the range is flat."""

    k, _ = stochastic(high, low, close, window, 1)
    return k - 100.0


def cci(
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    close: npt.ArrayLike,
    window: int = 20,
    constant: float = 0.015,
) -> FloatArray:
    """Commodity Channel Index on the typical price, mean absolute deviation scaling."""

    highs, lows, closes = _hlc(high, low, close)
    size = _window(window)
    if not np.isfinite(constant) or constant <= 0:
        raise ValueError("constant must be finite and > 0")
    tp = (highs + lows + closes) / 3.0
    out = np.full(tp.size, np.nan)
    if size > tp.size:
        return out
    view = np.lib.stride_tricks.sliding_window_view(tp, size)
    mean = view.mean(axis=1)
    mad = np.abs(view - mean[:, None]).mean(axis=1)
    last = tp[size - 1 :]
    out[size - 1 :] = np.where(
        mad > 0, (last - mean) / (constant * np.where(mad > 0, mad, 1.0)), 0.0
    )
    return out


def obv(close: npt.ArrayLike, volume: npt.ArrayLike | None) -> FloatArray | None:
    """On-balance volume starting at 0; ``None`` when no volume is available (skip it)."""

    if volume is None:
        return None
    closes = as_float_array(close, name="close")
    volumes = as_float_array(volume, name="volume")
    if volumes.size != closes.size:
        raise ValueError("close and volume must have equal length")
    if np.any(volumes < 0):
        raise ValueError("volume cannot be negative")
    direction = np.sign(np.diff(closes))
    return np.concatenate([[0.0], np.cumsum(direction * volumes[1:])])


def vwap(
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    close: npt.ArrayLike,
    volume: npt.ArrayLike,
    window: int | None = None,
) -> FloatArray:
    """Typical-price VWAP: cumulative when ``window`` is None, otherwise rolling."""

    if window is None:
        return cumulative_vwap(high, low, close, volume)
    highs, lows, closes = _hlc(high, low, close)
    volumes = as_float_array(volume, name="volume")
    if volumes.size != highs.size:
        raise ValueError("volume must match price length")
    if np.any(volumes < 0):
        raise ValueError("volume cannot be negative")
    size = _window(window)
    out = np.full(highs.size, np.nan)
    if size > highs.size:
        return out
    tp = (highs + lows + closes) / 3.0
    pv = np.lib.stride_tricks.sliding_window_view(tp * volumes, size).sum(axis=1)
    vol = np.lib.stride_tricks.sliding_window_view(volumes, size).sum(axis=1)
    out[size - 1 :] = np.where(vol > 0, pv / np.where(vol > 0, vol, 1.0), np.nan)
    return out


def adx(
    high: npt.ArrayLike, low: npt.ArrayLike, close: npt.ArrayLike, window: int = 14
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Wilder's ADX with +DI and -DI. DI values start at bar ``window``, ADX at ``2*window-1``."""

    highs, lows, closes = _hlc(high, low, close)
    size = _window(window)
    n = highs.size
    nan = np.full(n, np.nan)
    if n <= 2 * size - 1:
        return nan.copy(), nan.copy(), nan.copy()
    up = np.diff(highs)
    down = -np.diff(lows)
    plus_dm = np.concatenate([[0.0], np.where((up > down) & (up > 0), up, 0.0)])
    minus_dm = np.concatenate([[0.0], np.where((down > up) & (down > 0), down, 0.0)])
    tr = np.concatenate(
        [
            [highs[0] - lows[0]],
            np.maximum.reduce(
                [
                    highs[1:] - lows[1:],
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1]),
                ]
            ),
        ]
    )
    str_ = _wilder_sum(tr, size, 1)
    spdm = _wilder_sum(plus_dm, size, 1)
    smdm = _wilder_sum(minus_dm, size, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        plus_di = np.where(str_ > 0, 100.0 * spdm / str_, 0.0)
        minus_di = np.where(str_ > 0, 100.0 * smdm / str_, 0.0)
        total = plus_di + minus_di
        dx = np.where(total > 0, 100.0 * np.abs(plus_di - minus_di) / total, 0.0)
    plus_di[np.isnan(str_)] = np.nan
    minus_di[np.isnan(str_)] = np.nan
    dx[np.isnan(str_)] = np.nan
    adx_out = nan.copy()
    first = 2 * size - 1
    adx_out[first] = float(np.mean(dx[size : first + 1]))
    for i in range(first + 1, n):
        adx_out[i] = (adx_out[i - 1] * (size - 1) + dx[i]) / size
    return adx_out, plus_di, minus_di


def keltner(
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    close: npt.ArrayLike,
    ema_window: int = 20,
    atr_window: int = 10,
    multiplier: float = 2.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Keltner channel ``(lower, middle, upper)``: EMA(close) +/- multiplier x ATR."""

    highs, lows, closes = _hlc(high, low, close)
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("multiplier must be finite and > 0")
    nan = np.full(highs.size, np.nan)
    if max(_window(ema_window), _window(atr_window)) > highs.size:
        return nan.copy(), nan.copy(), nan.copy()
    mid = ema(closes, ema_window)
    band = multiplier * atr(highs, lows, closes, atr_window)
    return mid - band, mid, mid + band


def donchian(
    high: npt.ArrayLike, low: npt.ArrayLike, window: int = 20
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Donchian ``(upper, middle, lower)`` over the last ``window`` bars including today.

    For breakout signals compare today's close with yesterday's channel (shift by one).
    """

    highs, lows, _ = _hlc(high, low)
    size = _window(window)
    upper, lower = _rolling_max(highs, size), _rolling_min(lows, size)
    return upper, (upper + lower) / 2.0, lower


@dataclass(frozen=True, slots=True)
class Ichimoku:
    conversion: FloatArray
    base: FloatArray
    span_a: FloatArray
    span_b: FloatArray


def ichimoku(
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    conversion: int = 9,
    base: int = 26,
    span_b: int = 52,
    displacement: int = 26,
) -> Ichimoku:
    """Tenkan, kijun and the two leading spans *as plotted at t* (built from ``t - shift``)."""

    highs, lows, _ = _hlc(high, low)
    shift = int(displacement)
    if shift != displacement or shift < 0:
        raise ValueError("displacement must be a non-negative integer")

    def mid(window: int, name: str) -> FloatArray:
        size = _window(window, name=name)
        return (_rolling_max(highs, size) + _rolling_min(lows, size)) / 2.0

    tenkan, kijun = mid(conversion, "conversion"), mid(base, "base")
    raw_a, raw_b = (tenkan + kijun) / 2.0, mid(span_b, "span_b")

    def forward(series: FloatArray) -> FloatArray:
        out = np.full(series.size, np.nan)
        if shift < series.size:
            out[shift:] = series[: series.size - shift]
        return out

    return Ichimoku(tenkan, kijun, forward(raw_a), forward(raw_b))


def parabolic_sar(
    high: npt.ArrayLike, low: npt.ArrayLike, step: float = 0.02, max_step: float = 0.2
) -> tuple[FloatArray, FloatArray]:
    """Wilder's Parabolic SAR and trend direction (+1 up, -1 down). Starts long at bar 1."""

    highs, lows, _ = _hlc(high, low)
    if not 0 < step <= max_step:
        raise ValueError("need 0 < step <= max_step")
    n = highs.size
    sar = np.full(n, np.nan)
    trend = np.full(n, np.nan)
    if n < 2:
        return sar, trend
    up = True
    af = step
    ep = highs[0]
    current = lows[0]
    for i in range(1, n):
        if i > 1:
            current = current + af * (ep - current)
            if up:
                current = min(current, lows[i - 1], lows[i - 2])
            else:
                current = max(current, highs[i - 1], highs[i - 2])
        if up:
            if lows[i] < current:
                up, current, ep, af = False, ep, lows[i], step
            elif highs[i] > ep:
                ep, af = highs[i], min(af + step, max_step)
        else:
            if highs[i] > current:
                up, current, ep, af = True, ep, highs[i], step
            elif lows[i] < ep:
                ep, af = lows[i], min(af + step, max_step)
        sar[i] = current
        trend[i] = 1.0 if up else -1.0
    return sar, trend


def supertrend(
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    close: npt.ArrayLike,
    window: int = 10,
    multiplier: float = 3.0,
) -> tuple[FloatArray, FloatArray]:
    """Supertrend line and direction (+1 up, -1 down), NaN until ATR is ready."""

    highs, lows, closes = _hlc(high, low, close)
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("multiplier must be finite and > 0")
    size = _window(window)
    n = highs.size
    line = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    if size > n:
        return line, direction
    band = multiplier * atr(highs, lows, closes, size)
    mid = (highs + lows) / 2.0
    basic_up, basic_lo = mid + band, mid - band
    start = size - 1
    final_up, final_lo = basic_up[start], basic_lo[start]
    up = closes[start] >= mid[start]
    line[start] = final_lo if up else final_up
    direction[start] = 1.0 if up else -1.0
    for i in range(start + 1, n):
        final_up = basic_up[i] if basic_up[i] < final_up or closes[i - 1] > final_up else final_up
        final_lo = basic_lo[i] if basic_lo[i] > final_lo or closes[i - 1] < final_lo else final_lo
        if up and closes[i] < final_lo:
            up = False
        elif not up and closes[i] > final_up:
            up = True
        line[i] = final_lo if up else final_up
        direction[i] = 1.0 if up else -1.0
    return line, direction
