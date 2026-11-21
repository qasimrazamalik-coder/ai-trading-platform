"""Leakage-free feature engineering from OHLC(V) bars.

Every feature at row ``t`` uses only bars ``0..t``. Rows before a feature's warm-up are NaN.
A test shifts future bars and checks that past features do not change.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

FEATURE_NAMES: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_20",
    "ret_60",
    "vol_20",
    "vol_ratio_20_60",
    "atr_pct_14",
    "rsi_14",
    "dist_sma_20",
    "dist_sma_50",
    "dist_sma_200",
    "donchian_pos_20",
    "vol_percentile_252",
    "volume_z_20",
)


def _rolling_mean(x: FloatArray, n: int) -> FloatArray:
    out = np.full(x.size, np.nan)
    if x.size >= n:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1 :] = (c[n:] - c[:-n]) / n
    return out


def _rolling_std(x: FloatArray, n: int) -> FloatArray:
    out = np.full(x.size, np.nan)
    for t in range(n - 1, x.size):
        window = x[t - n + 1 : t + 1]
        if np.isfinite(window).all():
            out[t] = float(np.std(window, ddof=1))
    return out


def _pct_change(close: FloatArray, n: int) -> FloatArray:
    out = np.full(close.size, np.nan)
    out[n:] = close[n:] / close[:-n] - 1
    return out


def _rsi(close: FloatArray, n: int = 14) -> FloatArray:
    out = np.full(close.size, np.nan)
    if close.size <= n:
        return out
    delta = np.diff(close)
    gain, loss = np.maximum(delta, 0), np.maximum(-delta, 0)
    avg_gain, avg_loss = float(gain[:n].mean()), float(loss[:n].mean())
    for t in range(n, close.size):
        if t > n:
            avg_gain = (avg_gain * (n - 1) + gain[t - 1]) / n
            avg_loss = (avg_loss * (n - 1) + loss[t - 1]) / n
        rs = avg_gain / avg_loss if avg_loss > 0 else np.inf
        out[t] = 100 - 100 / (1 + rs)
    return out


def build_features(
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    close: npt.ArrayLike,
    volume: npt.ArrayLike | None = None,
) -> FloatArray:
    """Feature matrix ``[n_bars, len(FEATURE_NAMES)]``; volume_z is 0 when volume is absent."""

    h, lo, c = (np.asarray(v, dtype=np.float64) for v in (high, low, close))
    if not (h.shape == lo.shape == c.shape) or c.ndim != 1 or (c <= 0).any():
        raise ValueError("high, low, close must be matching 1D arrays of positive prices")
    n = c.size
    logret = np.full(n, np.nan)
    logret[1:] = np.diff(np.log(c))
    vol20, vol60 = _rolling_std(logret, 20), _rolling_std(logret, 60)

    prev = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - lo, np.abs(h - prev), np.abs(lo - prev)]), axis=0)
    atr14 = _rolling_mean(tr, 14)

    hi20 = np.full(n, np.nan)
    lo20 = np.full(n, np.nan)
    for t in range(19, n):
        hi20[t], lo20[t] = h[t - 19 : t + 1].max(), lo[t - 19 : t + 1].min()
    span = hi20 - lo20
    donchian = np.where(span > 0, (c - lo20) / np.where(span > 0, span, 1), 0.5)

    vol_pct = np.full(n, np.nan)
    for t in range(n):
        start = max(0, t - 251)
        window = vol20[start : t + 1]
        window = window[np.isfinite(window)]
        if window.size >= 60 and np.isfinite(vol20[t]):
            vol_pct[t] = float(np.mean(window <= vol20[t]))

    if volume is None:
        volume_z = np.zeros(n)
    else:
        v = np.asarray(volume, dtype=np.float64)
        if v.shape != c.shape:
            raise ValueError("volume must match close")
        mean, std = _rolling_mean(v, 20), _rolling_std(v, 20)
        volume_z = np.where(std > 0, (v - mean) / np.where(std > 0, std, 1), 0.0)

    columns = [
        _pct_change(c, 1),
        _pct_change(c, 5),
        _pct_change(c, 20),
        _pct_change(c, 60),
        vol20,
        vol20 / vol60,
        atr14 / c,
        _rsi(c) / 100 - 0.5,
        c / _rolling_mean(c, 20) - 1,
        c / _rolling_mean(c, 50) - 1,
        c / _rolling_mean(c, 200) - 1,
        donchian,
        vol_pct,
        volume_z,
    ]
    return np.column_stack(columns)


def ready_rows(features: FloatArray) -> npt.NDArray[np.bool_]:
    """Rows where every feature has finished warming up."""

    return np.isfinite(features).all(axis=1)
