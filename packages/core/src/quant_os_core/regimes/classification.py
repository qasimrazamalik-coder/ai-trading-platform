"""Deterministic regime classifiers used as transparent baselines."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from quant_os_core.regimes.base import (
    CorrelationRegime,
    LiquidityRegime,
    MacroRegime,
    RegimeEstimate,
    RiskRegime,
    TrendRegime,
    VolatilityRegime,
)


def _finite_1d(values: npt.ArrayLike, *, name: str, minimum: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < minimum:
        raise ValueError(f"{name} must be one-dimensional with at least {minimum} values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _confidence_from_margin(margin: float, scale: float) -> float:
    if scale <= 0.0 or not np.isfinite(scale):
        return 1.0 if margin > 0.0 else 0.5
    return float(np.clip(0.5 + 0.5 * np.tanh(abs(margin) / scale), 0.5, 1.0))


def classify_trend(
    prices: npt.ArrayLike,
    *,
    fast_window: int = 20,
    slow_window: int = 80,
    neutral_band: float = 0.005,
) -> RegimeEstimate:
    values = _finite_1d(prices, name="prices", minimum=max(fast_window, slow_window) + 1)
    if np.any(values <= 0.0):
        raise ValueError("prices must be positive")
    if fast_window < 2 or slow_window <= fast_window:
        raise ValueError("windows require 2 <= fast_window < slow_window")
    if neutral_band < 0.0:
        raise ValueError("neutral_band must be non-negative")

    fast = float(np.mean(values[-fast_window:]))
    slow = float(np.mean(values[-slow_window:]))
    slow_prev = float(np.mean(values[-slow_window - 1 : -1]))
    ma_spread = fast / slow - 1.0
    slow_slope = slow / slow_prev - 1.0 if slow_prev > 0.0 else 0.0
    score = 0.75 * ma_spread + 0.25 * slow_slope

    if score > neutral_band:
        label = TrendRegime.BULL.value
    elif score < -neutral_band:
        label = TrendRegime.BEAR.value
    else:
        label = TrendRegime.SIDEWAYS.value
    confidence = _confidence_from_margin(abs(score) - neutral_band, max(neutral_band, 1e-4))
    return RegimeEstimate(label=label, confidence=confidence, score=float(score))


def classify_volatility(
    returns: npt.ArrayLike,
    *,
    lookback: int = 20,
    low_quantile: float = 0.30,
    high_quantile: float = 0.70,
    history_window: int = 252,
) -> RegimeEstimate:
    values = _finite_1d(returns, name="returns", minimum=max(lookback + 2, 4))
    if lookback < 2 or history_window < lookback:
        raise ValueError("history_window must be >= lookback >= 2")
    if not 0.0 < low_quantile < high_quantile < 1.0:
        raise ValueError("volatility quantiles must satisfy 0 < low < high < 1")

    rolling: list[float] = []
    start = max(lookback, values.size - history_window)
    for end in range(start, values.size + 1):
        window = values[end - lookback : end]
        rolling.append(float(np.std(window, ddof=1)))
    vols = np.asarray(rolling, dtype=np.float64)
    current = float(vols[-1])
    low = float(np.quantile(vols, low_quantile))
    high = float(np.quantile(vols, high_quantile))

    if current <= low:
        label = VolatilityRegime.LOW.value
        margin = low - current
    elif current >= high:
        label = VolatilityRegime.HIGH.value
        margin = current - high
    else:
        label = VolatilityRegime.NORMAL.value
        margin = min(current - low, high - current)
    scale = max(high - low, 1e-12)
    return RegimeEstimate(
        label=label,
        confidence=_confidence_from_margin(margin, scale),
        score=current,
    )


def classify_risk(
    growth_asset_returns: npt.ArrayLike,
    defensive_asset_returns: npt.ArrayLike,
    *,
    neutral_band: float = 0.001,
) -> RegimeEstimate:
    growth = _finite_1d(growth_asset_returns, name="growth_asset_returns")
    defensive = _finite_1d(defensive_asset_returns, name="defensive_asset_returns")
    if growth.size != defensive.size:
        raise ValueError("growth and defensive returns must have equal length")
    if neutral_band < 0.0:
        raise ValueError("neutral_band must be non-negative")
    score = float(np.mean(growth) - np.mean(defensive))
    if score > neutral_band:
        label = RiskRegime.RISK_ON.value
    elif score < -neutral_band:
        label = RiskRegime.RISK_OFF.value
    else:
        label = RiskRegime.NEUTRAL.value
    return RegimeEstimate(
        label=label,
        confidence=_confidence_from_margin(abs(score) - neutral_band, max(neutral_band, 1e-6)),
        score=score,
    )


def classify_liquidity(
    *,
    spread_bps: float,
    depth_zscore: float,
    volume_zscore: float,
    tight_threshold: float = -0.75,
    abundant_threshold: float = 0.75,
) -> RegimeEstimate:
    values = (spread_bps, depth_zscore, volume_zscore)
    if any(not np.isfinite(value) for value in values):
        raise ValueError("liquidity inputs must be finite")
    if spread_bps < 0.0:
        raise ValueError("spread_bps must be non-negative")
    if tight_threshold >= abundant_threshold:
        raise ValueError("tight_threshold must be below abundant_threshold")

    spread_penalty = np.log1p(spread_bps) / np.log(11.0)
    score = float(0.45 * depth_zscore + 0.35 * volume_zscore - 0.20 * spread_penalty)
    if score <= tight_threshold:
        label = LiquidityRegime.TIGHT.value
        margin = tight_threshold - score
    elif score >= abundant_threshold:
        label = LiquidityRegime.ABUNDANT.value
        margin = score - abundant_threshold
    else:
        label = LiquidityRegime.NORMAL.value
        margin = min(score - tight_threshold, abundant_threshold - score)
    return RegimeEstimate(
        label=label,
        confidence=_confidence_from_margin(margin, abundant_threshold - tight_threshold),
        score=score,
    )


def average_absolute_correlation(correlation_matrix: npt.ArrayLike) -> float:
    matrix = np.asarray(correlation_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("correlation_matrix must be square with at least two assets")
    if not np.isfinite(matrix).all():
        raise ValueError("correlation_matrix must contain finite values")
    if np.any(np.abs(matrix) > 1.0 + 1e-9):
        raise ValueError("correlations must be in [-1, 1]")
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return float(np.mean(np.abs(upper)))


def classify_correlation(
    correlation_matrix: npt.ArrayLike,
    *,
    low_threshold: float = 0.25,
    high_threshold: float = 0.65,
) -> RegimeEstimate:
    if not 0.0 <= low_threshold < high_threshold <= 1.0:
        raise ValueError("correlation thresholds must satisfy 0 <= low < high <= 1")
    score = average_absolute_correlation(correlation_matrix)
    if score <= low_threshold:
        label = CorrelationRegime.LOW.value
        margin = low_threshold - score
    elif score >= high_threshold:
        label = CorrelationRegime.HIGH.value
        margin = score - high_threshold
    else:
        label = CorrelationRegime.NORMAL.value
        margin = min(score - low_threshold, high_threshold - score)
    return RegimeEstimate(
        label=label,
        confidence=_confidence_from_margin(margin, high_threshold - low_threshold),
        score=score,
    )


def classify_macro(
    *,
    growth_zscore: float,
    inflation_zscore: float,
    neutral_band: float = 0.20,
) -> RegimeEstimate:
    if not np.isfinite(growth_zscore) or not np.isfinite(inflation_zscore):
        raise ValueError("macro scores must be finite")
    if neutral_band < 0.0:
        raise ValueError("neutral_band must be non-negative")

    if abs(growth_zscore) <= neutral_band or abs(inflation_zscore) <= neutral_band:
        label = MacroRegime.MIXED.value
        margin = min(abs(growth_zscore), abs(inflation_zscore))
    elif growth_zscore > 0.0 and inflation_zscore < 0.0:
        label = MacroRegime.GOLDILOCKS.value
        margin = min(growth_zscore, -inflation_zscore)
    elif growth_zscore > 0.0 and inflation_zscore > 0.0:
        label = MacroRegime.REFLATION.value
        margin = min(growth_zscore, inflation_zscore)
    elif growth_zscore < 0.0 and inflation_zscore > 0.0:
        label = MacroRegime.STAGFLATION.value
        margin = min(-growth_zscore, inflation_zscore)
    else:
        label = MacroRegime.DEFLATION.value
        margin = min(-growth_zscore, -inflation_zscore)
    return RegimeEstimate(
        label=label,
        confidence=_confidence_from_margin(margin, max(neutral_band, 0.25)),
        score=float(growth_zscore - inflation_zscore),
    )


def volatility_clustering_score(returns: npt.ArrayLike, *, lag: int = 1) -> float:
    """Correlation of squared returns with their lagged values."""
    values = _finite_1d(returns, name="returns", minimum=lag + 3)
    if lag < 1 or lag >= values.size - 1:
        raise ValueError("lag must be positive and smaller than the sample")
    squared = values * values
    left = squared[:-lag]
    right = squared[lag:]
    if np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])
