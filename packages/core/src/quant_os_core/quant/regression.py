"""Linear regression and statistical-arbitrage foundations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from statsmodels.tsa.stattools import coint

from quant_os_core.quant._validation import (
    FloatArray,
    as_float_array,
    paired_float_arrays,
    require_positive,
)


@dataclass(frozen=True, slots=True)
class OLSResult:
    intercept: float
    slope: float
    residuals: FloatArray
    r_squared: float


def ordinary_least_squares(y: ArrayLike, x: ArrayLike) -> OLSResult:
    dependent, independent = paired_float_arrays(y, x, left_name="y", right_name="x")
    if float(np.std(independent, ddof=1)) == 0.0:
        raise ValueError("x must not be constant")
    design = np.column_stack((np.ones(independent.size), independent))
    coefficients, *_ = np.linalg.lstsq(design, dependent, rcond=None)
    fitted = design @ coefficients
    residuals = dependent - fitted
    ss_res = float(residuals @ residuals)
    centered = dependent - np.mean(dependent)
    ss_tot = float(centered @ centered)
    if np.isclose(ss_tot, 0.0, atol=1e-15):
        r_squared = 1.0 if np.isclose(ss_res, 0.0, atol=1e-15) else 0.0
    else:
        r_squared = 1.0 - ss_res / ss_tot
    return OLSResult(
        intercept=float(coefficients[0]),
        slope=float(coefficients[1]),
        residuals=residuals,
        r_squared=float(r_squared),
    )


def beta(asset_returns: ArrayLike, benchmark_returns: ArrayLike) -> float:
    asset, benchmark = paired_float_arrays(
        asset_returns,
        benchmark_returns,
        left_name="asset_returns",
        right_name="benchmark_returns",
    )
    variance = float(np.var(benchmark, ddof=1))
    if variance == 0.0:
        raise ValueError("benchmark variance must be > 0")
    return float(np.cov(asset, benchmark, ddof=1)[0, 1] / variance)


def alpha(
    asset_returns: ArrayLike,
    benchmark_returns: ArrayLike,
    *,
    risk_free_period_return: float = 0.0,
) -> float:
    asset, benchmark = paired_float_arrays(
        asset_returns,
        benchmark_returns,
        left_name="asset_returns",
        right_name="benchmark_returns",
    )
    b = beta(asset, benchmark)
    if not np.isfinite(risk_free_period_return):
        raise ValueError("risk_free_period_return must be finite")
    expected = risk_free_period_return + b * (np.mean(benchmark) - risk_free_period_return)
    return float(np.mean(asset) - expected)


def tracking_error(
    asset_returns: ArrayLike,
    benchmark_returns: ArrayLike,
    *,
    periods_per_year: float = 252.0,
) -> float:
    asset, benchmark = paired_float_arrays(asset_returns, benchmark_returns)
    periods = require_positive(periods_per_year, name="periods_per_year")
    active = asset - benchmark
    return float(np.std(active, ddof=1) * np.sqrt(periods))


def information_ratio(
    asset_returns: ArrayLike,
    benchmark_returns: ArrayLike,
    *,
    periods_per_year: float = 252.0,
) -> float:
    asset, benchmark = paired_float_arrays(asset_returns, benchmark_returns)
    periods = require_positive(periods_per_year, name="periods_per_year")
    active = asset - benchmark
    std = float(np.std(active, ddof=1))
    if std == 0.0:
        raise ValueError("information ratio is undefined for zero tracking error")
    return float(np.mean(active) / std * np.sqrt(periods))


@dataclass(frozen=True, slots=True)
class CointegrationResult:
    hedge_ratio: float
    intercept: float
    adf_statistic: float
    p_value: float
    critical_values: dict[str, float]
    residuals: FloatArray

    @property
    def is_stationary_5pct(self) -> bool:
        return self.p_value < 0.05


def engle_granger(
    y: ArrayLike, x: ArrayLike, *, maxlag: int | None = None
) -> CointegrationResult:
    """Two-step Engle-Granger foundation: OLS spread then ADF residual test."""
    dependent = as_float_array(y, name="y", min_size=20)
    independent = as_float_array(x, name="x", min_size=20)
    if dependent.size != independent.size:
        raise ValueError("x and y must have equal length")
    regression = ordinary_least_squares(dependent, independent)
    statistic, p_value, critical_array = coint(
        dependent,
        independent,
        trend="c",
        maxlag=maxlag,
        autolag="aic",
    )
    critical = {
        "1%": float(critical_array[0]),
        "5%": float(critical_array[1]),
        "10%": float(critical_array[2]),
    }
    return CointegrationResult(
        hedge_ratio=regression.slope,
        intercept=regression.intercept,
        adf_statistic=float(statistic),
        p_value=float(p_value),
        critical_values=critical,
        residuals=regression.residuals,
    )


def mean_reversion_half_life(spread: ArrayLike) -> float:
    """Estimate Ornstein-Uhlenbeck-like half-life from an AR(1) approximation."""
    s = as_float_array(spread, name="spread", min_size=10)
    lagged = s[:-1]
    delta = np.diff(s)
    regression = ordinary_least_squares(delta, lagged)
    coefficient = regression.slope
    if coefficient >= -1e-12:
        raise ValueError("series does not exhibit mean-reverting AR(1) dynamics")
    half_life = -np.log(2.0) / coefficient
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("invalid half-life estimate")
    return float(half_life)
