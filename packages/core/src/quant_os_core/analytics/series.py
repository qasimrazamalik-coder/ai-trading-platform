"""Series transforms for drawdown, volatility, liquidity, and funding views."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from quant_os_core.analytics._validation import finite_1d


def drawdown_series(returns: ArrayLike) -> np.ndarray:
    values = finite_1d(returns, name="returns")
    if np.any(values <= -1.0):
        raise ValueError("returns must be greater than -1")
    wealth = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(wealth)
    return wealth / peak - 1.0


def rolling_volatility_series(
    returns: ArrayLike,
    *,
    window: int,
    periods_per_year: float = 252.0,
) -> np.ndarray:
    values = finite_1d(returns, name="returns")
    if window < 2 or window > values.size:
        raise ValueError("window must be in [2, len(returns)]")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive")
    output = np.full(values.size, np.nan, dtype=np.float64)
    scale = float(np.sqrt(periods_per_year))
    for index in range(window - 1, values.size):
        output[index] = float(np.std(values[index - window + 1 : index + 1], ddof=1) * scale)
    return output


def liquidity_score(
    spreads_bps: ArrayLike,
    depth_notional: ArrayLike,
) -> np.ndarray:
    spread = finite_1d(spreads_bps, name="spreads_bps")
    depth = finite_1d(depth_notional, name="depth_notional")
    if spread.shape != depth.shape:
        raise ValueError("spreads_bps and depth_notional must have equal length")
    if np.any(spread < 0.0) or np.any(depth <= 0.0):
        raise ValueError("spread must be non-negative and depth strictly positive")
    relative_depth = depth / np.median(depth)
    return np.asarray(relative_depth / (1.0 + spread), dtype=np.float64)


def annualized_funding_rate(periodic_rates: ArrayLike, *, periods_per_year: float) -> np.ndarray:
    rates = finite_1d(periodic_rates, name="periodic_rates")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive")
    return rates * float(periods_per_year)
