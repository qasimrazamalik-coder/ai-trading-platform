"""Historical, rolling and exponentially weighted volatility estimators."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from quant_os_core.quant._validation import FloatArray, as_float_array, require_positive
from quant_os_core.quant.statistics import rolling_std


def annualized_volatility(returns: ArrayLike, *, periods_per_year: float = 252.0) -> float:
    r = as_float_array(returns, name="returns", min_size=2)
    periods = require_positive(periods_per_year, name="periods_per_year")
    return float(np.std(r, ddof=1) * np.sqrt(periods))


def rolling_volatility(
    returns: ArrayLike,
    window: int,
    *,
    periods_per_year: float = 252.0,
) -> FloatArray:
    periods = require_positive(periods_per_year, name="periods_per_year")
    return rolling_std(returns, window, ddof=1) * np.sqrt(periods)


def ewma_variance(
    returns: ArrayLike,
    *,
    decay: float = 0.94,
    initial_variance: float | None = None,
) -> FloatArray:
    """RiskMetrics-style EWMA variance path.

    ``variance[t] = decay * variance[t-1] + (1-decay) * r[t-1]^2``.
    The returned array has the same length as the input return series.
    """
    r = as_float_array(returns, name="returns", min_size=2)
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be strictly between 0 and 1")
    if initial_variance is None:
        seed = float(np.var(r, ddof=1))
    else:
        seed = float(initial_variance)
        if not np.isfinite(seed) or seed < 0:
            raise ValueError("initial_variance must be finite and >= 0")
    out = np.empty_like(r)
    out[0] = seed
    for index in range(1, r.size):
        out[index] = decay * out[index - 1] + (1.0 - decay) * r[index - 1] ** 2
    return out


def ewma_volatility(
    returns: ArrayLike,
    *,
    decay: float = 0.94,
    periods_per_year: float = 252.0,
) -> FloatArray:
    periods = require_positive(periods_per_year, name="periods_per_year")
    return np.sqrt(ewma_variance(returns, decay=decay) * periods)
