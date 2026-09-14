"""Return transformations and wealth/drawdown arithmetic."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from quant_os_core.quant._validation import FloatArray, as_float_array


def simple_returns(prices: ArrayLike) -> FloatArray:
    """Compute simple returns ``P[t] / P[t-1] - 1``.

    Prices must be finite and strictly positive.
    """
    p = as_float_array(prices, name="prices", min_size=2)
    if np.any(p <= 0):
        raise ValueError("prices must be strictly positive")
    return p[1:] / p[:-1] - 1.0


def log_returns(prices: ArrayLike) -> FloatArray:
    """Compute continuously compounded returns ``log(P[t] / P[t-1])``."""
    p = as_float_array(prices, name="prices", min_size=2)
    if np.any(p <= 0):
        raise ValueError("prices must be strictly positive")
    return np.diff(np.log(p))


def cumulative_return(returns: ArrayLike) -> float:
    """Compound periodic simple returns into one total return."""
    r = as_float_array(returns, name="returns", min_size=1)
    if np.any(r <= -1.0):
        if np.any(r < -1.0):
            raise ValueError("simple returns cannot be less than -100%")
        return -1.0
    return float(np.prod(1.0 + r) - 1.0)


def wealth_index(returns: ArrayLike, *, initial: float = 1.0) -> FloatArray:
    """Build a compounded wealth path including the initial value."""
    r = as_float_array(returns, name="returns", min_size=1)
    if np.any(r < -1.0):
        raise ValueError("simple returns cannot be less than -100%")
    if not np.isfinite(initial) or initial <= 0:
        raise ValueError("initial must be finite and > 0")
    return np.concatenate(([float(initial)], float(initial) * np.cumprod(1.0 + r)))


def drawdown_series(returns: ArrayLike) -> FloatArray:
    """Return percentage drawdown from the running wealth peak.

    Values are zero at peaks and negative during drawdowns.
    """
    wealth = wealth_index(returns)
    peaks = np.maximum.accumulate(wealth)
    return wealth / peaks - 1.0


def max_drawdown(returns: ArrayLike) -> float:
    """Return maximum drawdown as a non-positive number."""
    return float(np.min(drawdown_series(returns)))
