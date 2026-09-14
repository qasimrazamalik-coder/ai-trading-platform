"""Distribution-based market risk measures.

Convention: VaR/CVaR are returned as positive loss fractions. For example ``0.03`` means
"3% loss" at the requested confidence level.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from quant_os_core.quant._validation import as_float_array, require_confidence


def historical_var(returns: ArrayLike, *, confidence: float = 0.95) -> float:
    r = as_float_array(returns, name="returns", min_size=2)
    c = require_confidence(confidence)
    quantile_return = float(np.quantile(r, 1.0 - c))
    return max(0.0, -quantile_return)


def historical_cvar(returns: ArrayLike, *, confidence: float = 0.95) -> float:
    r = as_float_array(returns, name="returns", min_size=2)
    c = require_confidence(confidence)
    cutoff = float(np.quantile(r, 1.0 - c))
    tail = r[r <= cutoff]
    return max(0.0, -float(np.mean(tail)))


def gaussian_var(returns: ArrayLike, *, confidence: float = 0.95) -> float:
    r = as_float_array(returns, name="returns", min_size=2)
    c = require_confidence(confidence)
    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1))
    if std == 0.0:
        return max(0.0, -mean)
    q = mean + std * float(norm.ppf(1.0 - c))
    return max(0.0, -q)


def gaussian_cvar(returns: ArrayLike, *, confidence: float = 0.95) -> float:
    r = as_float_array(returns, name="returns", min_size=2)
    c = require_confidence(confidence)
    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1))
    if std == 0.0:
        return max(0.0, -mean)
    alpha = 1.0 - c
    z = float(norm.ppf(alpha))
    tail_mean = mean - std * float(norm.pdf(z)) / alpha
    return max(0.0, -tail_mean)
