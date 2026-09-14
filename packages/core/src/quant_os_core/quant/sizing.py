"""Risk-based position sizing formulas."""

from __future__ import annotations

import numpy as np

from quant_os_core.quant._validation import require_positive, require_probability


def kelly_fraction(win_probability: float, payoff_ratio: float) -> float:
    """Binary-outcome Kelly fraction ``p - (1-p)/b``.

    A negative result indicates no long bet under the supplied assumptions.
    """
    p = require_probability(win_probability, name="win_probability")
    b = require_positive(payoff_ratio, name="payoff_ratio")
    return float(p - (1.0 - p) / b)


def fractional_kelly(
    win_probability: float, payoff_ratio: float, *, fraction: float = 0.25
) -> float:
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    return kelly_fraction(win_probability, payoff_ratio) * fraction


def kelly_from_moments(expected_excess_return: float, variance: float) -> float:
    """Continuous approximation ``f* = expected excess return / variance``."""
    if not np.isfinite(expected_excess_return):
        raise ValueError("expected_excess_return must be finite")
    v = require_positive(variance, name="variance")
    return float(expected_excess_return / v)


def volatility_target_weight(
    forecast_volatility: float,
    *,
    target_volatility: float,
    max_leverage: float = 1.0,
) -> float:
    forecast = require_positive(forecast_volatility, name="forecast_volatility")
    target = require_positive(target_volatility, name="target_volatility")
    cap = require_positive(max_leverage, name="max_leverage")
    return float(min(target / forecast, cap))


def fixed_fractional_position_size(
    equity: float,
    risk_fraction: float,
    entry_price: float,
    stop_price: float,
    *,
    contract_multiplier: float = 1.0,
) -> float:
    """Units sized so stop loss approximately equals ``equity * risk_fraction``."""
    capital = require_positive(equity, name="equity")
    entry = require_positive(entry_price, name="entry_price")
    stop = require_positive(stop_price, name="stop_price")
    multiplier = require_positive(contract_multiplier, name="contract_multiplier")
    if not 0.0 < risk_fraction <= 1.0:
        raise ValueError("risk_fraction must be in (0, 1]")
    distance = abs(entry - stop) * multiplier
    if distance == 0.0:
        raise ValueError("entry_price and stop_price must differ")
    return float(capital * risk_fraction / distance)
