"""Risk-adjusted performance metrics with explicit annualization conventions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from quant_os_core.quant._validation import as_float_array, require_positive
from quant_os_core.quant.returns import cumulative_return, max_drawdown


def annualized_return(returns: ArrayLike, *, periods_per_year: float = 252.0) -> float:
    """Geometrically annualize periodic simple returns."""
    r = as_float_array(returns, name="returns", min_size=1)
    periods = require_positive(periods_per_year, name="periods_per_year")
    if np.any(r < -1.0):
        raise ValueError("simple returns cannot be less than -100%")
    growth = 1.0 + cumulative_return(r)
    if growth <= 0.0:
        return -1.0
    return float(growth ** (periods / r.size) - 1.0)


def sharpe_ratio(
    returns: ArrayLike,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Annualized Sharpe using arithmetic excess-period returns.

    ``risk_free_rate`` is an annual simple rate and is converted to a per-period rate.
    """
    r = as_float_array(returns, name="returns", min_size=2)
    periods = require_positive(periods_per_year, name="periods_per_year")
    if risk_free_rate <= -1.0 or not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite and > -1")
    rf_period = (1.0 + risk_free_rate) ** (1.0 / periods) - 1.0
    excess = r - rf_period
    volatility = float(np.std(excess, ddof=1))
    if volatility == 0.0:
        raise ValueError("Sharpe ratio is undefined for zero volatility")
    return float(np.mean(excess) / volatility * np.sqrt(periods))


def downside_deviation(
    returns: ArrayLike,
    *,
    minimum_acceptable_return: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    r = as_float_array(returns, name="returns", min_size=1)
    periods = require_positive(periods_per_year, name="periods_per_year")
    if minimum_acceptable_return <= -1.0 or not np.isfinite(minimum_acceptable_return):
        raise ValueError("minimum_acceptable_return must be finite and > -1")
    mar_period = (1.0 + minimum_acceptable_return) ** (1.0 / periods) - 1.0
    downside = np.minimum(r - mar_period, 0.0)
    return float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods))


def sortino_ratio(
    returns: ArrayLike,
    *,
    minimum_acceptable_return: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    r = as_float_array(returns, name="returns", min_size=2)
    periods = require_positive(periods_per_year, name="periods_per_year")
    if minimum_acceptable_return <= -1.0 or not np.isfinite(minimum_acceptable_return):
        raise ValueError("minimum_acceptable_return must be finite and > -1")
    mar_period = (1.0 + minimum_acceptable_return) ** (1.0 / periods) - 1.0
    downside = np.minimum(r - mar_period, 0.0)
    downside_std = float(np.sqrt(np.mean(np.square(downside))))
    if downside_std == 0.0:
        raise ValueError("Sortino ratio is undefined with no downside deviation")
    return float((np.mean(r) - mar_period) / downside_std * np.sqrt(periods))


def calmar_ratio(returns: ArrayLike, *, periods_per_year: float = 252.0) -> float:
    r = as_float_array(returns, name="returns", min_size=2)
    drawdown = abs(max_drawdown(r))
    if drawdown == 0.0:
        raise ValueError("Calmar ratio is undefined with zero maximum drawdown")
    return annualized_return(r, periods_per_year=periods_per_year) / drawdown


def omega_ratio(returns: ArrayLike, *, threshold: float = 0.0) -> float:
    """Omega ratio = gains above threshold / losses below threshold."""
    r = as_float_array(returns, name="returns", min_size=1)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    excess = r - threshold
    gains = float(np.sum(np.clip(excess, 0.0, None)))
    losses = float(-np.sum(np.clip(excess, None, 0.0)))
    if losses == 0.0:
        raise ValueError("Omega ratio is undefined when there are no losses below threshold")
    return gains / losses


def profit_factor(trade_returns: ArrayLike) -> float:
    r = as_float_array(trade_returns, name="trade_returns", min_size=1)
    gross_profit = float(r[r > 0].sum())
    gross_loss = float(-r[r < 0].sum())
    if gross_loss == 0.0:
        raise ValueError("profit factor is undefined without losing trades")
    return gross_profit / gross_loss


def expectancy(trade_returns: ArrayLike) -> float:
    r = as_float_array(trade_returns, name="trade_returns", min_size=1)
    return float(np.mean(r))


def hit_rate(trade_returns: ArrayLike) -> float:
    r = as_float_array(trade_returns, name="trade_returns", min_size=1)
    return float(np.mean(r > 0.0))
