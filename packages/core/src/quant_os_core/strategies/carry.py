"""Carry, funding and basis signal helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from quant_os_core.quant._validation import as_float_array, paired_float_arrays


def funding_carry_signal(
    funding_rates: npt.ArrayLike,
    *,
    threshold: float = 0.0001,
    contrarian: bool = True,
) -> npt.NDArray[np.float64]:
    """Return normalized exposure from funding rates.

    In contrarian mode, expensive positive funding favors a short and negative funding favors
    a long. Trend mode flips that relationship and is useful only as an experimental feature.
    """
    rates = as_float_array(funding_rates, name="funding_rates")
    level = float(threshold)
    if not np.isfinite(level) or level < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    positions = np.zeros(rates.size, dtype=np.float64)
    positions[rates > level] = -1.0 if contrarian else 1.0
    positions[rates < -level] = 1.0 if contrarian else -1.0
    return positions


def annualized_basis(
    spot_prices: npt.ArrayLike,
    futures_prices: npt.ArrayLike,
    *,
    days_to_expiry: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    spot, futures = paired_float_arrays(
        spot_prices,
        futures_prices,
        left_name="spot_prices",
        right_name="futures_prices",
        min_size=1,
    )
    days = as_float_array(days_to_expiry, name="days_to_expiry")
    if days.size != spot.size:
        raise ValueError("days_to_expiry must match price lengths")
    if np.any(spot <= 0.0) or np.any(futures <= 0.0) or np.any(days <= 0.0):
        raise ValueError("spot, futures and days_to_expiry must be > 0")
    return (futures / spot - 1.0) * (365.0 / days)


def cash_and_carry_signal(
    spot_prices: npt.ArrayLike,
    futures_prices: npt.ArrayLike,
    *,
    days_to_expiry: npt.ArrayLike,
    financing_rate: float = 0.0,
    min_edge: float = 0.01,
) -> npt.NDArray[np.float64]:
    basis = annualized_basis(
        spot_prices,
        futures_prices,
        days_to_expiry=days_to_expiry,
    )
    funding_cost = float(financing_rate)
    edge = float(min_edge)
    if not np.isfinite(funding_cost) or not np.isfinite(edge) or edge < 0.0:
        raise ValueError("financing_rate and min_edge must be finite; min_edge cannot be negative")
    net_basis = basis - funding_cost
    signal = np.zeros(basis.size, dtype=np.float64)
    signal[net_basis > edge] = 1.0
    signal[net_basis < -edge] = -1.0
    return signal
