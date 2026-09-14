"""Portfolio heat, leverage, and stress-risk analytics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class StressResult:
    shock_name: str
    pnl_fraction: float
    stressed_equity_fraction: float


def portfolio_heat_fraction(
    position_sizes: ArrayLike,
    entry_prices: ArrayLike,
    stop_prices: ArrayLike,
    equity: float,
) -> float:
    sizes = np.asarray(position_sizes, dtype=np.float64)
    entry = np.asarray(entry_prices, dtype=np.float64)
    stops = np.asarray(stop_prices, dtype=np.float64)
    shapes_match = entry.shape == sizes.shape and stops.shape == sizes.shape
    if sizes.ndim != 1 or sizes.size == 0 or not shapes_match:
        raise ValueError("sizes, entries and stops must be equal-length 1-D arrays")
    if not np.isfinite(sizes).all() or not np.isfinite(entry).all() or not np.isfinite(stops).all():
        raise ValueError("portfolio heat inputs must be finite")
    if not np.isfinite(equity) or equity <= 0.0:
        raise ValueError("equity must be positive")
    heat = np.sum(np.abs(sizes) * np.abs(entry - stops))
    return float(heat / equity)


def leverage_ratios(notionals: ArrayLike, equity: float) -> tuple[float, float]:
    values = np.asarray(notionals, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("notionals must be a finite non-empty 1-D array")
    if not np.isfinite(equity) or equity <= 0.0:
        raise ValueError("equity must be positive")
    gross = float(np.sum(np.abs(values)) / equity)
    net = float(abs(np.sum(values)) / equity)
    return gross, net


def correlation_concentration(
    weights: ArrayLike,
    correlation: ArrayLike,
    *,
    threshold: float = 0.8,
) -> float:
    w = np.asarray(weights, dtype=np.float64)
    corr = np.asarray(correlation, dtype=np.float64)
    if w.ndim != 1 or w.size == 0 or not np.isfinite(w).all():
        raise ValueError("weights must be a finite non-empty 1-D array")
    if corr.shape != (w.size, w.size) or not np.isfinite(corr).all():
        raise ValueError("correlation must be square and match weights")
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [-1, 1]")
    abs_w = np.abs(w)
    total = float(np.sum(abs_w))
    if total <= 0.0:
        return 0.0
    max_cluster = 0.0
    for i in range(w.size):
        cluster = abs_w[np.abs(corr[i]) >= threshold]
        max_cluster = max(max_cluster, float(np.sum(cluster)))
    return max_cluster / total


def stress_portfolio(
    weights: ArrayLike,
    shocks: dict[str, ArrayLike],
) -> tuple[StressResult, ...]:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.size == 0 or not np.isfinite(w).all():
        raise ValueError("weights must be a finite non-empty 1-D array")
    if not shocks:
        raise ValueError("shocks cannot be empty")
    results: list[StressResult] = []
    for name, scenario in shocks.items():
        shock = np.asarray(scenario, dtype=np.float64)
        if shock.shape != w.shape or not np.isfinite(shock).all():
            raise ValueError("each shock must match weights and contain finite values")
        pnl = float(w @ shock)
        results.append(StressResult(name, pnl, 1.0 + pnl))
    return tuple(results)
