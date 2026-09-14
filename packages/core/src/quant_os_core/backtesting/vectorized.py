"""Fast single-asset vectorized strategy screening."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class VectorizedBacktestResult:
    asset_returns: npt.NDArray[np.float64]
    strategy_returns: npt.NDArray[np.float64]
    applied_positions: npt.NDArray[np.float64]
    equity_curve: npt.NDArray[np.float64]
    turnover: npt.NDArray[np.float64]
    total_return: float


def run_vectorized_backtest(
    prices: npt.ArrayLike,
    positions: npt.ArrayLike,
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    signal_delay: int = 1,
) -> VectorizedBacktestResult:
    """Backtest target positions using close-to-close returns.

    ``signal_delay=1`` means a position derived from close ``t`` is first applied to the
    return from ``t`` to ``t+1``. A zero delay is allowed only for diagnostic experiments.
    """

    px = np.asarray(prices, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    if px.ndim != 1 or pos.ndim != 1 or px.size != pos.size:
        raise ValueError("prices and positions must be same-length 1D arrays")
    if px.size < 2:
        raise ValueError("at least two prices are required")
    if not np.all(np.isfinite(px)) or not np.all(np.isfinite(pos)):
        raise ValueError("prices and positions must be finite")
    if np.any(px <= 0):
        raise ValueError("prices must be > 0")
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("costs cannot be negative")
    if signal_delay < 0:
        raise ValueError("signal_delay cannot be negative")

    asset_returns = px[1:] / px[:-1] - 1.0
    applied = np.zeros(px.size - 1, dtype=np.float64)
    usable = applied.size - signal_delay
    if usable > 0:
        applied[signal_delay:] = pos[:usable]

    turnover = np.abs(np.diff(np.concatenate(([0.0], applied))))
    cost_rate = (fee_bps + slippage_bps) / 10000.0
    strategy_returns = applied * asset_returns - turnover * cost_rate
    equity = np.cumprod(1.0 + strategy_returns)
    total_return = float(equity[-1] - 1.0)
    return VectorizedBacktestResult(
        asset_returns=asset_returns,
        strategy_returns=strategy_returns,
        applied_positions=applied,
        equity_curve=equity,
        turnover=turnover,
        total_return=total_return,
    )
