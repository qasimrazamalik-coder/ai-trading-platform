"""Performance summaries derived directly from event-driven equity ledgers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from quant_os_core.backtesting.models import BacktestResult
from quant_os_core.quant.performance import (
    annualized_return,
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio,
)
from quant_os_core.quant.returns import max_drawdown


@dataclass(frozen=True, slots=True)
class BacktestSummary:
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe: float | None
    sortino: float | None
    calmar: float | None
    fill_count: int
    submitted_orders: int
    rejected_orders: int
    cancelled_orders: int


def equity_returns(result: BacktestResult) -> np.ndarray:
    values = np.asarray([float(point.equity) for point in result.equity_curve], dtype=np.float64)
    if values.size < 2:
        return np.asarray([], dtype=np.float64)
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("equity curve must be finite and strictly positive")
    return values[1:] / values[:-1] - 1.0


def summarize_backtest(
    result: BacktestResult,
    *,
    periods_per_year: float = 252.0,
) -> BacktestSummary:
    returns = equity_returns(result)
    if returns.size == 0:
        return BacktestSummary(
            total_return=float(result.total_return),
            annualized_return=0.0,
            max_drawdown=0.0,
            sharpe=None,
            sortino=None,
            calmar=None,
            fill_count=len(result.fills),
            submitted_orders=result.submitted_orders,
            rejected_orders=result.rejected_orders,
            cancelled_orders=result.cancelled_orders,
        )

    def optional_metric(function: Callable[..., float]) -> float | None:
        try:
            return float(function(returns, periods_per_year=periods_per_year))
        except ValueError:
            return None

    return BacktestSummary(
        total_return=float(result.total_return),
        annualized_return=float(annualized_return(returns, periods_per_year=periods_per_year)),
        max_drawdown=float(max_drawdown(returns)),
        sharpe=optional_metric(sharpe_ratio),
        sortino=optional_metric(sortino_ratio),
        calmar=optional_metric(calmar_ratio),
        fill_count=len(result.fills),
        submitted_orders=result.submitted_orders,
        rejected_orders=result.rejected_orders,
        cancelled_orders=result.cancelled_orders,
    )
