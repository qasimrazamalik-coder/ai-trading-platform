"""Benchmark-relative analysis for backtest equity curves."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quant_os_core.backtesting.analytics import equity_returns
from quant_os_core.backtesting.models import BacktestResult
from quant_os_core.quant.regression import beta, information_ratio, tracking_error


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    strategy_total_return: float
    benchmark_total_return: float
    active_total_return: float
    beta: float | None
    tracking_error: float | None
    information_ratio: float | None


def compare_to_benchmark(
    result: BacktestResult,
    benchmark_equity: npt.ArrayLike,
    *,
    periods_per_year: float = 252.0,
) -> BenchmarkComparison:
    strategy = equity_returns(result)
    benchmark_values = np.asarray(benchmark_equity, dtype=np.float64)
    if benchmark_values.ndim != 1 or benchmark_values.size != len(result.equity_curve):
        raise ValueError("benchmark_equity must match the event equity-curve length")
    if not np.all(np.isfinite(benchmark_values)) or np.any(benchmark_values <= 0):
        raise ValueError("benchmark_equity must be finite and strictly positive")
    if strategy.size == 0:
        benchmark_total = 0.0
        if benchmark_values.size > 1:
            benchmark_total = float(benchmark_values[-1] / benchmark_values[0] - 1.0)
        return BenchmarkComparison(
            strategy_total_return=float(result.total_return),
            benchmark_total_return=benchmark_total,
            active_total_return=float(result.total_return) - benchmark_total,
            beta=None,
            tracking_error=None,
            information_ratio=None,
        )

    benchmark = benchmark_values[1:] / benchmark_values[:-1] - 1.0
    benchmark_total = float(benchmark_values[-1] / benchmark_values[0] - 1.0)

    def optional(function: Callable[..., float]) -> float | None:
        try:
            return float(function(strategy, benchmark, periods_per_year=periods_per_year))
        except TypeError:
            try:
                return float(function(strategy, benchmark))
            except ValueError:
                return None
        except ValueError:
            return None

    return BenchmarkComparison(
        strategy_total_return=float(result.total_return),
        benchmark_total_return=benchmark_total,
        active_total_return=float(result.total_return) - benchmark_total,
        beta=optional(beta),
        tracking_error=optional(tracking_error),
        information_ratio=optional(information_ratio),
    )
