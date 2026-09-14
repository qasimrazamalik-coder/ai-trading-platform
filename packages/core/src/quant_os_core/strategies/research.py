"""Bridges strategy signals to the vectorized backtester and fit diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quant_os_core.backtesting.vectorized import VectorizedBacktestResult, run_vectorized_backtest
from quant_os_core.quant.performance import sharpe_ratio
from quant_os_core.right_fit.selection import FitDiagnosis, diagnose_fit


@dataclass(frozen=True, slots=True)
class StrategyResearchResult:
    backtest: VectorizedBacktestResult
    annualized_sharpe: float
    train_sharpe: float
    test_sharpe: float
    fit_diagnosis: FitDiagnosis


def _safe_sharpe(returns: npt.NDArray[np.float64], periods_per_year: int) -> float:
    if returns.size < 2 or float(np.std(returns, ddof=1)) <= np.finfo(np.float64).eps:
        return 0.0
    return float(sharpe_ratio(returns, periods_per_year=periods_per_year))


def evaluate_strategy_signals(
    prices: npt.ArrayLike,
    positions: npt.ArrayLike,
    *,
    periods_per_year: int = 252,
    train_fraction: float = 0.7,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    signal_delay: int = 1,
) -> StrategyResearchResult:
    result = run_vectorized_backtest(
        prices,
        positions,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        signal_delay=signal_delay,
    )
    fraction = float(train_fraction)
    if not 0.2 <= fraction <= 0.9:
        raise ValueError("train_fraction must be between 0.2 and 0.9")
    split = int(np.floor(result.strategy_returns.size * fraction))
    if split < 2 or result.strategy_returns.size - split < 2:
        raise ValueError("not enough returns for requested train/test split")
    train = result.strategy_returns[:split]
    test = result.strategy_returns[split:]
    train_sharpe = _safe_sharpe(train, periods_per_year)
    test_sharpe = _safe_sharpe(test, periods_per_year)
    diagnosis = diagnose_fit(
        train_score=train_sharpe,
        validation_score=test_sharpe,
    )
    return StrategyResearchResult(
        backtest=result,
        annualized_sharpe=_safe_sharpe(result.strategy_returns, periods_per_year),
        train_sharpe=train_sharpe,
        test_sharpe=test_sharpe,
        fit_diagnosis=diagnosis,
    )
