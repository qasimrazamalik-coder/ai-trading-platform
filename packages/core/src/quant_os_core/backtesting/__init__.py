"""Institutional backtesting primitives."""

from quant_os_core.backtesting.analytics import BacktestSummary, summarize_backtest
from quant_os_core.backtesting.benchmark import BenchmarkComparison, compare_to_benchmark
from quant_os_core.backtesting.broker import BrokerConfig, SimulatedBroker
from quant_os_core.backtesting.costs import FinancingModel, ProportionalFeeModel
from quant_os_core.backtesting.engine import EventDrivenBacktester, StrategyContext
from quant_os_core.backtesting.models import (
    BacktestResult,
    Fill,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from quant_os_core.backtesting.portfolio import Portfolio
from quant_os_core.backtesting.slippage import BpsSlippageModel
from quant_os_core.backtesting.vectorized import (
    VectorizedBacktestResult,
    run_vectorized_backtest,
)

__all__ = [
    "BacktestResult",
    "BacktestSummary",
    "BenchmarkComparison",
    "BpsSlippageModel",
    "BrokerConfig",
    "EventDrivenBacktester",
    "Fill",
    "FinancingModel",
    "OrderRequest",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Portfolio",
    "ProportionalFeeModel",
    "SimulatedBroker",
    "StrategyContext",
    "TimeInForce",
    "VectorizedBacktestResult",
    "run_vectorized_backtest",
    "summarize_backtest",
    "compare_to_benchmark",
]
