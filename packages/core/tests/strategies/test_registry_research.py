import numpy as np
import pytest

from quant_os_core.strategies import (
    ParameterDefinition,
    StrategyCategory,
    StrategyMetadata,
    StrategyRegistry,
    StrategySignals,
    create_default_registry,
    evaluate_strategy_signals,
)


def test_parameter_definition_guards_bounds() -> None:
    with pytest.raises(ValueError, match="default"):
        ParameterDefinition("x", default=5, minimum=0, maximum=4)


def test_strategy_signals_reject_leverage() -> None:
    metadata = StrategyMetadata("test", "Test", StrategyCategory.TREND, "test")
    with pytest.raises(ValueError, match="inside"):
        StrategySignals(np.array([0.0, 1.1]), metadata)


def test_default_registry_has_core_strategies() -> None:
    registry = create_default_registry()
    ids = {item.strategy_id for item in registry.list_metadata()}
    assert {
        "trend.ma_crossover",
        "breakout.donchian",
        "momentum.time_series",
        "mean_reversion.zscore",
        "mean_reversion.bollinger",
    }.issubset(ids)


def test_registry_duplicate_protection() -> None:
    registry = StrategyRegistry()
    metadata = StrategyMetadata("x", "X", StrategyCategory.TREND, "test")

    def factory(prices):
        return StrategySignals(np.zeros(len(prices)), metadata)

    registry.register(metadata, factory)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(metadata, factory)


def test_registry_unknown_strategy() -> None:
    with pytest.raises(KeyError, match="unknown strategy"):
        StrategyRegistry().get("missing")


def test_strategy_research_bridge_runs_backtester_and_fit_diagnostic() -> None:
    rng = np.random.default_rng(10)
    returns = rng.normal(0.001, 0.01, 300)
    prices = 100 * np.cumprod(1 + returns)
    positions = np.ones(prices.size)
    result = evaluate_strategy_signals(prices, positions, periods_per_year=252)
    assert np.isfinite(result.annualized_sharpe)
    assert result.backtest.equity_curve.size == prices.size - 1
    assert result.fit_diagnosis.value in {"underfit", "right_fit", "overfit", "inconclusive"}


def test_strategy_research_rejects_bad_train_fraction() -> None:
    prices = np.arange(1.0, 30.0)
    with pytest.raises(ValueError, match="train_fraction"):
        evaluate_strategy_signals(prices, np.ones(prices.size), train_fraction=0.99)
