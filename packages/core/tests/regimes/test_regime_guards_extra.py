import numpy as np
import pytest

from quant_os_core.regimes import (
    RegimeEstimate,
    StrategyRegimeMatrix,
    classify_correlation,
    classify_liquidity,
    classify_macro,
    classify_risk,
    classify_trend,
    classify_volatility,
    dynamic_strategy_weights,
    normalized_entropy,
    stationary_distribution,
    transition_matrix,
)


def test_regime_estimate_guards() -> None:
    with pytest.raises(ValueError):
        RegimeEstimate("", 0.5, 0.0)
    with pytest.raises(ValueError):
        RegimeEstimate("x", 1.1, 0.0)
    with pytest.raises(ValueError):
        RegimeEstimate("x", 0.5, np.inf)


def test_classifier_finite_and_parameter_guards() -> None:
    prices = np.linspace(100.0, 120.0, 100)
    with pytest.raises(ValueError):
        classify_trend(np.append(prices, np.nan), fast_window=10, slow_window=40)
    with pytest.raises(ValueError):
        classify_trend(prices, fast_window=10, slow_window=40, neutral_band=-0.1)
    with pytest.raises(ValueError):
        classify_volatility(np.ones(50), lookback=1)
    with pytest.raises(ValueError):
        classify_risk(np.ones(5), np.ones(5), neutral_band=-1.0)
    with pytest.raises(ValueError):
        classify_liquidity(spread_bps=np.nan, depth_zscore=0.0, volume_zscore=0.0)
    with pytest.raises(ValueError):
        classify_correlation(np.eye(2), low_threshold=0.8, high_threshold=0.2)
    with pytest.raises(ValueError):
        classify_macro(growth_zscore=np.nan, inflation_zscore=0.0)
    with pytest.raises(ValueError):
        classify_macro(growth_zscore=0.0, inflation_zscore=0.0, neutral_band=-0.1)


def test_correlation_low_and_normal_paths() -> None:
    low = np.array([[1.0, 0.1], [0.1, 1.0]])
    normal = np.array([[1.0, 0.4], [0.4, 1.0]])
    assert classify_correlation(low).label == "low_correlation"
    assert classify_correlation(normal).label == "normal_correlation"


def test_volatility_normal_path() -> None:
    rng = np.random.default_rng(55)
    values = rng.normal(0.0, 0.01, 260)
    result = classify_volatility(values, lookback=20)
    assert result.label in {"low_volatility", "normal_volatility", "high_volatility"}


def test_transition_and_entropy_more_guards() -> None:
    with pytest.raises(ValueError):
        transition_matrix([0, 1], smoothing=-0.1)
    with pytest.raises(ValueError):
        stationary_distribution(np.array([[1.0, -0.1], [0.0, 1.1]]))
    with pytest.raises(ValueError):
        stationary_distribution(np.array([[np.nan, 0.0], [0.0, 1.0]]))
    assert normalized_entropy([1.0]) == pytest.approx(0.0)


def test_strategy_weight_guards() -> None:
    matrix = StrategyRegimeMatrix()
    with pytest.raises(ValueError):
        matrix.get("a", "x", default=2.0)
    with pytest.raises(ValueError):
        dynamic_strategy_weights({"a": 1.0}, regime="x", compatibility=matrix, confidence=2.0)
    with pytest.raises(ValueError):
        dynamic_strategy_weights(
            {"a": 1.0}, regime="x", compatibility=matrix, concentration_power=0.0
        )
    with pytest.raises(ValueError):
        dynamic_strategy_weights(
            {"a": 1.0},
            regime="x",
            compatibility=matrix,
            performance_scores={"a": -1.0},
        )
    with pytest.raises(ValueError):
        dynamic_strategy_weights({"a": 0.0}, regime="x", compatibility=matrix)


def test_volatility_clustering_score() -> None:
    from quant_os_core.regimes import volatility_clustering_score

    rng = np.random.default_rng(123)
    shocks = np.zeros(400)
    variance = 0.0001
    for i in range(1, shocks.size):
        variance = 0.00002 + 0.92 * variance + 0.05 * shocks[i - 1] ** 2
        shocks[i] = rng.normal(0.0, np.sqrt(variance))
    assert volatility_clustering_score(shocks) > 0.0
    with pytest.raises(ValueError):
        volatility_clustering_score([0.1, 0.2, 0.3], lag=0)
