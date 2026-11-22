import pytest

from quant_os_core.regimes import (
    StrategyRegimeMatrix,
    StrategyRegimeScore,
    dynamic_strategy_weights,
)


def test_matrix_heatmap_and_lookup() -> None:
    matrix = StrategyRegimeMatrix(
        (
            StrategyRegimeScore("trend", "bull", 0.9),
            StrategyRegimeScore("meanrev", "bull", 0.2),
            StrategyRegimeScore("trend", "sideways", 0.3),
            StrategyRegimeScore("meanrev", "sideways", 0.9),
        )
    )
    strategies, regimes, values = matrix.heatmap()
    assert strategies == ("meanrev", "trend")
    assert regimes == ("bull", "sideways")
    assert values.shape == (2, 2)
    assert matrix.get("unknown", "bull") == pytest.approx(0.5)


def test_dynamic_weights_follow_regime_when_confident() -> None:
    matrix = StrategyRegimeMatrix(
        (
            StrategyRegimeScore("trend", "bull", 1.0),
            StrategyRegimeScore("meanrev", "bull", 0.1),
        )
    )
    weights = dynamic_strategy_weights(
        {"trend": 0.5, "meanrev": 0.5},
        regime="bull",
        compatibility=matrix,
        confidence=1.0,
    )
    assert weights["trend"] > 0.85
    assert sum(weights.values()) == pytest.approx(1.0)


def test_low_confidence_preserves_base_weights() -> None:
    matrix = StrategyRegimeMatrix((StrategyRegimeScore("trend", "bull", 1.0),))
    weights = dynamic_strategy_weights(
        {"trend": 0.5, "other": 0.5},
        regime="bull",
        compatibility=matrix,
        confidence=0.0,
    )
    assert weights == pytest.approx({"trend": 0.5, "other": 0.5})


def test_dynamic_weights_accept_performance_scores() -> None:
    matrix = StrategyRegimeMatrix()
    weights = dynamic_strategy_weights(
        {"a": 1.0, "b": 1.0},
        regime="any",
        compatibility=matrix,
        performance_scores={"a": 2.0, "b": 1.0},
    )
    assert weights["a"] == pytest.approx(2 / 3)


def test_compatibility_guards() -> None:
    with pytest.raises(ValueError):
        StrategyRegimeScore("", "bull", 0.5)
    with pytest.raises(ValueError):
        StrategyRegimeScore("a", "bull", 1.5)
    with pytest.raises(ValueError):
        dynamic_strategy_weights({}, regime="x", compatibility=StrategyRegimeMatrix())
    with pytest.raises(ValueError):
        dynamic_strategy_weights(
            {"a": -1.0},
            regime="x",
            compatibility=StrategyRegimeMatrix(),
        )


def test_composite_strategy_weights_use_multiple_regime_axes() -> None:
    from quant_os_core.regimes import composite_strategy_weights

    matrix = StrategyRegimeMatrix(
        (
            StrategyRegimeScore("trend", "bull", 1.0),
            StrategyRegimeScore("trend", "high_vol", 0.8),
            StrategyRegimeScore("meanrev", "bull", 0.2),
            StrategyRegimeScore("meanrev", "high_vol", 0.3),
        )
    )
    weights = composite_strategy_weights(
        {"trend": 0.5, "meanrev": 0.5},
        active_regimes={"bull": 1.0, "high_vol": 0.8},
        compatibility=matrix,
    )
    assert weights["trend"] > weights["meanrev"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_composite_strategy_weights_guards() -> None:
    from quant_os_core.regimes import composite_strategy_weights

    with pytest.raises(ValueError):
        composite_strategy_weights(
            {"a": 1.0}, active_regimes={}, compatibility=StrategyRegimeMatrix()
        )
    with pytest.raises(ValueError):
        composite_strategy_weights(
            {"a": 1.0},
            active_regimes={"x": 2.0},
            compatibility=StrategyRegimeMatrix(),
        )
