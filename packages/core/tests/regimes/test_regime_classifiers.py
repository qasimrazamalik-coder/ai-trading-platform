import numpy as np
import pytest

from quant_os_core.regimes import (
    CorrelationRegime,
    LiquidityRegime,
    MacroRegime,
    RiskRegime,
    TrendRegime,
    VolatilityRegime,
    average_absolute_correlation,
    classify_correlation,
    classify_liquidity,
    classify_macro,
    classify_risk,
    classify_trend,
    classify_volatility,
    regime_snapshot,
)


def test_trend_classifies_clear_bull_and_bear() -> None:
    bull = np.linspace(100.0, 180.0, 120)
    bear = bull[::-1]
    assert classify_trend(bull).label == TrendRegime.BULL.value
    assert classify_trend(bear).label == TrendRegime.BEAR.value


def test_trend_classifies_flat_as_sideways() -> None:
    prices = np.full(120, 100.0)
    assert classify_trend(prices).label == TrendRegime.SIDEWAYS.value


def test_trend_guards_bad_input() -> None:
    with pytest.raises(ValueError):
        classify_trend(np.ones(20), fast_window=20, slow_window=80)
    with pytest.raises(ValueError):
        classify_trend(np.full(100, -1.0), fast_window=10, slow_window=40)
    with pytest.raises(ValueError):
        classify_trend(np.ones(100), fast_window=40, slow_window=20)


def test_volatility_detects_quiet_and_hot_tail() -> None:
    quiet = np.full(200, 0.001)
    history = np.concatenate([quiet, np.linspace(-0.001, 0.001, 30), np.full(20, 0.08)])
    result = classify_volatility(history, lookback=20)
    assert result.label == VolatilityRegime.HIGH.value


def test_volatility_guards_quantiles() -> None:
    with pytest.raises(ValueError):
        classify_volatility(np.arange(40.0), low_quantile=0.8, high_quantile=0.2)


def test_risk_on_off_and_neutral() -> None:
    assert classify_risk(np.full(20, 0.01), np.zeros(20)).label == RiskRegime.RISK_ON.value
    assert classify_risk(np.full(20, -0.01), np.zeros(20)).label == RiskRegime.RISK_OFF.value
    assert classify_risk(np.zeros(20), np.zeros(20)).label == RiskRegime.NEUTRAL.value


def test_risk_requires_equal_lengths() -> None:
    with pytest.raises(ValueError):
        classify_risk(np.zeros(5), np.zeros(4))


def test_liquidity_classification() -> None:
    abundant = classify_liquidity(spread_bps=1.0, depth_zscore=3.0, volume_zscore=2.0)
    tight = classify_liquidity(spread_bps=50.0, depth_zscore=-3.0, volume_zscore=-2.0)
    normal = classify_liquidity(spread_bps=1.0, depth_zscore=0.0, volume_zscore=0.0)
    assert abundant.label == LiquidityRegime.ABUNDANT.value
    assert tight.label == LiquidityRegime.TIGHT.value
    assert normal.label == LiquidityRegime.NORMAL.value


def test_liquidity_guards() -> None:
    with pytest.raises(ValueError):
        classify_liquidity(spread_bps=-1.0, depth_zscore=0.0, volume_zscore=0.0)
    with pytest.raises(ValueError):
        classify_liquidity(
            spread_bps=1.0,
            depth_zscore=0.0,
            volume_zscore=0.0,
            tight_threshold=1.0,
            abundant_threshold=0.0,
        )


def test_average_absolute_correlation_and_regimes() -> None:
    matrix = np.array([[1.0, 0.8, -0.7], [0.8, 1.0, 0.75], [-0.7, 0.75, 1.0]])
    assert average_absolute_correlation(matrix) == pytest.approx(0.75)
    assert classify_correlation(matrix).label == CorrelationRegime.HIGH.value


def test_correlation_guard() -> None:
    with pytest.raises(ValueError):
        average_absolute_correlation(np.ones((2, 3)))
    with pytest.raises(ValueError):
        average_absolute_correlation(np.array([[1.0, 1.2], [1.2, 1.0]]))


@pytest.mark.parametrize(
    ("growth", "inflation", "expected"),
    [
        (1.0, -1.0, MacroRegime.GOLDILOCKS.value),
        (1.0, 1.0, MacroRegime.REFLATION.value),
        (-1.0, 1.0, MacroRegime.STAGFLATION.value),
        (-1.0, -1.0, MacroRegime.DEFLATION.value),
        (0.05, 1.0, MacroRegime.MIXED.value),
    ],
)
def test_macro_regimes(growth: float, inflation: float, expected: str) -> None:
    assert classify_macro(growth_zscore=growth, inflation_zscore=inflation).label == expected


def test_regime_snapshot_optional_inputs() -> None:
    prices = np.linspace(100.0, 160.0, 120)
    returns = np.diff(np.log(prices))
    snapshot = regime_snapshot(
        prices=prices,
        returns=returns,
        growth_asset_returns=np.full(30, 0.01),
        defensive_asset_returns=np.zeros(30),
        correlation_matrix=np.array([[1.0, 0.8], [0.8, 1.0]]),
        spread_bps=1.0,
        depth_zscore=2.0,
        volume_zscore=2.0,
        growth_zscore=1.0,
        inflation_zscore=-1.0,
    )
    assert snapshot.trend.label == TrendRegime.BULL.value
    assert snapshot.correlation is not None
    assert snapshot.liquidity is not None
    assert snapshot.macro is not None


def test_snapshot_rejects_partial_optional_groups() -> None:
    prices = np.linspace(100.0, 160.0, 120)
    returns = np.diff(np.log(prices))
    kwargs = {
        "prices": prices,
        "returns": returns,
        "growth_asset_returns": np.zeros(30),
        "defensive_asset_returns": np.zeros(30),
    }
    with pytest.raises(ValueError):
        regime_snapshot(**kwargs, spread_bps=1.0)
    with pytest.raises(ValueError):
        regime_snapshot(**kwargs, growth_zscore=1.0)
