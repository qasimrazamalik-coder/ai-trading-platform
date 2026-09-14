import numpy as np
import pytest

from quant_os_core.analytics.series import (
    annualized_funding_rate,
    drawdown_series,
    liquidity_score,
    rolling_volatility_series,
)


def test_drawdown_series() -> None:
    result = drawdown_series([0.1, -0.1, 0.05])
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(-0.1)
    assert result[2] < 0.0
    with pytest.raises(ValueError):
        drawdown_series([-1.0])


def test_rolling_volatility_series() -> None:
    result = rolling_volatility_series([0.01, -0.01, 0.02, 0.0], window=3, periods_per_year=252)
    assert np.isnan(result[:2]).all()
    assert np.isfinite(result[2:]).all()
    with pytest.raises(ValueError):
        rolling_volatility_series([0.01], window=2)


def test_liquidity_score_rewards_depth_and_penalizes_spread() -> None:
    result = liquidity_score([1.0, 10.0], [1_000_000.0, 1_000_000.0])
    assert result[0] > result[1]
    with pytest.raises(ValueError):
        liquidity_score([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        liquidity_score([-1.0], [1.0])


def test_annualized_funding_rate() -> None:
    result = annualized_funding_rate([0.0001, -0.0002], periods_per_year=1095)
    assert result.tolist() == pytest.approx([0.1095, -0.219])
    with pytest.raises(ValueError):
        annualized_funding_rate([0.1], periods_per_year=0)
