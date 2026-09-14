import numpy as np
import pytest

from quant_os_core.strategies import (
    CanSlimInputs,
    MinerviniInputs,
    avellaneda_stoikov_quote,
    canslim_score,
    majority_vote,
    minervini_trend_template,
    pairs_zscore_signal,
    weighted_ensemble,
)


def test_pairs_signal_recovers_hedge_ratio() -> None:
    rng = np.random.default_rng(7)
    x = 100 + np.cumsum(rng.normal(0, 0.5, 250))
    stationary = np.zeros(250)
    for index in range(1, stationary.size):
        stationary[index] = 0.75 * stationary[index - 1] + rng.normal(0, 0.2)
    y = 5 + 1.8 * x + stationary
    result = pairs_zscore_signal(y, x, window=40, require_cointegration=False)
    assert result.hedge_ratio == pytest.approx(1.8, rel=0.02)
    assert result.spread.shape == x.shape
    assert set(np.unique(result.long_y_short_x)).issubset({-1.0, 0.0, 1.0})


def test_pairs_signal_blocks_non_cointegrated_pair_when_required() -> None:
    rng = np.random.default_rng(2)
    x = 100 + np.cumsum(rng.normal(size=220))
    y = 50 + np.cumsum(rng.normal(size=220))
    result = pairs_zscore_signal(
        y,
        x,
        window=30,
        require_cointegration=True,
        max_cointegration_pvalue=0.0,
    )
    assert np.all(result.long_y_short_x == 0.0)


def test_minervini_template_passes_strong_setup() -> None:
    result = minervini_trend_template(
        MinerviniInputs(
            price=100,
            sma_50=90,
            sma_150=80,
            sma_200=70,
            sma_200_20d_ago=68,
            low_52w=60,
            high_52w=110,
            relative_strength_percentile=90,
        )
    )
    assert result.passed
    assert result.score == pytest.approx(1.0)


def test_minervini_template_scores_weak_setup_lower() -> None:
    result = minervini_trend_template(
        MinerviniInputs(
            price=80,
            sma_50=85,
            sma_150=90,
            sma_200=95,
            sma_200_20d_ago=96,
            low_52w=70,
            high_52w=120,
            relative_strength_percentile=40,
        )
    )
    assert not result.passed
    assert result.score < 0.5


def test_canslim_score_is_bounded() -> None:
    result = canslim_score(
        CanSlimInputs(
            quarterly_eps_growth=0.40,
            annual_eps_growth=0.35,
            revenue_growth=0.30,
            catalyst_score=1.0,
            supply_demand_score=0.9,
            relative_strength_percentile=95,
            institutional_sponsorship_score=0.8,
            market_regime_score=0.9,
        )
    )
    assert 0.0 <= result.total <= 1.0
    assert result.total > 0.85


def test_weighted_ensemble() -> None:
    signals = [[1, 1, -1], [-1, 1, 1]]
    result = weighted_ensemble(signals, weights=[0.75, 0.25])
    np.testing.assert_allclose(result, [0.5, 1.0, -0.5])


def test_majority_vote_requires_agreement() -> None:
    signals = [[1, 1, -1], [1, -1, -1], [-1, 1, 0]]
    result = majority_vote(signals, minimum_agreement=2 / 3)
    np.testing.assert_array_equal(result, [1, 1, -1])


def test_avellaneda_stoikov_inventory_shifts_reservation() -> None:
    neutral = avellaneda_stoikov_quote(
        mid_price=100,
        inventory=0,
        volatility=0.02,
        risk_aversion=0.1,
        time_remaining=1.0,
        liquidity_k=1.5,
    )
    long_inventory = avellaneda_stoikov_quote(
        mid_price=100,
        inventory=10,
        volatility=0.02,
        risk_aversion=0.1,
        time_remaining=1.0,
        liquidity_k=1.5,
    )
    assert long_inventory.reservation_price < neutral.reservation_price
    assert neutral.bid < neutral.reservation_price < neutral.ask


def test_market_making_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        avellaneda_stoikov_quote(
            mid_price=-1,
            inventory=0,
            volatility=0.02,
            risk_aversion=0.1,
            time_remaining=1.0,
            liquidity_k=1.0,
        )
