import numpy as np
import pytest

from quant_os_core.strategies import (
    annualized_basis,
    cash_and_carry_signal,
    credibility_weighted_event_signal,
    decay_event_signal,
    funding_carry_signal,
    inverse_volatility_weights,
    microprice,
    order_book_imbalance,
    trade_flow_imbalance,
    volatility_breakout_signal,
    vwap_deviation_signal,
)


def test_funding_carry_is_contrarian_by_default() -> None:
    signal = funding_carry_signal([0.001, 0, -0.001], threshold=0.0005)
    np.testing.assert_array_equal(signal, [-1, 0, 1])


def test_annualized_basis() -> None:
    basis = annualized_basis([100], [101], days_to_expiry=[30])
    assert basis[0] == pytest.approx(0.01 * 365 / 30)


def test_cash_and_carry_signal_after_financing() -> None:
    signal = cash_and_carry_signal(
        [100, 100],
        [103, 100],
        days_to_expiry=[30, 30],
        financing_rate=0.05,
        min_edge=0.10,
    )
    assert signal[0] == 1.0
    assert signal[1] == 0.0


def test_order_book_imbalance_bounds() -> None:
    result = order_book_imbalance([9, 1, 0], [1, 9, 0])
    np.testing.assert_allclose(result, [0.8, -0.8, 0.0])


def test_trade_flow_imbalance() -> None:
    result = trade_flow_imbalance([8, 2], [2, 8])
    np.testing.assert_allclose(result, [0.6, -0.6])


def test_microprice_leans_toward_ask_when_bid_depth_is_large() -> None:
    result = microprice([99], [101], [9], [1])
    assert result[0] == pytest.approx(100.8)


def test_vwap_deviation_signal_mean_reverts() -> None:
    result = vwap_deviation_signal([101, 99, 100], [100, 100, 100], threshold_fraction=0.005)
    np.testing.assert_array_equal(result, [-1, 1, 0])


def test_volatility_breakout() -> None:
    returns = np.array([0.0] * 20 + [0.1, -0.1])
    signal = volatility_breakout_signal(returns, window=10, threshold_std=2.0)
    assert signal[-2] == 1.0
    assert signal[-1] == -1.0


def test_inverse_volatility_weights_sum_to_one() -> None:
    weights = inverse_volatility_weights([0.1, 0.2, 0.4])
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] > weights[1] > weights[2]


def test_credibility_weighted_event_signal_filters_low_trust() -> None:
    result = credibility_weighted_event_signal(
        [0.9, -0.9, 0.9],
        [0.9, 0.9, 0.2],
        minimum_credibility=0.7,
        activation_threshold=0.5,
    )
    np.testing.assert_array_equal(result, [1, -1, 0])


def test_event_signal_decays() -> None:
    result = decay_event_signal([1.0, 0.0, 0.0], decay=0.5)
    np.testing.assert_allclose(result, [1.0, 0.5, 0.25])


@pytest.mark.parametrize(
    "call",
    [
        lambda: funding_carry_signal([0.1], threshold=-1),
        lambda: annualized_basis([100], [101], days_to_expiry=[0]),
        lambda: order_book_imbalance([-1], [1]),
        lambda: microprice([101], [100], [1], [1]),
        lambda: inverse_volatility_weights([0.1, 0.0]),
        lambda: credibility_weighted_event_signal([2], [1]),
    ],
)
def test_alt_strategy_guards(call) -> None:
    with pytest.raises(ValueError):
        call()
