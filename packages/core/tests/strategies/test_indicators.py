import numpy as np
import pytest

from quant_os_core.strategies import (
    atr,
    bollinger_bands,
    cumulative_vwap,
    donchian_channels,
    ema,
    rate_of_change,
    rolling_zscore,
    rsi,
    sma,
    true_range,
)


def test_sma_known_values() -> None:
    result = sma([1, 2, 3, 4, 5], 3)
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    np.testing.assert_allclose(result[2:], [2, 3, 4])


def test_ema_seeds_with_simple_mean() -> None:
    result = ema([1, 2, 3, 4, 5], 3)
    assert result[2] == pytest.approx(2.0)
    assert result[3] == pytest.approx(3.0)
    assert result[4] == pytest.approx(4.0)


def test_true_range_handles_gap() -> None:
    result = true_range([10, 13], [8, 11], [9, 12])
    np.testing.assert_allclose(result, [2, 4])


def test_atr_wilder_smoothing() -> None:
    result = atr([10, 11, 12, 13], [8, 9, 10, 11], [9, 10, 11, 12], 2)
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(2.0)
    assert result[2] == pytest.approx(2.0)


def test_rsi_rising_series_reaches_100() -> None:
    result = rsi(np.arange(1.0, 20.0), 5)
    assert result[-1] == pytest.approx(100.0)


def test_rsi_flat_series_is_neutral() -> None:
    result = rsi(np.ones(20), 5)
    assert result[-1] == pytest.approx(50.0)


def test_rolling_zscore_constant_is_zero() -> None:
    result = rolling_zscore(np.ones(10), 5)
    assert result[-1] == pytest.approx(0.0)


def test_bollinger_middle_matches_sma() -> None:
    values = np.arange(1.0, 10.0)
    lower, middle, upper = bollinger_bands(values, 3, 2.0)
    np.testing.assert_allclose(middle[2:], sma(values, 3)[2:])
    assert np.all(upper[2:] >= middle[2:])
    assert np.all(lower[2:] <= middle[2:])


def test_donchian_excludes_current_bar() -> None:
    high = [1, 2, 3, 10]
    low = [0.5, 1, 2, 9]
    upper, lower = donchian_channels(high, low, 3)
    assert upper[3] == pytest.approx(3.0)
    assert lower[3] == pytest.approx(0.5)


def test_rate_of_change() -> None:
    result = rate_of_change([100, 105, 110, 120], 2)
    assert result[2] == pytest.approx(0.10)
    assert result[3] == pytest.approx(120 / 105 - 1)


def test_cumulative_vwap_known_values() -> None:
    result = cumulative_vwap([10, 12], [8, 10], [9, 11], [1, 3])
    assert result[0] == pytest.approx(9.0)
    assert result[1] == pytest.approx((9 + 33) / 4)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: sma([1, 2], 0), "positive integer"),
        (lambda: ema([1, 2], 3), "cannot exceed"),
        (lambda: true_range([1], [2], [1.5]), "high cannot"),
        (lambda: rsi([1, 2], 2), "smaller"),
        (lambda: rate_of_change([1, 2], 2), "smaller"),
        (lambda: cumulative_vwap([1], [1], [1], [-1]), "negative"),
    ],
)
def test_indicator_guards(call, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()
