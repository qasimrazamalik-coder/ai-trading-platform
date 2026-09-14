import numpy as np
import pytest

from quant_os_core.quant.statistics import (
    arithmetic_mean,
    autocorrelation,
    correlation,
    covariance,
    rolling_mean,
    rolling_std,
    sample_std,
    sample_variance,
    zscore,
)


def test_descriptive_statistics_match_numpy() -> None:
    values = [1, 2, 3, 4, 5]
    assert arithmetic_mean(values) == 3.0
    assert sample_variance(values) == pytest.approx(np.var(values, ddof=1))
    assert sample_std(values) == pytest.approx(np.std(values, ddof=1))


def test_covariance_and_correlation() -> None:
    x = [1, 2, 3, 4]
    y = [2, 4, 6, 8]
    assert covariance(x, y) == pytest.approx(np.cov(x, y, ddof=1)[0, 1])
    assert correlation(x, y) == pytest.approx(1.0)


def test_constant_correlation_rejected() -> None:
    with pytest.raises(ValueError):
        correlation([1, 1, 1], [1, 2, 3])


def test_zscore_is_centered_and_scaled() -> None:
    scores = zscore([1, 2, 3, 4, 5])
    assert scores.mean() == pytest.approx(0.0, abs=1e-12)
    assert scores.std(ddof=0) == pytest.approx(1.0)


def test_rolling_statistics() -> None:
    np.testing.assert_allclose(rolling_mean([1, 2, 3, 4], 2), [1.5, 2.5, 3.5])
    expected = [np.std([1, 2, 3], ddof=1), np.std([2, 3, 4], ddof=1)]
    np.testing.assert_allclose(rolling_std([1, 2, 3, 4], 3), expected)


def test_autocorrelation_positive_for_trend() -> None:
    assert autocorrelation([1, 2, 3, 4, 5], lag=1) == pytest.approx(1.0)


def test_nan_rejected() -> None:
    with pytest.raises(ValueError):
        arithmetic_mean([1.0, float("nan")])
