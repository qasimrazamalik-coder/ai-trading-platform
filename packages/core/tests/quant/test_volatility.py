import numpy as np
import pytest

from quant_os_core.quant.volatility import (
    annualized_volatility,
    ewma_variance,
    ewma_volatility,
    rolling_volatility,
)


def test_annualized_volatility_matches_definition() -> None:
    returns = [0.01, -0.01, 0.02, -0.02]
    expected = np.std(returns, ddof=1) * np.sqrt(252)
    assert annualized_volatility(returns) == pytest.approx(expected)


def test_rolling_volatility_length_and_values() -> None:
    result = rolling_volatility([0.01, 0.02, -0.01, 0.03], 3, periods_per_year=12)
    assert result.shape == (2,)
    assert result[0] == pytest.approx(np.std([0.01, 0.02, -0.01], ddof=1) * np.sqrt(12))


def test_ewma_variance_recurrence() -> None:
    returns = [0.10, 0.20, -0.10]
    result = ewma_variance(returns, decay=0.5, initial_variance=0.04)
    np.testing.assert_allclose(result, [0.04, 0.025, 0.0325])


def test_ewma_volatility_is_nonnegative() -> None:
    result = ewma_volatility([0.01, 0.02, -0.01], decay=0.94)
    assert np.all(result >= 0)


def test_invalid_decay_rejected() -> None:
    with pytest.raises(ValueError):
        ewma_variance([0.01, 0.02], decay=1.0)
