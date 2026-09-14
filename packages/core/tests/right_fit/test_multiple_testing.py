import numpy as np
import pytest

from quant_os_core.right_fit import (
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
    sidak_adjusted_pvalue,
)


def _positive_returns() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(0.002, 0.01, size=500)


def test_probabilistic_sharpe_is_probability() -> None:
    probability = probabilistic_sharpe_ratio(_positive_returns())
    assert 0.5 < probability <= 1.0


def test_harder_sharpe_benchmark_reduces_probability() -> None:
    returns = _positive_returns()
    easy = probabilistic_sharpe_ratio(returns, benchmark_sharpe=0.0)
    hard = probabilistic_sharpe_ratio(returns, benchmark_sharpe=0.20)
    assert hard < easy


def test_expected_maximum_sharpe_grows_with_trial_dispersion() -> None:
    narrow = expected_maximum_sharpe([0.09, 0.10, 0.11, 0.10])
    wide = expected_maximum_sharpe([-0.2, 0.0, 0.2, 0.4])
    assert wide > narrow >= 0.0


def test_deflated_sharpe_returns_metadata() -> None:
    result = deflated_sharpe_ratio(
        _positive_returns(),
        trial_sharpes=[-0.05, 0.02, 0.08, 0.11, 0.15],
    )
    assert 0.0 <= result.probability <= 1.0
    assert result.trial_count == 5
    assert result.benchmark_sharpe >= 0.0


def test_sidak_adjustment_increases_raw_pvalue() -> None:
    adjusted = sidak_adjusted_pvalue(0.01, n_tests=20)
    assert adjusted > 0.01
    assert adjusted < 1.0


def test_probabilistic_sharpe_rejects_zero_variance() -> None:
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio([0.01, 0.01, 0.01])
