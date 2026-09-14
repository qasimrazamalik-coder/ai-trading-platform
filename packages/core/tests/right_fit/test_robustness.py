import numpy as np
import pytest

from quant_os_core.right_fit import monte_carlo_robustness, moving_block_bootstrap


def test_moving_block_bootstrap_is_deterministic_with_seed() -> None:
    returns = np.linspace(-0.01, 0.02, 30)
    first = moving_block_bootstrap(returns, n_resamples=10, block_size=4, seed=5)
    second = moving_block_bootstrap(returns, n_resamples=10, block_size=4, seed=5)
    assert np.array_equal(first, second)
    assert first.shape == (10, 30)


def test_bootstrap_values_come_from_original_series() -> None:
    returns = np.array([-0.01, 0.0, 0.01, 0.02, 0.03])
    samples = moving_block_bootstrap(returns, n_resamples=5, block_size=2, seed=3)
    assert set(np.unique(samples)).issubset(set(returns))


def test_monte_carlo_positive_strategy_reports_high_positive_probability() -> None:
    rng = np.random.default_rng(99)
    returns = rng.normal(0.002, 0.005, size=150)
    result = monte_carlo_robustness(returns, n_resamples=200, block_size=5, seed=8)
    assert result.positive_return_probability > 0.9
    assert result.samples == 200
    assert result.cumulative_return_p05 <= result.median_cumulative_return


def test_block_size_validation() -> None:
    with pytest.raises(ValueError):
        moving_block_bootstrap([0.1, 0.2], block_size=3)
