import numpy as np
import pytest

from quant_os_core.right_fit import probability_backtest_overfitting


def test_pbo_is_bounded_and_reports_expected_split_count() -> None:
    rng = np.random.default_rng(11)
    matrix = rng.normal(0.0, 0.01, size=(160, 6))
    result = probability_backtest_overfitting(matrix, n_slices=8)
    assert 0.0 <= result.probability <= 1.0
    assert result.split_count == 70
    assert result.logits.shape == (70,)
    assert result.selected_strategy_indices.shape == (70,)


def test_stable_dominant_strategy_has_lower_pbo_than_regime_chaser() -> None:
    rng = np.random.default_rng(12)
    stable = rng.normal(0.002, 0.01, size=240)
    alternatives = rng.normal(0.0, 0.01, size=(240, 4))
    stable_matrix = np.column_stack([stable, alternatives])
    stable_pbo = probability_backtest_overfitting(stable_matrix, n_slices=6).probability

    regime = rng.normal(0.0, 0.01, size=(240, 5))
    for block, strategy in enumerate([0, 1, 2, 3, 4, 0]):
        start = block * 40
        regime[start : start + 40, strategy] += 0.004
    regime_pbo = probability_backtest_overfitting(regime, n_slices=6).probability
    assert stable_pbo < regime_pbo


def test_pbo_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        probability_backtest_overfitting([0.1, 0.2, 0.3])
