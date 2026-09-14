import numpy as np
import pytest

from quant_os_core.ml.metrics import (
    brier_score,
    directional_accuracy,
    expected_calibration_error,
    information_coefficient,
    mean_absolute_error,
    mean_squared_error,
)


def test_regression_metrics_known_values():
    y = np.array([1.0, -1.0, 2.0])
    p = np.array([0.5, -1.5, 2.0])
    assert mean_squared_error(y, p) == pytest.approx(1 / 6)
    assert mean_absolute_error(y, p) == pytest.approx(1 / 3)
    assert directional_accuracy(y, p) == 1.0
    assert information_coefficient(y, y) == pytest.approx(1.0)


def test_information_coefficient_constant_prediction_is_zero():
    assert information_coefficient([1, 2, 3], [0, 0, 0]) == 0.0


def test_brier_and_calibration_error():
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    assert brier_score(y, p) == pytest.approx(0.025)
    assert 0 <= expected_calibration_error(y, p, bins=2) <= 1


def test_invalid_probabilities_raise():
    with pytest.raises(ValueError):
        brier_score([0, 1], [0.2, 1.2])
    with pytest.raises(ValueError):
        expected_calibration_error([0, 2], [0.2, 0.8])
