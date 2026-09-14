import numpy as np
import pytest

from quant_os_core.ml.calibration import PlattCalibrator
from quant_os_core.ml.ensemble import weighted_predictions


def test_weighted_predictions():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = weighted_predictions(matrix, [0.75, 0.25])
    assert np.allclose(out, [1.5, 2.5])


def test_weight_validation():
    with pytest.raises(ValueError):
        weighted_predictions([[1, 2]], [0])
    with pytest.raises(ValueError):
        weighted_predictions([[1, 2]], [-1])


def test_platt_calibrator_outputs_probabilities():
    scores = np.array([-3, -2, -1, -0.5, 0.5, 1, 2, 3], dtype=float)
    targets = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    calibrator = PlattCalibrator().fit(scores, targets)
    p = calibrator.predict_proba(np.array([-2.0, 0.0, 2.0]))
    assert np.all((p >= 0) & (p <= 1))
    assert p[0] < p[1] < p[2]
