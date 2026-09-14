import numpy as np

from quant_os_core.ml.validation import evaluate_regression_predictions


def test_model_evaluation_beats_zero_baseline():
    y = np.array([0.01, -0.02, 0.03, -0.04])
    good = y * 0.9
    result = evaluate_regression_predictions(y, good)
    assert result.beats_baseline
    assert result.mse_improvement > 0.9
    assert result.directional_accuracy == 1.0
