import numpy as np

from quant_os_core.ml.classical import random_forest_regressor, ridge_regressor
from quant_os_core.ml.importance import permutation_importance
from quant_os_core.ml.validation import evaluate_regression_predictions


def test_ridge_learns_linear_relationship():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(120, 3))
    y = 2 * x[:, 0] - x[:, 1] + 0.01 * rng.normal(size=120)
    model = ridge_regressor(alpha=0.01).fit(x[:90], y[:90])
    pred = model.predict(x[90:])
    evaluation = evaluate_regression_predictions(y[90:], pred)
    assert evaluation.beats_baseline
    assert evaluation.information_coefficient > 0.95


def test_random_forest_and_permutation_importance():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(160, 2))
    y = x[:, 0] ** 2 + 0.01 * rng.normal(size=160)
    model = random_forest_regressor(n_estimators=60).fit(x[:120], y[:120])
    importance = permutation_importance(model, x[120:], y[120:])
    assert importance.shape == (2,)
    assert importance[0] > importance[1]
