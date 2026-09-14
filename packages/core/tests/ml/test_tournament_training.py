import numpy as np
import pytest

from quant_os_core.ml.classical import logistic_classifier, ridge_regressor
from quant_os_core.ml.data import lagged_return_dataset
from quant_os_core.ml.tournament import ModelCandidate, walk_forward_tournament


def test_logistic_classifier_probabilities():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(100, 2))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    model = logistic_classifier().fit(x[:80], y[:80])
    p = model.predict_proba(x[80:])
    assert p.shape == (20,)
    assert np.all((p >= 0) & (p <= 1))


def test_walk_forward_tournament_prefers_predictive_ridge():
    rng = np.random.default_rng(22)
    returns = 0.003 * np.sin(np.arange(240) / 7) + rng.normal(0, 0.0002, 240)
    prices = 100 * np.exp(np.cumsum(returns))
    data = lagged_return_dataset(prices, lags=8)

    class ZeroModel:
        def fit(self, x, y):
            return self
        def predict(self, x):
            return np.zeros(len(x))

    results = walk_forward_tournament(
        data,
        (
            ModelCandidate("ridge", lambda: ridge_regressor(alpha=0.001)),
            ModelCandidate("zero", ZeroModel, standardize=False),
        ),
        train_size=100,
        test_size=25,
        gap=1,
    )
    assert results[0].model_id == "ridge"
    assert len(results[0].fold_mse) >= 4


def test_duplicate_tournament_ids_rejected():
    prices = np.linspace(100, 150, 80)
    data = lagged_return_dataset(prices, lags=5)
    with pytest.raises(ValueError):
        walk_forward_tournament(
            data,
            (
                ModelCandidate("x", lambda: ridge_regressor()),
                ModelCandidate("x", lambda: ridge_regressor()),
            ),
            train_size=30,
            test_size=10,
        )


def test_torch_training_early_stopping_and_patch_models():
    torch = pytest.importorskip("torch")
    from quant_os_core.ml.deep import ITransformerForecaster, PatchTSTForecaster
    from quant_os_core.ml.training import train_torch_regressor

    torch.manual_seed(2)
    x = torch.randn(36, 8, 2)
    y = (x[:, -1, 0] * 0.3).numpy()
    model = PatchTSTForecaster(2, patch_size=2, model_dim=8, heads=2, layers=1)
    result = train_torch_regressor(
        model,
        x[:28].numpy(),
        y[:28],
        x[28:].numpy(),
        y[28:],
        epochs=15,
        patience=4,
        learning_rate=0.01,
    )
    assert 1 <= result.epochs_run <= 15
    assert np.isfinite(result.best_validation_loss)
    assert model(x[28:]).shape == (8,)
    inv = ITransformerForecaster(2, sequence_length=8, model_dim=8, heads=2, layers=1)
    assert inv(x).shape == (36,)
