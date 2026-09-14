import numpy as np
import pytest

from quant_os_core.ml.baselines import historical_mean_forecast, zero_return_forecast
from quant_os_core.ml.calibration import PlattCalibrator
from quant_os_core.ml.data import chronological_split, lagged_return_dataset
from quant_os_core.ml.ensemble import weighted_predictions
from quant_os_core.ml.experiments import JsonlExperimentTracker
from quant_os_core.ml.registry import ModelFamily, ModelRegistry
from quant_os_core.ml.tournament import ModelCandidate, walk_forward_tournament


def test_baseline_guards_and_historical_mean():
    assert np.allclose(historical_mean_forecast([1.0, 3.0], 2), [2.0, 2.0])
    with pytest.raises(ValueError):
        zero_return_forecast(0)
    with pytest.raises(ValueError):
        historical_mean_forecast([], 2)


def test_split_fraction_guards():
    data = lagged_return_dataset(np.linspace(100, 140, 30), lags=3)
    with pytest.raises(ValueError):
        chronological_split(data, train_fraction=0.9, validation_fraction=0.2)


def test_calibrator_guards():
    calibrator = PlattCalibrator()
    with pytest.raises(RuntimeError):
        calibrator.predict_proba([0.0])
    with pytest.raises(ValueError):
        calibrator.fit([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1])


def test_registry_and_tracker_guards(tmp_path):
    registry = ModelRegistry()
    with pytest.raises(ValueError):
        registry.register(" ", ModelFamily.LINEAR)
    with pytest.raises(KeyError):
        registry.latest("missing")
    tracker = JsonlExperimentTracker(tmp_path / "e.jsonl")
    with pytest.raises(ValueError):
        tracker.log("", "ridge", metrics={}, parameters={})


def test_ensemble_shape_guard():
    with pytest.raises(ValueError):
        weighted_predictions([[1.0, 2.0], [2.0, 3.0]], [1.0])


def test_tournament_requires_usable_model():
    prices = np.linspace(100, 150, 70)
    data = lagged_return_dataset(prices, lags=4)
    with pytest.raises(TypeError):
        walk_forward_tournament(
            data,
            (ModelCandidate("bad", object),),
            train_size=30,
            test_size=10,
        )


def test_deep_model_input_guards():
    torch = pytest.importorskip("torch")
    from quant_os_core.ml.deep import ITransformerForecaster, PatchTSTForecaster

    patch = PatchTSTForecaster(2, patch_size=4, model_dim=8, heads=2, layers=1)
    with pytest.raises(ValueError):
        patch(torch.randn(2, 3, 2))
    inv = ITransformerForecaster(2, sequence_length=8, model_dim=8, heads=2, layers=1)
    with pytest.raises(ValueError):
        inv(torch.randn(2, 7, 2))


def test_training_argument_guards():
    pytest.importorskip("torch")
    from quant_os_core.ml.deep import LSTMForecaster
    from quant_os_core.ml.training import train_torch_regressor

    model = LSTMForecaster(2, hidden_size=4)
    x = np.zeros((4, 3, 2), dtype=float)
    y = np.zeros(4, dtype=float)
    with pytest.raises(ValueError):
        train_torch_regressor(model, x, y, x, y, epochs=0)
    with pytest.raises(ValueError):
        train_torch_regressor(model, x, y[:3], x, y)
