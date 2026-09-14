import numpy as np
import pytest

from quant_os_core.ml.data import Standardizer, chronological_split, lagged_return_dataset


def test_lagged_dataset_uses_past_features_and_future_target():
    prices = np.exp(np.arange(10, dtype=float) * 0.01)
    data = lagged_return_dataset(prices, lags=3, horizon=2)
    assert data.features.shape[1] == 3
    assert np.allclose(data.features[0], 0.01)
    assert data.targets[0] == pytest.approx(0.02)


def test_lagged_dataset_preserves_integer_timestamps():
    prices = np.linspace(100, 120, 12)
    ts = np.arange(12, dtype=np.int64) * 1_000_000_001
    data = lagged_return_dataset(prices, lags=3, timestamps=ts)
    assert data.timestamps is not None
    assert data.timestamps.dtype == np.int64
    assert data.timestamps[0] == ts[3]


def test_chronological_split_order_and_sizes():
    prices = np.linspace(100, 160, 40)
    data = lagged_return_dataset(prices, lags=3)
    train, valid, test = chronological_split(data)
    assert train.targets.size > valid.targets.size
    assert test.targets.size > 0
    assert np.allclose(train.features[-1], data.features[train.targets.size - 1])


def test_standardizer_fit_train_only_behavior():
    train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = Standardizer.fit(train)
    transformed = scaler.transform(train)
    assert np.allclose(transformed.mean(axis=0), 0.0)
    shifted = scaler.transform(np.array([[100.0, 200.0]]))
    assert shifted[0, 0] > 10


def test_invalid_dataset_inputs_raise():
    with pytest.raises(ValueError):
        lagged_return_dataset([1.0, 2.0, 3.0], lags=4)
    with pytest.raises(ValueError):
        Standardizer.fit([[1.0, np.nan], [2.0, 3.0]])
