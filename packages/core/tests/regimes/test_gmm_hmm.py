import numpy as np
import pytest

from quant_os_core.regimes import DiagonalGaussianMixture, GaussianHMMRegimeDetector


def _clustered_data(seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low = rng.normal([-2.0, 0.3], [0.15, 0.1], size=(80, 2))
    high = rng.normal([2.0, 1.5], [0.15, 0.1], size=(80, 2))
    return np.vstack([low, high])


def test_gmm_fits_and_separates_clusters() -> None:
    data = _clustered_data()
    model = DiagonalGaussianMixture(n_components=2, max_iter=200)
    result = model.fit(data)
    probabilities = model.predict_proba(data)
    states = model.predict(data)
    assert result.weights.sum() == pytest.approx(1.0)
    assert probabilities.shape == (160, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert states[:80].mean() != pytest.approx(states[80:].mean(), abs=0.25)


def test_gmm_prediction_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        DiagonalGaussianMixture(2).predict(np.ones((10, 2)))


def test_gmm_input_guards() -> None:
    with pytest.raises(ValueError):
        DiagonalGaussianMixture(1)
    with pytest.raises(ValueError):
        DiagonalGaussianMixture(2).fit(np.ones((3, 2)))


def test_hmm_fits_persistent_states() -> None:
    rng = np.random.default_rng(8)
    first = rng.normal([-1.5, 0.2], [0.12, 0.06], size=(80, 2))
    second = rng.normal([1.5, 1.2], [0.12, 0.06], size=(80, 2))
    data = np.vstack([first, second])
    model = GaussianHMMRegimeDetector(n_states=2, max_iter=100)
    fit = model.fit(data)
    states = model.predict(data)
    probabilities = model.predict_proba(data)
    assert fit.transition_matrix.shape == (2, 2)
    assert np.allclose(fit.transition_matrix.sum(axis=1), 1.0)
    assert probabilities.shape == (160, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert states[:70].mean() != pytest.approx(states[-70:].mean(), abs=0.25)


def test_hmm_prediction_requires_fit_and_guards() -> None:
    model = GaussianHMMRegimeDetector(2)
    with pytest.raises(RuntimeError):
        model.predict(np.ones((10, 2)))
    with pytest.raises(ValueError):
        GaussianHMMRegimeDetector(1)
    with pytest.raises(ValueError):
        model.fit(np.ones((3, 2)))


def test_gmm_can_predict_single_live_observation_after_fit() -> None:
    model = DiagonalGaussianMixture(2).fit(_clustered_data())
    detector = DiagonalGaussianMixture(2)
    detector.fit(_clustered_data())
    probabilities = detector.predict_proba(np.array([[-2.0, 0.3]]))
    assert probabilities.shape == (1, 2)
    assert model.weights.shape == (2,)


def test_hmm_can_predict_single_live_observation_after_fit() -> None:
    rng = np.random.default_rng(9)
    data = np.vstack(
        [
            rng.normal([-1.0, 0.2], [0.1, 0.05], size=(30, 2)),
            rng.normal([1.0, 1.0], [0.1, 0.05], size=(30, 2)),
        ]
    )
    detector = GaussianHMMRegimeDetector(2)
    detector.fit(data)
    probabilities = detector.predict_proba(np.array([[1.0, 1.0]]))
    states = detector.predict(np.array([[1.0, 1.0]]))
    assert probabilities.shape == (1, 2)
    assert states.shape == (1,)
