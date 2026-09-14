import numpy as np
import pytest

from quant_os_core.ml.classical import (
    catboost_regressor,
    lightgbm_regressor,
    xgboost_regressor,
)


@pytest.mark.parametrize(
    "factory,package",
    [
        (xgboost_regressor, "xgboost"),
        (lightgbm_regressor, "lightgbm"),
        (catboost_regressor, "catboost"),
    ],
)
def test_optional_booster_smoke(factory, package):
    pytest.importorskip(package)
    rng = np.random.default_rng(15)
    x = rng.normal(size=(60, 3))
    y = x[:, 0] - 0.3 * x[:, 1]
    model = factory().fit(x[:50], y[:50])
    prediction = model.predict(x[50:])
    assert prediction.shape == (10,)
    assert np.isfinite(prediction).all()
