import pytest

torch = pytest.importorskip("torch")

from quant_os_core.ml.deep import (  # noqa: E402
    GRUForecaster,
    LSTMForecaster,
    TemporalConvForecaster,
    TransformerForecaster,
)


@pytest.mark.parametrize(
    "model",
    [
        LSTMForecaster(4, hidden_size=8),
        GRUForecaster(4, hidden_size=8),
        TemporalConvForecaster(4, channels=8),
        TransformerForecaster(4, model_dim=8, heads=2, layers=1),
    ],
)
def test_sequence_model_shapes_and_finite(model):
    x = torch.randn(5, 12, 4)
    y = model(x)
    assert y.shape == (5,)
    assert torch.isfinite(y).all()


def test_transformer_requires_divisible_heads():
    with pytest.raises(ValueError):
        TransformerForecaster(4, model_dim=10, heads=4)
