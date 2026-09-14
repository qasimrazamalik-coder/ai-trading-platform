import numpy as np
import pytest

from quant_os_core.quant.risk import gaussian_cvar, gaussian_var, historical_cvar, historical_var


def test_historical_var_and_cvar_positive_loss_convention() -> None:
    returns = [-0.10, -0.05, 0.0, 0.01, 0.02, 0.03]
    var = historical_var(returns, confidence=0.80)
    cvar = historical_cvar(returns, confidence=0.80)
    assert var >= 0
    assert cvar >= var


def test_gaussian_cvar_exceeds_var_for_normal_tail() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.02, size=1000)
    assert gaussian_cvar(returns, confidence=0.95) > gaussian_var(returns, confidence=0.95)


def test_invalid_confidence_rejected() -> None:
    with pytest.raises(ValueError):
        historical_var([-0.1, 0.1], confidence=1.0)
