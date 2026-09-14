import pytest

from quant_os_core.quant.sizing import (
    fixed_fractional_position_size,
    fractional_kelly,
    kelly_fraction,
    kelly_from_moments,
    volatility_target_weight,
)


def test_kelly_binary_formula() -> None:
    assert kelly_fraction(0.60, 1.0) == pytest.approx(0.20)
    assert fractional_kelly(0.60, 1.0, fraction=0.25) == pytest.approx(0.05)


def test_kelly_from_moments() -> None:
    assert kelly_from_moments(0.02, 0.04) == pytest.approx(0.5)


def test_vol_target_respects_leverage_cap() -> None:
    assert volatility_target_weight(0.10, target_volatility=0.20, max_leverage=1.5) == 1.5
    assert volatility_target_weight(0.20, target_volatility=0.10, max_leverage=1.5) == 0.5


def test_fixed_fractional_position_size() -> None:
    # Risk $100 on a $5 stop -> 20 units.
    assert fixed_fractional_position_size(10_000, 0.01, 100, 95) == pytest.approx(20.0)


def test_zero_stop_distance_rejected() -> None:
    with pytest.raises(ValueError):
        fixed_fractional_position_size(10_000, 0.01, 100, 100)
