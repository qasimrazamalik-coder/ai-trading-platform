import numpy as np
import pytest

from quant_os_core.regimes import cusum_change_points


def test_cusum_detects_large_positive_shift() -> None:
    series = np.concatenate([np.zeros(60), np.full(60, 4.0)])
    points = cusum_change_points(series, threshold=4.0, drift=0.1)
    assert points
    assert any(point.direction == 1 for point in points)


def test_cusum_constant_series_has_no_points() -> None:
    assert cusum_change_points(np.ones(40)) == ()


def test_cusum_guards() -> None:
    with pytest.raises(ValueError):
        cusum_change_points([1.0, 2.0])
    with pytest.raises(ValueError):
        cusum_change_points([1.0, 2.0, 3.0], threshold=0.0)
