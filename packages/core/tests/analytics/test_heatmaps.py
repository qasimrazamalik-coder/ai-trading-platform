import numpy as np
import pytest

from quant_os_core.analytics.heatmaps import (
    correlation_heatmap,
    factor_exposure_heatmap,
    matrix_heatmap,
    risk_contribution_heatmap,
    strategy_correlation_heatmap,
)


def test_matrix_heatmap_contract() -> None:
    result = matrix_heatmap(
        [[1.0, 2.0], [3.0, 4.0]],
        x_labels=["A", "B"],
        y_labels=["X", "Y"],
        title="Demo",
        metric="score",
    )
    assert result.values == ((1.0, 2.0), (3.0, 4.0))
    assert result.value_min == 1.0
    assert result.value_max == 4.0


def test_matrix_heatmap_rejects_duplicate_or_bad_labels() -> None:
    with pytest.raises(ValueError):
        matrix_heatmap([[1.0, 2.0]], x_labels=["A", "A"], y_labels=["X"], title="T", metric="m")
    with pytest.raises(ValueError):
        matrix_heatmap([[1.0, 2.0]], x_labels=["A"], y_labels=["X"], title="T", metric="m")


def test_correlation_heatmap_matches_numpy() -> None:
    observations = np.asarray(
        [[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]],
        dtype=float,
    )
    result = correlation_heatmap(observations, ["A", "B"])
    assert result.values[0][0] == pytest.approx(1.0)
    assert result.values[0][1] == pytest.approx(-1.0)
    assert result.value_min == pytest.approx(-1.0)


def test_single_series_correlation_is_one() -> None:
    result = correlation_heatmap([[1.0], [2.0], [3.0]], ["Only"])
    assert result.values == ((1.0,),)


def test_correlation_rejects_constant_series() -> None:
    with pytest.raises(ValueError, match="non-constant"):
        correlation_heatmap([[1.0, 2.0], [1.0, 3.0]], ["A", "B"])


def test_specialized_heatmaps() -> None:
    strategies = strategy_correlation_heatmap(
        [[0.1, 0.2], [0.2, 0.1], [0.3, 0.0]], ["trend", "meanrev"]
    )
    factors = factor_exposure_heatmap(
        [[0.2, -0.1], [0.4, 0.3]],
        asset_names=["BTC", "ETH"],
        factor_names=["momentum", "value"],
    )
    risk = risk_contribution_heatmap([0.6, 0.4], ["BTC", "ETH"])
    assert strategies.title == "Strategy correlation"
    assert factors.y_labels == ("BTC", "ETH")
    assert risk.values == ((0.6, 0.4),)
