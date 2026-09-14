import numpy as np
import pytest

from quant_os_core.analytics.distributions import (
    distribution_data,
    drawdown_distribution,
    monte_carlo_terminal_distribution,
)
from quant_os_core.analytics.surfaces import parameter_stability_surface, walk_forward_heatmap


def test_parameter_stability_surface_identifies_best_coordinates() -> None:
    values = [[1.0, 1.1, 1.05], [1.1, 1.2, 1.15], [1.05, 1.15, 1.1]]
    result = parameter_stability_surface(
        values,
        x_values=[10, 20, 30],
        y_values=[1, 2, 3],
        x_name="lookback",
        y_name="stop",
        metric="sharpe",
    )
    assert result.best_x == 20.0
    assert result.best_y == 2.0
    assert result.best_value == pytest.approx(1.2)
    assert 0.0 <= result.stability_score <= 1.0


def test_parameter_stability_shape_guard() -> None:
    with pytest.raises(ValueError, match="surface shape"):
        parameter_stability_surface(
            [[1.0, 2.0], [3.0, 4.0]],
            x_values=[1.0, 2.0, 3.0],
            y_values=[1.0, 2.0],
            x_name="x",
            y_name="y",
        )


def test_walk_forward_heatmap_contract() -> None:
    result = walk_forward_heatmap(
        [[1.0, 0.5], [0.7, 0.9]],
        fold_labels=["2024", "2025"],
        model_labels=["trend", "meanrev"],
        metric="sharpe",
    )
    assert result.x_labels == ("trend", "meanrev")
    assert result.y_labels == ("2024", "2025")


def test_distribution_data_summary_and_histogram() -> None:
    result = distribution_data(
        np.arange(1.0, 101.0), title="Returns", metric="return", bins=10
    )
    assert sum(result.counts) == 100
    assert result.mean == pytest.approx(50.5)
    assert len(result.bin_edges) == 11
    assert result.lower_quantile < result.median < result.upper_quantile


def test_monte_carlo_and_drawdown_distribution() -> None:
    mc = monte_carlo_terminal_distribution([-0.1, 0.0, 0.2, 0.3], bins=2)
    dd = drawdown_distribution([-0.1, -0.2, -0.05, -0.4], bins=2)
    assert mc.metric == "terminal_return"
    assert dd.metric == "drawdown"
    with pytest.raises(ValueError, match="non-positive"):
        drawdown_distribution([-0.1, 0.01])


def test_distribution_guards() -> None:
    with pytest.raises(ValueError):
        distribution_data([1.0], title="T", metric="M")
    with pytest.raises(ValueError):
        distribution_data([1.0, 2.0], title="T", metric="M", bins=1)
    with pytest.raises(ValueError):
        distribution_data([1.0, 2.0], title="T", metric="M", quantile=0.5)
