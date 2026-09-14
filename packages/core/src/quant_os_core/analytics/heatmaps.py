"""Heatmap builders for market, strategy, factor, and risk analytics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from quant_os_core.analytics._validation import finite_1d, finite_2d, labels_tuple
from quant_os_core.analytics.models import HeatmapData


def matrix_heatmap(
    values: ArrayLike,
    *,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    title: str,
    metric: str,
) -> HeatmapData:
    matrix = finite_2d(values, name="values")
    xs = labels_tuple(x_labels, expected=matrix.shape[1], name="x_labels")
    ys = labels_tuple(y_labels, expected=matrix.shape[0], name="y_labels")
    if not title.strip() or not metric.strip():
        raise ValueError("title and metric must be non-empty")
    return HeatmapData(
        title=title.strip(),
        metric=metric.strip(),
        x_labels=xs,
        y_labels=ys,
        values=tuple(tuple(float(value) for value in row) for row in matrix),
        value_min=float(np.min(matrix)),
        value_max=float(np.max(matrix)),
    )


def correlation_heatmap(
    observations: ArrayLike,
    labels: Sequence[str],
    *,
    title: str = "Correlation",
) -> HeatmapData:
    """Create a correlation heatmap from rows=observations, columns=series."""
    data = finite_2d(observations, name="observations", min_rows=2)
    names = labels_tuple(labels, expected=data.shape[1], name="labels")
    if data.shape[1] == 1:
        corr = np.asarray([[1.0]], dtype=np.float64)
    else:
        standard_deviation = np.std(data, axis=0)
        if np.any(standard_deviation <= np.finfo(np.float64).eps):
            raise ValueError("correlation requires non-constant series")
        corr = np.corrcoef(data, rowvar=False)
    corr = np.clip(np.asarray(corr, dtype=np.float64), -1.0, 1.0)
    return matrix_heatmap(
        corr,
        x_labels=names,
        y_labels=names,
        title=title,
        metric="correlation",
    )


def strategy_correlation_heatmap(
    strategy_returns: ArrayLike,
    strategy_names: Sequence[str],
) -> HeatmapData:
    return correlation_heatmap(
        strategy_returns,
        strategy_names,
        title="Strategy correlation",
    )


def factor_exposure_heatmap(
    exposures: ArrayLike,
    *,
    asset_names: Sequence[str],
    factor_names: Sequence[str],
) -> HeatmapData:
    return matrix_heatmap(
        exposures,
        x_labels=factor_names,
        y_labels=asset_names,
        title="Factor exposure",
        metric="exposure",
    )


def risk_contribution_heatmap(
    contributions: ArrayLike,
    asset_names: Sequence[str],
) -> HeatmapData:
    values = finite_1d(contributions, name="contributions")
    names = labels_tuple(asset_names, expected=values.size, name="asset_names")
    return matrix_heatmap(
        values.reshape(1, -1),
        x_labels=names,
        y_labels=("Portfolio",),
        title="Portfolio risk contribution",
        metric="risk_contribution",
    )
