"""Parameter and walk-forward surface payloads."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from quant_os_core.analytics._validation import finite_1d, finite_2d, labels_tuple
from quant_os_core.analytics.models import HeatmapData, SurfaceData
from quant_os_core.right_fit.stability import parameter_surface_stability


def parameter_stability_surface(
    values: ArrayLike,
    *,
    x_values: ArrayLike,
    y_values: ArrayLike,
    x_name: str,
    y_name: str,
    metric: str = "score",
    title: str = "Parameter stability",
) -> SurfaceData:
    surface = finite_2d(values, name="values", min_rows=2, min_cols=2)
    xs = finite_1d(x_values, name="x_values", min_size=2)
    ys = finite_1d(y_values, name="y_values", min_size=2)
    if surface.shape != (ys.size, xs.size):
        raise ValueError("surface shape must equal (len(y_values), len(x_values))")
    if not all(value.strip() for value in (x_name, y_name, metric, title)):
        raise ValueError("axis names, metric and title must be non-empty")
    stability = parameter_surface_stability(surface)
    row, col = stability.best_index
    return SurfaceData(
        title=title.strip(),
        metric=metric.strip(),
        x_name=x_name.strip(),
        y_name=y_name.strip(),
        x_values=tuple(float(value) for value in xs),
        y_values=tuple(float(value) for value in ys),
        values=tuple(tuple(float(value) for value in row_values) for row_values in surface),
        best_x=float(xs[col]),
        best_y=float(ys[row]),
        best_value=stability.best_value,
        stability_score=stability.score,
        plateau_fraction=stability.plateau_fraction,
    )


def walk_forward_heatmap(
    values: ArrayLike,
    *,
    fold_labels: Sequence[str],
    model_labels: Sequence[str],
    metric: str = "score",
) -> HeatmapData:
    surface = finite_2d(values, name="values")
    folds = labels_tuple(fold_labels, expected=surface.shape[0], name="fold_labels")
    models = labels_tuple(model_labels, expected=surface.shape[1], name="model_labels")
    from quant_os_core.analytics.heatmaps import matrix_heatmap

    return matrix_heatmap(
        surface,
        x_labels=models,
        y_labels=folds,
        title="Walk-forward performance",
        metric=metric,
    )
