"""Typed visualization contracts shared by APIs, notebooks, and frontends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HeatmapData:
    """Rectangular numeric matrix with explicit axis labels."""

    title: str
    metric: str
    x_labels: tuple[str, ...]
    y_labels: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    value_min: float
    value_max: float


@dataclass(frozen=True, slots=True)
class TimeBucketData:
    """Aggregated metric by named calendar/time bucket."""

    title: str
    metric: str
    labels: tuple[str, ...]
    values: tuple[float, ...]
    counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CalendarCell:
    """Single calendar-day metric value."""

    date: str
    value: float
    count: int


@dataclass(frozen=True, slots=True)
class DistributionData:
    """Histogram plus robust summary statistics."""

    title: str
    metric: str
    bin_edges: tuple[float, ...]
    counts: tuple[int, ...]
    mean: float
    median: float
    lower_quantile: float
    upper_quantile: float


@dataclass(frozen=True, slots=True)
class SurfaceData:
    """Two-dimensional parameter/result surface with stability metadata."""

    title: str
    metric: str
    x_name: str
    y_name: str
    x_values: tuple[float, ...]
    y_values: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    best_x: float
    best_y: float
    best_value: float
    stability_score: float
    plateau_fraction: float
