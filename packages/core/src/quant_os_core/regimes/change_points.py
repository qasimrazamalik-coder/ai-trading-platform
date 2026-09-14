"""Lightweight change-point detection for regime shifts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class ChangePoint:
    index: int
    direction: int
    statistic: float


def cusum_change_points(
    values: npt.ArrayLike,
    *,
    threshold: float = 5.0,
    drift: float = 0.25,
    min_distance: int = 5,
) -> tuple[ChangePoint, ...]:
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or series.size < 3:
        raise ValueError("values must contain at least three observations")
    if not np.isfinite(series).all():
        raise ValueError("values must be finite")
    if threshold <= 0.0 or drift < 0.0 or min_distance < 1:
        raise ValueError("invalid CUSUM parameters")

    scale = float(np.std(series, ddof=1))
    if scale <= 1e-15:
        return ()
    z = (series - float(np.mean(series))) / scale
    positive = 0.0
    negative = 0.0
    last = -min_distance
    detected: list[ChangePoint] = []
    for index, observation in enumerate(z):
        positive = max(0.0, positive + float(observation) - drift)
        negative = min(0.0, negative + float(observation) + drift)
        if positive >= threshold and index - last >= min_distance:
            detected.append(ChangePoint(index=index, direction=1, statistic=positive))
            positive = 0.0
            negative = 0.0
            last = index
        elif negative <= -threshold and index - last >= min_distance:
            detected.append(ChangePoint(index=index, direction=-1, statistic=-negative))
            positive = 0.0
            negative = 0.0
            last = index
    return tuple(detected)
