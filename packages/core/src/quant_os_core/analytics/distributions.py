"""Distribution payloads for Monte Carlo, returns, and risk diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from quant_os_core.analytics._validation import finite_1d
from quant_os_core.analytics.models import DistributionData


def distribution_data(
    samples: ArrayLike,
    *,
    title: str,
    metric: str,
    bins: int = 30,
    quantile: float = 0.05,
) -> DistributionData:
    values = finite_1d(samples, name="samples", min_size=2)
    if not title.strip() or not metric.strip():
        raise ValueError("title and metric must be non-empty")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if not 0.0 < quantile < 0.5:
        raise ValueError("quantile must be in (0, 0.5)")
    counts, edges = np.histogram(values, bins=bins)
    return DistributionData(
        title=title.strip(),
        metric=metric.strip(),
        bin_edges=tuple(float(value) for value in edges),
        counts=tuple(int(value) for value in counts),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        lower_quantile=float(np.quantile(values, quantile)),
        upper_quantile=float(np.quantile(values, 1.0 - quantile)),
    )


def monte_carlo_terminal_distribution(samples: ArrayLike, *, bins: int = 30) -> DistributionData:
    return distribution_data(
        samples,
        title="Monte Carlo terminal return",
        metric="terminal_return",
        bins=bins,
    )


def drawdown_distribution(samples: ArrayLike, *, bins: int = 30) -> DistributionData:
    values = finite_1d(samples, name="samples", min_size=2)
    if np.any(values > 0.0):
        raise ValueError("drawdown samples must be non-positive")
    return distribution_data(
        values,
        title="Drawdown distribution",
        metric="drawdown",
        bins=bins,
    )
