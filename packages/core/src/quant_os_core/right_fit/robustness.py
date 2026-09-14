"""Bootstrap and Monte Carlo robustness diagnostics for strategy returns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quant_os_core.quant._validation import as_float_array
from quant_os_core.quant.returns import cumulative_return, max_drawdown

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MonteCarloRobustnessResult:
    median_cumulative_return: float
    cumulative_return_p05: float
    cumulative_return_p95: float
    median_max_drawdown: float
    drawdown_p05: float
    positive_return_probability: float
    samples: int


def moving_block_bootstrap(
    returns: ArrayLike,
    *,
    n_resamples: int = 1_000,
    block_size: int = 5,
    seed: int | None = None,
) -> FloatArray:
    """Resample contiguous return blocks while approximately preserving dependence."""
    values = as_float_array(returns, name="returns", min_size=2)
    resamples = int(n_resamples)
    block = int(block_size)
    if resamples != n_resamples or resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if block != block_size or block <= 0 or block > values.size:
        raise ValueError("block_size must be a positive integer <= return count")

    rng = np.random.default_rng(seed)
    result = np.empty((resamples, values.size), dtype=np.float64)
    max_start = values.size - block
    blocks_needed = int(np.ceil(values.size / block))
    for row in range(resamples):
        starts = rng.integers(0, max_start + 1, size=blocks_needed)
        sample = np.concatenate([values[start : start + block] for start in starts])
        result[row] = sample[: values.size]
    return result


def monte_carlo_robustness(
    returns: ArrayLike,
    *,
    n_resamples: int = 1_000,
    block_size: int = 5,
    seed: int | None = None,
) -> MonteCarloRobustnessResult:
    """Summarize return and drawdown distributions under block-bootstrap resampling."""
    bootstraps = moving_block_bootstrap(
        returns,
        n_resamples=n_resamples,
        block_size=block_size,
        seed=seed,
    )
    total_returns = np.asarray([cumulative_return(row) for row in bootstraps], dtype=np.float64)
    drawdowns = np.asarray([max_drawdown(row) for row in bootstraps], dtype=np.float64)
    return MonteCarloRobustnessResult(
        median_cumulative_return=float(np.median(total_returns)),
        cumulative_return_p05=float(np.quantile(total_returns, 0.05)),
        cumulative_return_p95=float(np.quantile(total_returns, 0.95)),
        median_max_drawdown=float(np.median(drawdowns)),
        drawdown_p05=float(np.quantile(drawdowns, 0.05)),
        positive_return_probability=float(np.mean(total_returns > 0.0)),
        samples=int(bootstraps.shape[0]),
    )
