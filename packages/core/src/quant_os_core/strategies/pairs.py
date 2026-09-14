"""Pairs-trading and statistical-arbitrage helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quant_os_core.quant._validation import paired_float_arrays
from quant_os_core.quant.regression import engle_granger, ordinary_least_squares
from quant_os_core.strategies.indicators import rolling_zscore


@dataclass(frozen=True, slots=True)
class PairsSignalResult:
    hedge_ratio: float
    intercept: float
    spread: npt.NDArray[np.float64]
    zscore: npt.NDArray[np.float64]
    long_y_short_x: npt.NDArray[np.float64]
    cointegration_pvalue: float


def pairs_zscore_signal(
    y_prices: npt.ArrayLike,
    x_prices: npt.ArrayLike,
    *,
    window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    require_cointegration: bool = True,
    max_cointegration_pvalue: float = 0.05,
) -> PairsSignalResult:
    """Fit Y = intercept + beta*X and trade normalized residual extremes."""
    y, x = paired_float_arrays(
        y_prices,
        x_prices,
        left_name="y_prices",
        right_name="x_prices",
        min_size=max(10, window + 1),
    )
    if np.any(y <= 0.0) or np.any(x <= 0.0):
        raise ValueError("pairs prices must be > 0")
    if not 0.0 <= max_cointegration_pvalue <= 1.0:
        raise ValueError("max_cointegration_pvalue must be between 0 and 1")
    if entry_z <= 0.0 or exit_z < 0.0 or exit_z >= entry_z:
        raise ValueError("require entry_z > exit_z >= 0")

    regression = ordinary_least_squares(y, x)
    spread = y - (regression.intercept + regression.slope * x)
    scores = rolling_zscore(spread, window)
    cointegration = engle_granger(y, x)
    if require_cointegration and cointegration.p_value > max_cointegration_pvalue:
        positions = np.zeros(y.size, dtype=np.float64)
    else:
        positions = np.zeros(y.size, dtype=np.float64)
        state = 0.0
        for index, score in enumerate(scores):
            if not np.isfinite(score):
                continue
            if state == 0.0:
                if score <= -entry_z:
                    state = 1.0
                elif score >= entry_z:
                    state = -1.0
            elif abs(score) <= exit_z:
                state = 0.0
            positions[index] = state

    return PairsSignalResult(
        hedge_ratio=float(regression.slope),
        intercept=float(regression.intercept),
        spread=spread,
        zscore=np.nan_to_num(scores),
        long_y_short_x=positions,
        cointegration_pvalue=float(cointegration.p_value),
    )
