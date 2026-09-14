"""High-level transparent regime snapshot engine."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from quant_os_core.regimes.base import RegimeSnapshot
from quant_os_core.regimes.classification import (
    classify_correlation,
    classify_liquidity,
    classify_macro,
    classify_risk,
    classify_trend,
    classify_volatility,
)


def regime_snapshot(
    *,
    prices: npt.ArrayLike,
    returns: npt.ArrayLike,
    growth_asset_returns: npt.ArrayLike,
    defensive_asset_returns: npt.ArrayLike,
    correlation_matrix: npt.ArrayLike | None = None,
    spread_bps: float | None = None,
    depth_zscore: float | None = None,
    volume_zscore: float | None = None,
    growth_zscore: float | None = None,
    inflation_zscore: float | None = None,
) -> RegimeSnapshot:
    trend = classify_trend(prices)
    volatility = classify_volatility(returns)
    risk = classify_risk(growth_asset_returns, defensive_asset_returns)

    liquidity = None
    liquidity_inputs = (spread_bps, depth_zscore, volume_zscore)
    if any(value is not None for value in liquidity_inputs):
        if any(value is None for value in liquidity_inputs):
            raise ValueError("liquidity inputs must be supplied together")
        liquidity = classify_liquidity(
            spread_bps=float(spread_bps),
            depth_zscore=float(depth_zscore),
            volume_zscore=float(volume_zscore),
        )

    correlation = None
    if correlation_matrix is not None:
        correlation = classify_correlation(correlation_matrix)

    macro = None
    if growth_zscore is not None or inflation_zscore is not None:
        if growth_zscore is None or inflation_zscore is None:
            raise ValueError("growth_zscore and inflation_zscore must be supplied together")
        macro = classify_macro(
            growth_zscore=float(growth_zscore),
            inflation_zscore=float(inflation_zscore),
        )

    return RegimeSnapshot(
        trend=trend,
        volatility=volatility,
        risk=risk,
        liquidity=liquidity,
        correlation=correlation,
        macro=macro,
    )
