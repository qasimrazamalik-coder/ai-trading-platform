"""Market-making quote mathematics for research and simulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np


@dataclass(frozen=True, slots=True)
class MarketMakingQuote:
    reservation_price: float
    bid: float
    ask: float
    half_spread: float


def avellaneda_stoikov_quote(
    *,
    mid_price: float,
    inventory: float,
    volatility: float,
    risk_aversion: float,
    time_remaining: float,
    liquidity_k: float,
) -> MarketMakingQuote:
    """Compute a simplified Avellaneda-Stoikov reservation price and symmetric quotes."""
    values = np.asarray(
        [mid_price, inventory, volatility, risk_aversion, time_remaining, liquidity_k],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("market-making inputs must be finite")
    if mid_price <= 0.0 or volatility < 0.0 or risk_aversion <= 0.0:
        raise ValueError("mid_price/risk_aversion must be > 0 and volatility >= 0")
    if time_remaining < 0.0 or liquidity_k <= 0.0:
        raise ValueError("time_remaining must be >= 0 and liquidity_k > 0")
    variance = volatility**2
    reservation = mid_price - inventory * risk_aversion * variance * time_remaining
    half_spread = 0.5 * (
        risk_aversion * variance * time_remaining
        + (2.0 / risk_aversion) * log(1.0 + risk_aversion / liquidity_k)
    )
    bid = reservation - half_spread
    ask = reservation + half_spread
    if bid <= 0.0 or ask <= bid:
        raise ValueError("parameters produce invalid market-making quotes")
    return MarketMakingQuote(
        reservation_price=float(reservation),
        bid=float(bid),
        ask=float(ask),
        half_spread=float(half_spread),
    )
