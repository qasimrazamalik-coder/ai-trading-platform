"""Diversified core: a monthly, long-only, unlevered allocation across halal ETFs.

This is the bot's default book after the edge hunt (``research/backtests/edge_hunt.py``)
found no signal that beats diversified holding out of sample. Two owner-selectable modes,
both measured on 2005-2026 daily data with next-day execution and 0.10% costs:

* ``GROWTH``: inverse-volatility weights across the universe (risk parity without
  leverage). Out of sample 2019-2026: 14.3% CAGR, Sharpe 0.93, worst drawdown -30.7%.
* ``DEFENSIVE``: the same weights, but an asset below its 200-day average is held as cash.
  Out of sample 2019-2026: 8.2% CAGR, Sharpe 0.87, worst drawdown -13.6%.

Neither is a forecast. They are allocations, so they carry no expected-edge number and
must still pass the Shariah gate and the risk engine before any order is sent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_DOWN, Decimal
from enum import StrEnum
from itertools import pairwise

CORE_UNIVERSE = ("SPY", "QQQ", "GLD", "XLK", "XLE", "XLV", "XLP", "XLY", "XLI", "XLB", "XLU")
VOL_WINDOW = 63
TREND_WINDOW = 200
TRADING_DAYS = 252
_WEIGHT_STEP = Decimal("0.0001")
_CENT = Decimal("0.01")


class CoreMode(StrEnum):
    GROWTH = "growth"
    DEFENSIVE = "defensive"


@dataclass(frozen=True, slots=True)
class RebalanceOrder:
    symbol: str
    side: str  # "buy" or "sell"
    notional: Decimal  # dollars, rounded down to the cent

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if self.notional <= 0:
            raise ValueError("notional must be positive")


def _annual_vol(closes: Sequence[float], window: int) -> float:
    tail = closes[-(window + 1) :]
    rets = [math.log(b / a) for a, b in pairwise(tail)]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    return math.sqrt(max(var, 1e-12) * TRADING_DAYS)


def target_weights(
    closes: Mapping[str, Sequence[float]],
    mode: CoreMode = CoreMode.DEFENSIVE,
    *,
    vol_window: int = VOL_WINDOW,
    trend_window: int = TREND_WINDOW,
) -> dict[str, Decimal]:
    """Target weight per symbol from daily closes ending at the decision bar.

    Weights are non-negative and sum to at most 1; the remainder is cash.
    """

    if not closes:
        raise ValueError("closes must not be empty")
    need = max(vol_window + 1, trend_window if mode is CoreMode.DEFENSIVE else 0)
    for symbol, series in closes.items():
        if len(series) < need:
            raise ValueError(f"{symbol}: need at least {need} closes, got {len(series)}")
        if any(not math.isfinite(p) or p <= 0 for p in series[-need:]):
            raise ValueError(f"{symbol}: closes must be finite and positive")
    inverse = {s: 1.0 / _annual_vol(c, vol_window) for s, c in closes.items()}
    total = sum(inverse.values())
    weights: dict[str, Decimal] = {}
    for symbol, inv in inverse.items():
        weight = inv / total
        if mode is CoreMode.DEFENSIVE:
            series = closes[symbol]
            average = sum(series[-trend_window:]) / trend_window
            if series[-1] <= average:
                weight = 0.0
        weights[symbol] = Decimal(repr(weight)).quantize(_WEIGHT_STEP, rounding=ROUND_DOWN)
    return weights


def is_rebalance_day(today: date, last_rebalance: date | None) -> bool:
    """Rebalance once per calendar month, on the first run of a new month."""

    return last_rebalance is None or (today.year, today.month) != (
        last_rebalance.year,
        last_rebalance.month,
    )


def rebalance_orders(
    targets: Mapping[str, Decimal],
    holdings: Mapping[str, Decimal],
    cash: Decimal,
    *,
    band: Decimal = Decimal("0.02"),
    min_notional: Decimal = Decimal("1.00"),
) -> list[RebalanceOrder]:
    """Orders that move ``holdings`` (market value per symbol) towards ``targets``.

    A position is traded only when its weight is off target by more than ``band``, or
    when the target is zero (full exit). Sells come first; buys are funded only by cash
    on hand plus sale proceeds, scaled down pro rata if short, so the book never levers.
    """

    if cash < 0 or any(v < 0 for v in holdings.values()):
        raise ValueError("cash and holdings must be non-negative (no margin, no shorts)")
    if any(w < 0 for w in targets.values()) or sum(targets.values(), Decimal(0)) > 1:
        raise ValueError("targets must be long-only and sum to at most 1")
    equity = cash + sum(holdings.values(), Decimal(0))
    if equity <= 0:
        return []
    sells: list[RebalanceOrder] = []
    wanted: dict[str, Decimal] = {}
    for symbol in sorted(set(targets) | set(holdings)):
        held = holdings.get(symbol, Decimal(0))
        goal = equity * targets.get(symbol, Decimal(0))
        drift = abs(held - goal) / equity
        if drift <= band and not (goal == 0 and held > 0):
            continue
        delta = (goal - held).quantize(_CENT, rounding=ROUND_DOWN)
        if delta <= -min_notional:
            sells.append(RebalanceOrder(symbol, "sell", -delta))
        elif delta >= min_notional:
            wanted[symbol] = delta
    budget = cash + sum((o.notional for o in sells), Decimal(0))
    need = sum(wanted.values(), Decimal(0))
    scale = min(Decimal(1), budget / need) if need > 0 else Decimal(1)
    buys = [
        RebalanceOrder(s, "buy", (d * scale).quantize(_CENT, rounding=ROUND_DOWN))
        for s, d in wanted.items()
        if d * scale >= min_notional
    ]
    return sells + buys
