"""Shared pieces for the spot bot simulators: bars, a spot wallet, fills and results.

Halal limits are enforced here, once, for every bot: the wallet holds cash and an owned
quantity of one spot asset. A buy that costs more than the cash on hand is refused (no
leverage, no margin, no borrowing) and a sell larger than the quantity owned is refused (no
short selling). All money and quantities are ``Decimal``.

These are research simulators. They never send orders; a live or paper version must still
pass the Shariah gate, the risk engine, the edge-over-cost filter and protections.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from enum import StrEnum

import numpy as np
import numpy.typing as npt

ZERO = Decimal(0)
ONE = Decimal(1)
QTY_STEP = Decimal("1e-10")
FloatArray = npt.NDArray[np.float64]


def to_decimal(value: float | int | str | Decimal) -> Decimal:
    """Convert to ``Decimal`` without binary float noise (floats go through 12 sig. digits)."""

    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, float):
        result = Decimal(f"{value:.12g}")
    else:
        result = Decimal(value)
    if not result.is_finite():
        raise ValueError("value must be finite")
    return result


def require_positive(value: Decimal, name: str) -> Decimal:
    if not value.is_finite() or value <= ZERO:
        raise ValueError(f"{name} must be > 0")
    return value


def require_fraction(value: Decimal, name: str, *, allow_zero: bool = True) -> Decimal:
    low_ok = value >= ZERO if allow_zero else value > ZERO
    if not value.is_finite() or not low_ok or value >= ONE:
        raise ValueError(f"{name} must be in {'[0, 1)' if allow_zero else '(0, 1)'}")
    return value


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLC bar without open (the CSVs carry ``time,high,low,close``)."""

    time: str
    high: Decimal
    low: Decimal
    close: Decimal

    def __post_init__(self) -> None:
        if self.low <= ZERO or self.close <= ZERO:
            raise ValueError("prices must be > 0")
        if self.high < self.low:
            raise ValueError("high cannot be below low")


def bars_from_arrays(
    times: Sequence[str],
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
) -> list[Bar]:
    """Build bars; ``high``/``low`` are widened to include the close if the data disagree."""

    if not len(times) == len(high) == len(low) == len(close):
        raise ValueError("times, high, low and close must have equal length")
    bars: list[Bar] = []
    for t, h, lo, c in zip(times, high, low, close, strict=True):
        hd, ld, cd = to_decimal(h), to_decimal(lo), to_decimal(c)
        bars.append(Bar(t, max(hd, cd), min(ld, cd), cd))
    return bars


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class Fill:
    time: str
    side: Side
    price: Decimal
    quantity: Decimal
    fee: Decimal
    reason: str

    @property
    def notional(self) -> Decimal:
        return self.price * self.quantity


class InsufficientFunds(ValueError):
    """A buy would need more cash than the wallet holds (leverage is not allowed)."""


class InsufficientInventory(ValueError):
    """A sell would need more of the asset than the wallet owns (shorting is not allowed)."""


@dataclass(slots=True)
class SpotWallet:
    """Cash plus an owned quantity of one spot asset. Never negative, never borrowed."""

    cash: Decimal
    quantity: Decimal = ZERO
    fills: list[Fill] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cash < ZERO or self.quantity < ZERO:
            raise ValueError("cash and quantity cannot be negative")

    def max_buy_quantity(self, price: Decimal, fee_rate: Decimal, budget: Decimal) -> Decimal:
        """Largest quantity whose cost plus fee fits in ``min(budget, cash)``."""

        spend = min(budget, self.cash)
        if spend <= ZERO:
            return ZERO
        return (spend / (price * (ONE + fee_rate))).quantize(QTY_STEP, rounding=ROUND_DOWN)

    def buy(
        self, time: str, price: Decimal, quantity: Decimal, fee_rate: Decimal, reason: str
    ) -> Fill:
        require_positive(price, "price")
        require_positive(quantity, "quantity")
        fee = price * quantity * fee_rate
        cost = price * quantity + fee
        if cost > self.cash:
            raise InsufficientFunds(f"buy needs {cost} but only {self.cash} cash is free")
        self.cash -= cost
        self.quantity += quantity
        fill = Fill(time, Side.BUY, price, quantity, fee, reason)
        self.fills.append(fill)
        return fill

    def sell(
        self, time: str, price: Decimal, quantity: Decimal, fee_rate: Decimal, reason: str
    ) -> Fill:
        require_positive(price, "price")
        require_positive(quantity, "quantity")
        if quantity > self.quantity:
            raise InsufficientInventory(
                f"sell of {quantity} exceeds owned {self.quantity} (no short selling)"
            )
        fee = price * quantity * fee_rate
        self.quantity -= quantity
        self.cash += price * quantity - fee
        fill = Fill(time, Side.SELL, price, quantity, fee, reason)
        self.fills.append(fill)
        return fill

    def equity(self, price: Decimal) -> Decimal:
        return self.cash + self.quantity * price


@dataclass(frozen=True, slots=True)
class BotResult:
    """Outcome of a simulation: fills, equity marked at each close, final balances."""

    name: str
    capital: Decimal
    times: tuple[str, ...]
    equity: tuple[Decimal, ...]
    fills: tuple[Fill, ...]
    final_cash: Decimal
    final_quantity: Decimal
    notes: tuple[str, ...] = ()

    @property
    def fees_paid(self) -> Decimal:
        return sum((f.fee for f in self.fills), ZERO)

    @property
    def total_return(self) -> Decimal:
        return self.equity[-1] / self.capital - ONE

    def equity_array(self) -> FloatArray:
        return np.asarray([float(x) for x in self.equity], dtype=np.float64)

    def returns(self) -> FloatArray:
        """Simple returns per bar, the first one measured from the starting capital."""

        curve: FloatArray = np.concatenate([[float(self.capital)], self.equity_array()])
        result: FloatArray = curve[1:] / curve[:-1] - 1.0
        return result


def buy_and_hold(bars: Sequence[Bar], capital: Decimal, fee_rate: Decimal) -> BotResult:
    """Benchmark: spend all capital at the first close, hold, mark at every close."""

    if not bars:
        raise ValueError("bars cannot be empty")
    wallet = SpotWallet(require_positive(capital, "capital"))
    first = bars[0]
    wallet.buy(
        first.time,
        first.close,
        wallet.max_buy_quantity(first.close, fee_rate, wallet.cash),
        fee_rate,
        "buy_and_hold",
    )
    equity = tuple(wallet.equity(bar.close) for bar in bars)
    return BotResult(
        "buy_and_hold",
        capital,
        tuple(b.time for b in bars),
        equity,
        tuple(wallet.fills),
        wallet.cash,
        wallet.quantity,
    )
