"""Spot market-making simulator (SIMULATION ONLY): inventory-constrained quotes around mid.

Each bar the bot posts one bid and one ask around the *previous* close, skewed by inventory:
holding more than the target lowers both quotes (sell more readily, buy less), holding less
raises them. Inventory lives in ``[0, max_inventory]``: the ask is only posted for coins the
bot already owns (no short inventory) and the bid only if cash covers it (no leverage).

Fill assumptions (conservative): a quote fills only if the bar trades through it
(``low < bid`` / ``high > ask``), once per bar, at the quoted price, paying ``fee_rate``.
Coins bought in a bar cannot be sold in that same bar. Real queue position, partial fills
and adverse selection are worse than this model; treat any profit here as an upper bound.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from quant_os_core.strategies.bots.common import (
    ONE,
    ZERO,
    Bar,
    BotResult,
    SpotWallet,
    require_fraction,
    require_positive,
)

SIMULATION_ONLY = True


@dataclass(frozen=True, slots=True)
class MarketMakerConfig:
    capital: Decimal
    order_size: Decimal
    max_inventory: Decimal
    half_spread: Decimal = Decimal("0.005")
    skew: Decimal = Decimal("0.01")
    target_ratio: Decimal = Decimal("0.5")
    fee_rate: Decimal = Decimal("0.001")

    def __post_init__(self) -> None:
        require_positive(self.capital, "capital")
        require_positive(self.order_size, "order_size")
        require_positive(self.max_inventory, "max_inventory")
        if self.order_size > self.max_inventory:
            raise ValueError("order_size cannot exceed max_inventory")
        require_fraction(self.half_spread, "half_spread", allow_zero=False)
        require_fraction(self.skew, "skew")
        if not ZERO <= self.target_ratio <= ONE:
            raise ValueError("target_ratio must be in [0, 1]")
        require_fraction(self.fee_rate, "fee_rate")


@dataclass(frozen=True, slots=True)
class Quotes:
    bid: Decimal | None
    ask: Decimal | None
    reservation: Decimal


def make_quotes(
    config: MarketMakerConfig, mid: Decimal, inventory: Decimal, cash: Decimal
) -> Quotes:
    """Skewed quotes for the next bar; a side is ``None`` when the constraint forbids it."""

    require_positive(mid, "mid")
    ratio = inventory / config.max_inventory
    reservation = mid * (ONE - config.skew * (ratio - config.target_ratio))
    bid = reservation * (ONE - config.half_spread)
    ask = reservation * (ONE + config.half_spread)
    buy_cost = bid * config.order_size * (ONE + config.fee_rate)
    can_buy = inventory + config.order_size <= config.max_inventory and buy_cost <= cash
    return Quotes(bid if can_buy else None, ask if inventory > ZERO else None, reservation)


def run_market_maker(
    bars: Sequence[Bar], config: MarketMakerConfig, *, name: str = "market_maker"
) -> BotResult:
    if not bars:
        raise ValueError("bars cannot be empty")
    wallet = SpotWallet(config.capital)
    equity: list[Decimal] = []
    prev_close: Decimal | None = None
    for bar in bars:
        if prev_close is not None:
            quotes = make_quotes(config, prev_close, wallet.quantity, wallet.cash)
            owned_at_open = wallet.quantity
            if quotes.ask is not None and bar.high > quotes.ask:
                qty = min(config.order_size, owned_at_open)
                wallet.sell(bar.time, quotes.ask, qty, config.fee_rate, "mm_ask")
            if quotes.bid is not None and bar.low < quotes.bid:
                wallet.buy(bar.time, quotes.bid, config.order_size, config.fee_rate, "mm_bid")
        equity.append(wallet.equity(bar.close))
        prev_close = bar.close
    return BotResult(
        name,
        config.capital,
        tuple(b.time for b in bars),
        tuple(equity),
        tuple(wallet.fills),
        wallet.cash,
        wallet.quantity,
        ("SIMULATION ONLY", "quotes from previous close; fills need trade-through"),
    )
