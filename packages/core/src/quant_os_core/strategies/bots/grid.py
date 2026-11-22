"""Spot grid bot: buy low at grid levels, sell the owned lot one level higher.

Halal shape: spot only, long-only, unlevered. The grid starts in cash, buys a lot at a level
below the market, and only ever sells a lot it already owns, one level up. It never sells
more than it holds and never spends more cash than it has (``SpotWallet`` refuses both).

Bar-by-bar fill assumptions (conservative, documented so results are not flattered):

* Orders are set from the previous close. A buy limit at level ``p`` is resting only if the
  previous close was above ``p`` (a buy above the market would be a chase, not a grid buy).
* A level fills only if the bar trades *through* it: buy needs ``low < p``, sell needs
  ``high > p``. A bare touch (``low == p``) does not fill.
* At most one fill per level per bar. A lot bought in a bar cannot be sold in the same bar,
  and a cell that sold in a bar cannot rebuy in the same bar (the intrabar path is unknown).
* Fills happen at the limit price with a fee on every fill.
* Range exit: when a close is more than ``stop_margin`` outside ``[lower, upper]`` the grid
  cancels its orders at that close. ``StopAction.SELL`` sells the grid's lots at the *next*
  close (minus ``slippage``) and stops; ``StopAction.PAUSE`` keeps the lots and resumes when
  a close is back inside the range.

Simulation only: a paper or live version must still pass the Shariah gate, the risk engine,
the edge-over-cost filter and protections before any order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum

from quant_os_core.strategies.bots.common import (
    ONE,
    ZERO,
    Bar,
    BotResult,
    SpotWallet,
    require_fraction,
    require_positive,
)

FILL_ASSUMPTIONS = (
    "orders set from the previous close; buy limit rests only below the previous close",
    "a level fills only if the bar trades through it (low < level / high > level)",
    "at most one fill per level per bar; no same-bar round trip on a lot",
    "range-exit market sale executes at the next close minus slippage",
)


class Spacing(StrEnum):
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


class StopAction(StrEnum):
    SELL = "sell"
    PAUSE = "pause"


class GridStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    EXITING = "exiting"
    STOPPED = "stopped"


def grid_levels(
    lower: Decimal, upper: Decimal, levels: int, spacing: Spacing
) -> tuple[Decimal, ...]:
    """``levels`` prices from ``lower`` to ``upper`` inclusive, evenly or geometrically spaced."""

    require_positive(lower, "lower")
    if upper <= lower:
        raise ValueError("upper must be above lower")
    if levels < 2:
        raise ValueError("a grid needs at least 2 levels")
    steps = levels - 1
    if spacing is Spacing.ARITHMETIC:
        step = (upper - lower) / steps
        prices = [lower + step * i for i in range(steps)]
    else:
        ratio = (upper / lower) ** (ONE / steps)
        prices = [lower * ratio**i for i in range(steps)]
    return (*prices, upper)


@dataclass(frozen=True, slots=True)
class GridConfig:
    lower: Decimal
    upper: Decimal
    levels: int
    capital: Decimal
    spacing: Spacing = Spacing.GEOMETRIC
    fee_rate: Decimal = Decimal("0.001")
    stop_margin: Decimal = Decimal("0.05")
    stop_action: StopAction = StopAction.SELL
    slippage: Decimal = Decimal("0.001")

    def __post_init__(self) -> None:
        grid_levels(self.lower, self.upper, self.levels, self.spacing)
        require_positive(self.capital, "capital")
        require_fraction(self.fee_rate, "fee_rate")
        require_fraction(self.stop_margin, "stop_margin")
        require_fraction(self.slippage, "slippage")

    @property
    def prices(self) -> tuple[Decimal, ...]:
        return grid_levels(self.lower, self.upper, self.levels, self.spacing)

    @property
    def exit_low(self) -> Decimal:
        return self.lower * (ONE - self.stop_margin)

    @property
    def exit_high(self) -> Decimal:
        return self.upper * (ONE + self.stop_margin)


class GridBot:
    """One fixed grid. Cell ``i`` buys at ``prices[i]`` and sells at ``prices[i + 1]``."""

    def __init__(self, config: GridConfig, wallet: SpotWallet | None = None) -> None:
        self.config = config
        self.wallet = wallet if wallet is not None else SpotWallet(config.capital)
        self.prices = config.prices
        cells = len(self.prices) - 1
        self.cell_budget = config.capital / cells
        self.lots: list[Decimal] = [ZERO] * cells
        self.bought_at: list[int] = [-1] * cells
        self.status = GridStatus.ACTIVE
        self.prev_close: Decimal | None = None
        self.round_trips = 0

    @property
    def grid_quantity(self) -> Decimal:
        return sum(self.lots, ZERO)

    def step(self, index: int, bar: Bar) -> None:
        cfg = self.config
        if self.status is GridStatus.EXITING:
            qty = self.grid_quantity
            if qty > ZERO:
                self.wallet.sell(
                    bar.time, bar.close * (ONE - cfg.slippage), qty, cfg.fee_rate, "grid_exit"
                )
            self.lots = [ZERO] * len(self.lots)
            self.status = GridStatus.STOPPED
        elif self.status is GridStatus.ACTIVE and self.prev_close is not None:
            self._fill(index, bar, self.prev_close)
        self._check_range(bar.close)
        self.prev_close = bar.close

    def _fill(self, index: int, bar: Bar, prev_close: Decimal) -> None:
        cfg = self.config
        used: set[int] = set()
        sold: set[int] = set()
        for cell, lot in enumerate(self.lots):
            level = cell + 1
            if lot > ZERO and self.bought_at[cell] < index and bar.high > self.prices[level]:
                self.wallet.sell(bar.time, self.prices[level], lot, cfg.fee_rate, "grid_sell")
                self.lots[cell] = ZERO
                used.add(level)
                sold.add(cell)
                self.round_trips += 1
        for cell, lot in enumerate(self.lots):
            price = self.prices[cell]
            if lot > ZERO or cell in sold or cell in used or not prev_close > price:
                continue
            if bar.low < price:
                qty = self.wallet.max_buy_quantity(price, cfg.fee_rate, self.cell_budget)
                if qty <= ZERO:
                    continue
                self.wallet.buy(bar.time, price, qty, cfg.fee_rate, "grid_buy")
                self.lots[cell] = qty
                self.bought_at[cell] = index
                used.add(cell)

    def _check_range(self, close: Decimal) -> None:
        cfg = self.config
        outside = close < cfg.exit_low or close > cfg.exit_high
        if self.status is GridStatus.ACTIVE and outside:
            if cfg.stop_action is StopAction.SELL:
                self.status = (
                    GridStatus.EXITING if self.grid_quantity > ZERO else GridStatus.STOPPED
                )
            else:
                self.status = GridStatus.PAUSED
        elif self.status is GridStatus.PAUSED and cfg.lower <= close <= cfg.upper:
            self.status = GridStatus.ACTIVE


def _result(
    name: str,
    capital: Decimal,
    bars: Sequence[Bar],
    equity: list[Decimal],
    wallet: SpotWallet,
    notes: tuple[str, ...],
) -> BotResult:
    return BotResult(
        name,
        capital,
        tuple(b.time for b in bars),
        tuple(equity),
        tuple(wallet.fills),
        wallet.cash,
        wallet.quantity,
        notes,
    )


def run_grid(bars: Sequence[Bar], config: GridConfig, *, name: str = "grid") -> BotResult:
    """Simulate one fixed grid over ``bars``; equity is marked at every close."""

    if not bars:
        raise ValueError("bars cannot be empty")
    bot = GridBot(config)
    equity: list[Decimal] = []
    for index, bar in enumerate(bars):
        bot.step(index, bar)
        equity.append(bot.wallet.equity(bar.close))
    notes = (*FILL_ASSUMPTIONS, f"status={bot.status.value}", f"round_trips={bot.round_trips}")
    return _result(name, config.capital, bars, equity, bot.wallet, notes)


class RollingGrid:
    """A grid centred on the price that re-centres after each range exit (sell to cash).

    The band is ``close x (1 -/+ width)``. After a range exit sells the lots, a new grid is
    built around the close of the bar that executed the sale, using all the wallet's cash.
    """

    def __init__(
        self,
        capital: Decimal,
        *,
        width: Decimal,
        levels: int,
        spacing: Spacing = Spacing.GEOMETRIC,
        fee_rate: Decimal = Decimal("0.001"),
        stop_margin: Decimal = Decimal("0.05"),
        slippage: Decimal = Decimal("0.001"),
    ) -> None:
        require_fraction(width, "width", allow_zero=False)
        self.wallet = SpotWallet(require_positive(capital, "capital"))
        self.template = GridConfig(
            Decimal(1) - width,
            Decimal(1) + width,
            levels,
            capital,
            spacing,
            fee_rate,
            stop_margin,
            StopAction.SELL,
            slippage,
        )
        self.width = width
        self.bot: GridBot | None = None
        self.recentres = 0

    def _new_grid(self, close: Decimal) -> GridBot:
        config = replace(
            self.template,
            lower=close * (ONE - self.width),
            upper=close * (ONE + self.width),
            capital=self.wallet.cash,
        )
        return GridBot(config, self.wallet)

    def step(self, index: int, bar: Bar) -> None:
        if self.bot is None:
            self.bot = self._new_grid(bar.close)
        self.bot.step(index, bar)
        if self.bot.status is GridStatus.STOPPED and self.wallet.cash > ZERO:
            self.bot = self._new_grid(bar.close)
            self.bot.prev_close = bar.close
            self.recentres += 1

    def equity(self, price: Decimal) -> Decimal:
        return self.wallet.equity(price)


def run_rolling_grid(
    bars: Sequence[Bar],
    capital: Decimal,
    *,
    width: Decimal,
    levels: int,
    spacing: Spacing = Spacing.GEOMETRIC,
    fee_rate: Decimal = Decimal("0.001"),
    stop_margin: Decimal = Decimal("0.05"),
    slippage: Decimal = Decimal("0.001"),
    name: str = "rolling_grid",
) -> BotResult:
    if not bars:
        raise ValueError("bars cannot be empty")
    grid = RollingGrid(
        capital,
        width=width,
        levels=levels,
        spacing=spacing,
        fee_rate=fee_rate,
        stop_margin=stop_margin,
        slippage=slippage,
    )
    equity: list[Decimal] = []
    for index, bar in enumerate(bars):
        grid.step(index, bar)
        equity.append(grid.equity(bar.close))
    notes = (*FILL_ASSUMPTIONS, f"recentres={grid.recentres}")
    return _result(name, capital, bars, equity, grid.wallet, notes)
