"""Combo bot: a DCA core position plus a capped grid slice trading around the price.

Capital is split up front: ``grid_fraction`` of it goes to a re-centring spot grid, the rest
to a DCA bot that accumulates the core position. The two sleeves keep separate wallets, so
the grid can never sell the core coins and neither sleeve can spend the other's cash. Both
are long-only and unlevered; the combined equity is the sum of the two wallets.

Simulation only; real orders still go through the Shariah gate, the risk engine, the
edge-over-cost filter and protections.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import Decimal

from quant_os_core.strategies.bots.common import ONE, Bar, BotResult, require_fraction
from quant_os_core.strategies.bots.dca import DcaBot, DcaConfig
from quant_os_core.strategies.bots.grid import FILL_ASSUMPTIONS, RollingGrid, Spacing


@dataclass(frozen=True, slots=True)
class ComboConfig:
    dca: DcaConfig
    grid_fraction: Decimal = Decimal("0.3")
    grid_width: Decimal = Decimal("0.15")
    grid_levels: int = 8
    grid_spacing: Spacing = Spacing.GEOMETRIC
    stop_margin: Decimal = Decimal("0.05")
    slippage: Decimal = Decimal("0.001")

    def __post_init__(self) -> None:
        require_fraction(self.grid_fraction, "grid_fraction", allow_zero=False)
        if self.grid_fraction > Decimal("0.5"):
            raise ValueError("grid_fraction is capped at 0.5 of capital")
        require_fraction(self.grid_width, "grid_width", allow_zero=False)

    @property
    def capital(self) -> Decimal:
        return self.dca.capital

    @property
    def grid_capital(self) -> Decimal:
        return self.capital * self.grid_fraction

    def core_config(self) -> DcaConfig:
        core_capital = self.capital * (ONE - self.grid_fraction)
        cap = self.dca.max_invested
        return replace(
            self.dca,
            capital=core_capital,
            max_invested=None if cap is None else min(cap, core_capital),
        )


def run_combo(bars: Sequence[Bar], config: ComboConfig, *, name: str = "combo") -> BotResult:
    if not bars:
        raise ValueError("bars cannot be empty")
    core = DcaBot(config.core_config())
    grid = RollingGrid(
        config.grid_capital,
        width=config.grid_width,
        levels=config.grid_levels,
        spacing=config.grid_spacing,
        fee_rate=config.dca.fee_rate,
        stop_margin=config.stop_margin,
        slippage=config.slippage,
    )
    equity: list[Decimal] = []
    for index, bar in enumerate(bars):
        core.step(index, bar)
        grid.step(index, bar)
        equity.append(core.wallet.equity(bar.close) + grid.equity(bar.close))
    fills = sorted([*core.wallet.fills, *grid.wallet.fills], key=lambda f: f.time)
    return BotResult(
        name,
        config.capital,
        tuple(b.time for b in bars),
        tuple(equity),
        tuple(fills),
        core.wallet.cash + grid.wallet.cash,
        core.wallet.quantity + grid.wallet.quantity,
        (
            *FILL_ASSUMPTIONS,
            f"core_quantity={core.wallet.quantity}",
            f"grid_quantity={grid.wallet.quantity}",
            f"grid_recentres={grid.recentres}",
        ),
    )
