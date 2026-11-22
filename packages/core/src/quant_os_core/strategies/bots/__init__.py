"""Spot bot simulators: grid, DCA, combo, arbitrage detectors and market making.

All of them are long-only, unlevered and spot only, with money in ``Decimal``. They are
research simulators: none of them sends orders. Arbitrage and market making are labelled
simulation-only. Anything built on them for paper trading must still pass the Shariah gate,
the risk engine, the edge-over-cost filter and protections.
"""

from quant_os_core.strategies.bots.arbitrage import (
    ArbCosts,
    ArbOpportunity,
    Quote,
    detect_cross_exchange,
    detect_triangular,
    survival,
)
from quant_os_core.strategies.bots.combo import ComboConfig, run_combo
from quant_os_core.strategies.bots.common import (
    Bar,
    BotResult,
    Fill,
    InsufficientFunds,
    InsufficientInventory,
    Side,
    SpotWallet,
    bars_from_arrays,
    buy_and_hold,
)
from quant_os_core.strategies.bots.dca import DcaBot, DcaConfig, run_dca
from quant_os_core.strategies.bots.grid import (
    GridBot,
    GridConfig,
    GridStatus,
    RollingGrid,
    Spacing,
    StopAction,
    grid_levels,
    run_grid,
    run_rolling_grid,
)
from quant_os_core.strategies.bots.market_making import MarketMakerConfig, run_market_maker

__all__ = [
    "ArbCosts",
    "ArbOpportunity",
    "Bar",
    "BotResult",
    "ComboConfig",
    "DcaBot",
    "DcaConfig",
    "Fill",
    "GridBot",
    "GridConfig",
    "GridStatus",
    "InsufficientFunds",
    "InsufficientInventory",
    "MarketMakerConfig",
    "Quote",
    "RollingGrid",
    "Side",
    "Spacing",
    "SpotWallet",
    "StopAction",
    "bars_from_arrays",
    "buy_and_hold",
    "detect_cross_exchange",
    "detect_triangular",
    "grid_levels",
    "run_combo",
    "run_dca",
    "run_grid",
    "run_market_maker",
    "run_rolling_grid",
    "survival",
]
