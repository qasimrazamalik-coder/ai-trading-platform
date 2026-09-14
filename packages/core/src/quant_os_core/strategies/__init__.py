"""Quant OS strategy factory."""

from quant_os_core.strategies.base import (
    ParameterDefinition,
    StrategyCategory,
    StrategyMetadata,
    StrategySignals,
)
from quant_os_core.strategies.carry import (
    annualized_basis,
    cash_and_carry_signal,
    funding_carry_signal,
)
from quant_os_core.strategies.defaults import create_default_registry
from quant_os_core.strategies.ensemble import majority_vote, weighted_ensemble
from quant_os_core.strategies.event_adapter import SingleAssetTargetAdapter
from quant_os_core.strategies.events import credibility_weighted_event_signal, decay_event_signal
from quant_os_core.strategies.indicators import (
    atr,
    bollinger_bands,
    cumulative_vwap,
    donchian_channels,
    ema,
    rate_of_change,
    rolling_zscore,
    rsi,
    sma,
    true_range,
)
from quant_os_core.strategies.market_making import MarketMakingQuote, avellaneda_stoikov_quote
from quant_os_core.strategies.mean_reversion import bollinger_mean_reversion, zscore_mean_reversion
from quant_os_core.strategies.microstructure import (
    microprice,
    order_book_imbalance,
    trade_flow_imbalance,
    vwap_deviation_signal,
)
from quant_os_core.strategies.momentum import cross_sectional_momentum, time_series_momentum
from quant_os_core.strategies.pairs import PairsSignalResult, pairs_zscore_signal
from quant_os_core.strategies.registry import StrategyRegistry
from quant_os_core.strategies.research import StrategyResearchResult, evaluate_strategy_signals
from quant_os_core.strategies.screeners import (
    CanSlimInputs,
    CanSlimScore,
    MinerviniInputs,
    MinerviniScreenResult,
    canslim_score,
    minervini_trend_template,
)
from quant_os_core.strategies.trend import donchian_breakout, moving_average_crossover
from quant_os_core.strategies.volatility import (
    inverse_volatility_weights,
    volatility_breakout_signal,
)

__all__ = [
    "CanSlimInputs",
    "CanSlimScore",
    "MarketMakingQuote",
    "MinerviniInputs",
    "MinerviniScreenResult",
    "PairsSignalResult",
    "ParameterDefinition",
    "SingleAssetTargetAdapter",
    "StrategyCategory",
    "StrategyMetadata",
    "StrategyRegistry",
    "StrategyResearchResult",
    "StrategySignals",
    "annualized_basis",
    "atr",
    "avellaneda_stoikov_quote",
    "bollinger_bands",
    "bollinger_mean_reversion",
    "canslim_score",
    "cash_and_carry_signal",
    "create_default_registry",
    "credibility_weighted_event_signal",
    "cross_sectional_momentum",
    "cumulative_vwap",
    "decay_event_signal",
    "donchian_breakout",
    "donchian_channels",
    "ema",
    "evaluate_strategy_signals",
    "funding_carry_signal",
    "inverse_volatility_weights",
    "majority_vote",
    "microprice",
    "minervini_trend_template",
    "moving_average_crossover",
    "order_book_imbalance",
    "pairs_zscore_signal",
    "rate_of_change",
    "rolling_zscore",
    "rsi",
    "sma",
    "time_series_momentum",
    "trade_flow_imbalance",
    "true_range",
    "volatility_breakout_signal",
    "vwap_deviation_signal",
    "weighted_ensemble",
    "zscore_mean_reversion",
]
