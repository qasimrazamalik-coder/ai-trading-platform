"""Universal market-data domain package."""

from .aggregation import aggregate_candles
from .corporate_actions import CorporateAction, CorporateActionType
from .models import (
    BookLevel,
    Candle,
    MarketDataKind,
    OrderBookSnapshot,
    Quote,
    Trade,
    TradeSide,
)
from .parquet import ParquetCandleWriter, ParquetUnavailableError
from .providers import MarketDataProvider, MarketDataStreamItem
from .replay import MarketDataReplay
from .storage import CandleStore, InMemoryCandleStore
from .timeframes import Timeframe
from .validation import DataQualityReport, Gap, MarketDataValidator, Outlier

__all__ = [
    "BookLevel",
    "Candle",
    "CandleStore",
    "CorporateAction",
    "CorporateActionType",
    "DataQualityReport",
    "Gap",
    "InMemoryCandleStore",
    "MarketDataKind",
    "MarketDataProvider",
    "MarketDataReplay",
    "MarketDataStreamItem",
    "MarketDataValidator",
    "OrderBookSnapshot",
    "Outlier",
    "ParquetCandleWriter",
    "ParquetUnavailableError",
    "Quote",
    "Timeframe",
    "Trade",
    "TradeSide",
    "aggregate_candles",
]
