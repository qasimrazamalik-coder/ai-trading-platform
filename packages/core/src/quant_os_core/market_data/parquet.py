"""Optional Parquet persistence adapter.

The core package does not require PyArrow. Installing ``quant-os[marketdata]`` enables this
adapter, keeping the default developer setup light and free of mandatory heavyweight binaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_os_core.market_data.models import Candle


class ParquetUnavailableError(RuntimeError):
    pass


class ParquetCandleWriter:
    def write(self, path: str | Path, candles: list[Candle]) -> None:
        try:
            import pyarrow as pa  # type: ignore[import-not-found]
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ParquetUnavailableError(
                "Parquet support requires the optional 'marketdata' dependency group"
            ) from exc

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for candle in candles:
            rows.append(
                {
                    "venue": candle.instrument_id.venue,
                    "symbol": candle.instrument_id.symbol,
                    "timestamp": candle.timestamp,
                    "close_time": candle.close_time,
                    "source": candle.source,
                    "timeframe": candle.timeframe,
                    "open": str(candle.open),
                    "high": str(candle.high),
                    "low": str(candle.low),
                    "close": str(candle.close),
                    "volume": str(candle.volume),
                    "quote_volume": (
                        str(candle.quote_volume) if candle.quote_volume is not None else None
                    ),
                    "trade_count": candle.trade_count,
                    "is_closed": candle.is_closed,
                }
            )
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, destination, compression="zstd")
