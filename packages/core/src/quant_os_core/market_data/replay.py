"""Deterministic replay of historical market-data records."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from quant_os_core.market_data.models import MarketDataRecord


class MarketDataReplay:
    def __init__(self, records: Iterable[MarketDataRecord]) -> None:
        self._records = tuple(
            sorted(
                records,
                key=lambda item: (item.timestamp, item.instrument_id.key, item.source),
            )
        )

    def __iter__(self) -> Iterator[MarketDataRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)
