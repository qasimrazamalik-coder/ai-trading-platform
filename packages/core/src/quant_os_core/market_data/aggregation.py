"""Deterministic candle aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from decimal import Decimal

from quant_os_core.market_data.models import Candle
from quant_os_core.market_data.timeframes import Timeframe


def aggregate_candles(candles: list[Candle], target_timeframe: str) -> list[Candle]:
    """Aggregate smaller fixed-duration candles into a larger timeframe.

    Incomplete target buckets are retained but marked ``is_closed=False``. Input candles must
    all refer to the same instrument/timeframe and the target must be an integer multiple.
    """

    if not candles:
        return []

    source = Timeframe.parse(candles[0].timeframe)
    target = Timeframe.parse(target_timeframe)
    if target.seconds < source.seconds or target.seconds % source.seconds != 0:
        raise ValueError("target timeframe must be an integer multiple of source timeframe")

    instrument = candles[0].instrument_id
    source_label = candles[0].timeframe
    for candle in candles:
        if candle.instrument_id != instrument:
            raise ValueError("cannot aggregate multiple instruments together")
        if candle.timeframe != source_label:
            raise ValueError("cannot aggregate mixed source timeframes")

    ordered = sorted(candles, key=lambda item: item.timestamp)
    buckets: dict[object, list[Candle]] = defaultdict(list)
    for candle in ordered:
        buckets[target.bucket_start(candle.timestamp)].append(candle)

    expected_count = target.seconds // source.seconds
    result: list[Candle] = []
    for start in sorted(buckets):
        items = sorted(buckets[start], key=lambda item: item.timestamp)
        result.append(
            Candle(
                instrument_id=instrument,
                timestamp=start,
                source=f"aggregate:{source_label}",
                timeframe=target_timeframe,
                open=items[0].open,
                high=max(item.high for item in items),
                low=min(item.low for item in items),
                close=items[-1].close,
                volume=sum((item.volume for item in items), Decimal(0)),
                quote_volume=_optional_decimal_sum(item.quote_volume for item in items),
                trade_count=_optional_int_sum(item.trade_count for item in items),
                close_time=start + target.delta,
                is_closed=len(items) == expected_count and all(item.is_closed for item in items),
            )
        )
    return result


def _optional_decimal_sum(values: Iterable[Decimal | None]) -> Decimal | None:
    items = list(values)
    if any(value is None for value in items):
        return None
    return sum((value for value in items if value is not None), Decimal(0))


def _optional_int_sum(values: Iterable[int | None]) -> int | None:
    items = list(values)
    if any(value is None for value in items):
        return None
    return sum(value for value in items if value is not None)
