"""Calendar and session-return analytics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import numpy as np
from numpy.typing import ArrayLike

from quant_os_core.analytics._validation import finite_1d
from quant_os_core.analytics.models import CalendarCell, TimeBucketData

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def _validate_timestamp_returns(
    timestamps: Sequence[datetime], returns: ArrayLike
) -> tuple[tuple[datetime, ...], np.ndarray]:
    values = finite_1d(returns, name="returns")
    times = tuple(timestamps)
    if len(times) != values.size:
        raise ValueError("timestamps and returns must have equal length")
    normalized: list[datetime] = []
    for timestamp in times:
        if not isinstance(timestamp, datetime):
            raise TypeError("timestamps must contain datetime values")
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        normalized.append(timestamp.astimezone(UTC))
    return tuple(normalized), values


def _bucket(
    timestamps: Sequence[datetime],
    returns: ArrayLike,
    *,
    bucket_count: int,
    indexer: Callable[[datetime], int],
    labels: Sequence[str],
    title: str,
) -> TimeBucketData:
    times, values = _validate_timestamp_returns(timestamps, returns)
    sums = np.zeros(bucket_count, dtype=np.float64)
    counts = np.zeros(bucket_count, dtype=np.int64)
    for timestamp, value in zip(times, values, strict=True):
        index = int(indexer(timestamp))
        sums[index] += float(value)
        counts[index] += 1
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return TimeBucketData(
        title=title,
        metric="mean_return",
        labels=tuple(labels),
        values=tuple(float(value) for value in means),
        counts=tuple(int(value) for value in counts),
    )


def returns_by_hour(timestamps: Sequence[datetime], returns: ArrayLike) -> TimeBucketData:
    return _bucket(
        timestamps,
        returns,
        bucket_count=24,
        indexer=lambda value: value.hour,
        labels=tuple(f"{hour:02d}:00" for hour in range(24)),
        title="Mean return by UTC hour",
    )


def returns_by_weekday(timestamps: Sequence[datetime], returns: ArrayLike) -> TimeBucketData:
    return _bucket(
        timestamps,
        returns,
        bucket_count=7,
        indexer=lambda value: value.weekday(),
        labels=_WEEKDAYS,
        title="Mean return by weekday",
    )


def returns_by_month(timestamps: Sequence[datetime], returns: ArrayLike) -> TimeBucketData:
    return _bucket(
        timestamps,
        returns,
        bucket_count=12,
        indexer=lambda value: value.month - 1,
        labels=_MONTHS,
        title="Mean return by month",
    )


def pnl_calendar(timestamps: Sequence[datetime], pnl: ArrayLike) -> tuple[CalendarCell, ...]:
    times, values = _validate_timestamp_returns(timestamps, pnl)
    aggregate: dict[str, list[float]] = defaultdict(list)
    for timestamp, value in zip(times, values, strict=True):
        aggregate[timestamp.date().isoformat()].append(float(value))
    return tuple(
        CalendarCell(date=date, value=float(np.sum(entries)), count=len(entries))
        for date, entries in sorted(aggregate.items())
    )
