"""Fixed-duration market-data timeframes and timestamp bucketing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

_TIMEFRAME_RE = re.compile(r"^(?P<count>[1-9][0-9]*)(?P<unit>s|m|h|d|w)$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3_600, "d": 86_400, "w": 604_800}


@dataclass(frozen=True, slots=True, order=True)
class Timeframe:
    seconds: int

    @classmethod
    def parse(cls, value: str) -> "Timeframe":
        match = _TIMEFRAME_RE.fullmatch(value.strip().lower())
        if match is None:
            raise ValueError(f"unsupported fixed timeframe: {value!r}")
        count = int(match.group("count"))
        seconds = count * _UNIT_SECONDS[match.group("unit")]
        return cls(seconds=seconds)

    @property
    def delta(self) -> timedelta:
        return timedelta(seconds=self.seconds)

    def bucket_start(self, timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        utc_ts = timestamp.astimezone(UTC)
        epoch_seconds = int(utc_ts.timestamp())
        bucket = epoch_seconds - (epoch_seconds % self.seconds)
        return datetime.fromtimestamp(bucket, tz=UTC)


def ensure_aware(timestamp: datetime, *, field: str = "timestamp") -> datetime:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return timestamp.astimezone(UTC)
