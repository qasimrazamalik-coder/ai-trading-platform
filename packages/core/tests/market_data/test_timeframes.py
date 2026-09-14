from datetime import UTC, datetime, timedelta

import pytest

from quant_os_core.market_data.timeframes import Timeframe


@pytest.mark.parametrize(
    ("label", "seconds"),
    [("1s", 1), ("5m", 300), ("4h", 14_400), ("1d", 86_400), ("1w", 604_800)],
)
def test_parse_timeframe(label: str, seconds: int) -> None:
    assert Timeframe.parse(label).seconds == seconds


def test_invalid_timeframe_rejected() -> None:
    with pytest.raises(ValueError):
        Timeframe.parse("1month")


def test_bucket_start_is_deterministic() -> None:
    timeframe = Timeframe.parse("5m")
    timestamp = datetime(2026, 1, 1, 12, 7, 42, tzinfo=UTC)
    assert timeframe.bucket_start(timestamp) == datetime(2026, 1, 1, 12, 5, tzinfo=UTC)
    assert timeframe.delta == timedelta(minutes=5)
