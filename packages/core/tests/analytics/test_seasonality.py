from datetime import UTC, datetime

import pytest

from quant_os_core.analytics.seasonality import (
    pnl_calendar,
    returns_by_hour,
    returns_by_month,
    returns_by_weekday,
)

TIMES = [
    datetime(2026, 1, 5, 9, tzinfo=UTC),
    datetime(2026, 1, 5, 9, 30, tzinfo=UTC),
    datetime(2026, 2, 3, 15, tzinfo=UTC),
]


def test_return_buckets() -> None:
    hourly = returns_by_hour(TIMES, [0.01, 0.03, -0.02])
    weekday = returns_by_weekday(TIMES, [0.01, 0.03, -0.02])
    monthly = returns_by_month(TIMES, [0.01, 0.03, -0.02])
    assert hourly.values[9] == pytest.approx(0.02)
    assert hourly.counts[9] == 2
    assert weekday.values[0] == pytest.approx(0.02)
    assert monthly.values[0] == pytest.approx(0.02)
    assert monthly.values[1] == pytest.approx(-0.02)


def test_pnl_calendar_aggregates_day() -> None:
    result = pnl_calendar(TIMES, [100.0, -25.0, 50.0])
    assert result[0].date == "2026-01-05"
    assert result[0].value == pytest.approx(75.0)
    assert result[0].count == 2
    assert result[1].value == pytest.approx(50.0)


def test_timestamp_validation() -> None:
    with pytest.raises(ValueError, match="equal length"):
        returns_by_hour(TIMES, [0.1])
    with pytest.raises(ValueError, match="timezone-aware"):
        returns_by_hour([datetime(2026, 1, 1)], [0.1])
    with pytest.raises(TypeError):
        returns_by_hour(["2026-01-01"], [0.1])  # type: ignore[list-item]
