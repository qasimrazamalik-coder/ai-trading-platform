from decimal import Decimal

import pytest

from quant_os_core.market_data.aggregation import aggregate_candles
from quant_os_core.market_data.models import Candle


def test_aggregate_five_one_minute_bars(minute_candles: list[Candle]) -> None:
    result = aggregate_candles(minute_candles, "5m")
    assert len(result) == 1
    bar = result[0]
    assert bar.open == Decimal("100")
    assert bar.high == Decimal("106")
    assert bar.low == Decimal("99")
    assert bar.close == Decimal("105")
    assert bar.volume == Decimal("60")
    assert bar.trade_count == 510
    assert bar.is_closed is True


def test_partial_bucket_is_marked_open(minute_candles: list[Candle]) -> None:
    result = aggregate_candles(minute_candles[:3], "5m")
    assert result[0].is_closed is False


def test_smaller_target_is_rejected(minute_candles: list[Candle]) -> None:
    with pytest.raises(ValueError):
        aggregate_candles(minute_candles, "30s")
