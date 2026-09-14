from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_os_core.backtesting.benchmark import compare_to_benchmark
from quant_os_core.backtesting.engine import EventDrivenBacktester
from quant_os_core.backtesting.models import OrderRequest
from quant_os_core.instruments import InstrumentId
from quant_os_core.market_data.models import Candle


def candle(instrument, day, price):
    start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=day)
    px = Decimal(price)
    return Candle(
        instrument_id=instrument,
        timestamp=start,
        close_time=start + timedelta(days=1),
        source="fixture",
        timeframe="1d",
        open=px,
        high=px + 2,
        low=px - 2,
        close=px + 1,
        volume=Decimal("100"),
    )


def test_multi_asset_order_waits_for_future_target_bar():
    a = InstrumentId(venue="TEST", symbol="AAA")
    b = InstrumentId(venue="TEST", symbol="BBB")
    candles = (
        candle(a, 0, "100"),
        candle(b, 0, "50"),
        candle(a, 1, "101"),
        candle(b, 1, "55"),
    )

    class CrossAssetStrategy:
        def on_bar(self, current, context):
            if current.instrument_id == a and context.instrument_bar_index == 0:
                return (
                    OrderRequest(
                        instrument_id=b,
                        side="buy",
                        quantity=Decimal("1"),
                    ),
                )
            return ()

    result = EventDrivenBacktester(initial_cash=Decimal("1000")).run(
        candles,
        CrossAssetStrategy(),
    )
    assert len(result.fills) == 1
    assert result.fills[0].instrument_id == b
    assert result.fills[0].price == Decimal("55")
    assert result.fills[0].bar_index == 1


def test_benchmark_comparison_uses_same_ledger_length(instrument, candles):
    class BuyOnce:
        def on_bar(self, current, context):
            if context.event_index == 0:
                return (
                    OrderRequest(
                        instrument_id=instrument,
                        side="buy",
                        quantity=Decimal("1"),
                    ),
                )
            return ()

    result = EventDrivenBacktester(initial_cash=Decimal("1000")).run(candles, BuyOnce())
    comparison = compare_to_benchmark(
        result,
        [1000, 1001, 1002, 1003, 1004],
        periods_per_year=252,
    )
    assert comparison.strategy_total_return == pytest.approx(float(result.total_return))
    assert comparison.benchmark_total_return == pytest.approx(0.004)


def test_benchmark_rejects_wrong_length(instrument, candles):
    class Noop:
        def on_bar(self, current, context):
            return ()

    result = EventDrivenBacktester(initial_cash=Decimal("1000")).run(candles, Noop())
    with pytest.raises(ValueError):
        compare_to_benchmark(result, [1000, 1001])
