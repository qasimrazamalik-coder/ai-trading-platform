from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quant_os_core.backtesting.broker import BrokerConfig, SimulatedBroker
from quant_os_core.backtesting.engine import EventDrivenBacktester
from quant_os_core.backtesting.models import OrderRequest
from quant_os_core.instruments import InstrumentId
from quant_os_core.market_data.models import Candle


def make_candle(instrument, day, open_price):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=day)
    price = Decimal(open_price)
    return Candle(
        instrument_id=instrument,
        timestamp=timestamp,
        close_time=timestamp + timedelta(days=1),
        source="fixture",
        timeframe="1d",
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=Decimal("100"),
    )


def test_latency_counts_target_instrument_bars_not_global_events():
    a = InstrumentId(venue="TEST", symbol="AAA")
    b = InstrumentId(venue="TEST", symbol="BBB")
    events = (
        make_candle(a, 0, "100"),
        make_candle(b, 0, "50"),
        make_candle(a, 1, "101"),
        make_candle(b, 1, "51"),
        make_candle(a, 2, "102"),
        make_candle(b, 2, "52"),
    )

    class Strategy:
        def on_bar(self, candle, context):
            if candle.instrument_id == a and context.instrument_bar_index == 0:
                return (
                    OrderRequest(
                        instrument_id=b,
                        side="buy",
                        quantity=Decimal("1"),
                    ),
                )
            return ()

    broker = SimulatedBroker(config=BrokerConfig(latency_bars=2))
    result = EventDrivenBacktester(initial_cash=Decimal("1000"), broker=broker).run(
        events,
        Strategy(),
    )
    assert len(result.fills) == 1
    assert result.fills[0].instrument_id == b
    assert result.fills[0].bar_index == 1
    assert result.fills[0].price == Decimal("51")
