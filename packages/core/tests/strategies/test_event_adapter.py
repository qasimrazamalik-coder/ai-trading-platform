from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quant_os_core.backtesting import EventDrivenBacktester
from quant_os_core.instruments import InstrumentId
from quant_os_core.market_data.models import Candle
from quant_os_core.strategies import SingleAssetTargetAdapter


def candle(index: int, price: str = "100") -> Candle:
    start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
    value = Decimal(price)
    return Candle(
        instrument_id=InstrumentId(venue="TEST", symbol="ABC"),
        timestamp=start,
        source="test",
        timeframe="1m",
        open=value,
        high=value + Decimal("1"),
        low=value - Decimal("1"),
        close=value,
        volume=Decimal("1000"),
        close_time=start + timedelta(minutes=1),
    )


def test_event_adapter_uses_actual_portfolio_position() -> None:
    strategy = SingleAssetTargetAdapter(
        signal_function=lambda closes: 1.0 if len(closes) >= 2 else 0.0,
        max_units=Decimal("2"),
        minimum_history=2,
    )
    bars = tuple(candle(index, str(100 + index)) for index in range(5))
    result = EventDrivenBacktester(initial_cash=Decimal("10000")).run(bars, strategy)
    assert len(result.fills) == 1
    assert result.fills[0].quantity == Decimal("2.0")


def test_event_adapter_rebalances_when_signal_changes() -> None:
    def signal(closes):
        return 1.0 if len(closes) < 4 else -1.0

    strategy = SingleAssetTargetAdapter(
        signal_function=signal,
        max_units=Decimal("1"),
        minimum_history=2,
    )
    bars = tuple(candle(index, str(100 + index)) for index in range(7))
    result = EventDrivenBacktester(initial_cash=Decimal("10000")).run(bars, strategy)
    quantities = [fill.signed_quantity for fill in result.fills]
    assert Decimal("1.0") in quantities
    assert Decimal("-2.0") in quantities


def test_event_adapter_rejects_signal_outside_bounds() -> None:
    strategy = SingleAssetTargetAdapter(
        signal_function=lambda closes: 2.0,
        max_units=Decimal("1"),
        minimum_history=1,
    )
    bars = (candle(0), candle(1))
    try:
        EventDrivenBacktester(initial_cash=Decimal("1000")).run(bars, strategy)
    except ValueError as exc:
        assert "inside" in str(exc)
    else:
        raise AssertionError("expected invalid signal to raise")


def test_event_adapter_does_not_duplicate_pending_partial_order() -> None:
    from quant_os_core.backtesting.broker import BrokerConfig, SimulatedBroker

    strategy = SingleAssetTargetAdapter(
        signal_function=lambda closes: 1.0,
        max_units=Decimal("10"),
        minimum_history=1,
    )
    broker = SimulatedBroker(
        config=BrokerConfig(latency_bars=1, max_bar_volume_fraction=Decimal("0.001"))
    )
    bars = tuple(candle(index, str(100 + index)) for index in range(6))
    result = EventDrivenBacktester(initial_cash=Decimal("10000"), broker=broker).run(bars, strategy)
    assert result.submitted_orders == 1
    assert sum((fill.quantity for fill in result.fills), Decimal("0")) < Decimal("10")
