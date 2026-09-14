from decimal import Decimal

from quant_os_core.backtesting.broker import BrokerConfig, SimulatedBroker
from quant_os_core.backtesting.costs import ProportionalFeeModel
from quant_os_core.backtesting.models import (
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from quant_os_core.backtesting.slippage import BpsSlippageModel


def submit(broker, candle, request):
    return broker.submit(request, submitted_at=candle.close_time, bar_index=0)


def test_market_order_cannot_fill_same_bar(instrument, candles):
    broker = SimulatedBroker()
    order = submit(
        broker,
        candles[0],
        OrderRequest(instrument_id=instrument, side="buy", quantity=Decimal("1")),
    )
    assert broker.process_bar(candles[0], bar_index=0) == ()
    fills = broker.process_bar(candles[1], bar_index=1)
    assert fills[0].price == candles[1].open
    assert order.status is OrderStatus.FILLED


def test_buy_limit_gap_gets_price_improvement(instrument, candles):
    broker = SimulatedBroker()
    submit(
        broker,
        candles[0],
        OrderRequest(
            instrument_id=instrument,
            side="buy",
            quantity=Decimal("1"),
            order_type="limit",
            limit_price=Decimal("104"),
        ),
    )
    fill = broker.process_bar(candles[1], bar_index=1)[0]
    assert fill.price == Decimal("103")


def test_limit_slippage_never_violates_limit(instrument, candles):
    broker = SimulatedBroker(slippage_model=BpsSlippageModel(fixed_bps=Decimal("1000")))
    submit(
        broker,
        candles[0],
        OrderRequest(
            instrument_id=instrument,
            side="buy",
            quantity=Decimal("1"),
            order_type="limit",
            limit_price=Decimal("104"),
        ),
    )
    fill = broker.process_bar(candles[1], bar_index=1)[0]
    assert fill.price <= Decimal("104")


def test_sell_limit_triggers(instrument, candles):
    broker = SimulatedBroker()
    submit(
        broker,
        candles[0],
        OrderRequest(
            instrument_id=instrument,
            side="sell",
            quantity=Decimal("1"),
            order_type="limit",
            limit_price=Decimal("105"),
        ),
    )
    fill = broker.process_bar(candles[1], bar_index=1)[0]
    assert fill.price == Decimal("105")


def test_buy_stop_gaps_to_open(instrument, candles):
    broker = SimulatedBroker()
    submit(
        broker,
        candles[0],
        OrderRequest(
            instrument_id=instrument,
            side="buy",
            quantity=Decimal("1"),
            order_type="stop",
            stop_price=Decimal("102"),
        ),
    )
    fill = broker.process_bar(candles[1], bar_index=1)[0]
    assert fill.price == Decimal("103")


def test_partial_fill_respects_volume_fraction(instrument, candles):
    broker = SimulatedBroker(config=BrokerConfig(max_bar_volume_fraction=Decimal("0.1")))
    order = submit(
        broker,
        candles[0],
        OrderRequest(instrument_id=instrument, side="buy", quantity=Decimal("15")),
    )
    fill = broker.process_bar(candles[1], bar_index=1)[0]
    assert fill.quantity == Decimal("10.0")
    assert order.remaining_quantity == Decimal("5.0")
    assert order.status is OrderStatus.PARTIALLY_FILLED


def test_ioc_cancels_unfilled_remainder(instrument, candles):
    broker = SimulatedBroker(config=BrokerConfig(max_bar_volume_fraction=Decimal("0.1")))
    order = submit(
        broker,
        candles[0],
        OrderRequest(
            instrument_id=instrument,
            side="buy",
            quantity=Decimal("15"),
            time_in_force=TimeInForce.IOC,
        ),
    )
    broker.process_bar(candles[1], bar_index=1)
    assert order.status is OrderStatus.CANCELLED
    assert broker.cancelled_orders == 1


def test_fee_is_recorded(instrument, candles):
    broker = SimulatedBroker(fee_model=ProportionalFeeModel(taker_bps=Decimal("10")))
    submit(
        broker,
        candles[0],
        OrderRequest(instrument_id=instrument, side="buy", quantity=Decimal("1")),
    )
    fill = broker.process_bar(candles[1], bar_index=1)[0]
    assert fill.fee == Decimal("0.103")


def test_half_spread_is_separate_execution_cost(instrument, candles):
    broker = SimulatedBroker(
        slippage_model=BpsSlippageModel(half_spread_bps=Decimal("10"))
    )
    submit(
        broker,
        candles[0],
        OrderRequest(instrument_id=instrument, side="buy", quantity=Decimal("1")),
    )
    fill = broker.process_bar(candles[1], bar_index=1)[0]
    assert fill.price == Decimal("103.103")


def test_cancel_open_orders_marks_pending(instrument, candles):
    broker = SimulatedBroker()
    order = submit(
        broker,
        candles[0],
        OrderRequest(
            instrument_id=instrument,
            side="buy",
            quantity=Decimal("1"),
            order_type="limit",
            limit_price=Decimal("1"),
        ),
    )
    assert broker.cancel_open_orders() == 1
    assert order.status is OrderStatus.CANCELLED


def test_ioc_cancels_when_limit_never_triggers(instrument, candles):
    broker = SimulatedBroker()
    order = submit(
        broker,
        candles[0],
        OrderRequest(
            instrument_id=instrument,
            side="buy",
            quantity=Decimal("1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("1"),
            time_in_force=TimeInForce.IOC,
        ),
    )
    assert broker.process_bar(candles[1], bar_index=1) == ()
    assert order.status is OrderStatus.CANCELLED


def test_sell_stop_and_nontrigger(instrument, candles):
    broker = SimulatedBroker()
    submit(
        broker,
        candles[0],
        OrderRequest(
            instrument_id=instrument,
            side="sell",
            quantity=Decimal("1"),
            order_type=OrderType.STOP,
            stop_price=Decimal("102"),
        ),
    )
    fill = broker.process_bar(candles[1], bar_index=1)[0]
    assert fill.price == Decimal("102")

    broker.reset()
    submit(
        broker,
        candles[0],
        OrderRequest(
            instrument_id=instrument,
            side="sell",
            quantity=Decimal("1"),
            order_type=OrderType.STOP,
            stop_price=Decimal("1"),
        ),
    )
    assert broker.process_bar(candles[1], bar_index=1) == ()
