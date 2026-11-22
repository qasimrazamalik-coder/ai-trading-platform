from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quant_os_core.backtesting import EventDrivenBacktester
from quant_os_core.backtesting.engine import StrategyContext
from quant_os_core.backtesting.models import OrderRequest, OrderSide
from quant_os_core.instruments import InstrumentId
from quant_os_core.market_data.models import Candle

BTC = InstrumentId(venue="REPLAY", symbol="BTCUSDT")
START = datetime(2026, 1, 1, tzinfo=UTC)


class BuyAndHold:
    def __init__(self) -> None:
        self.sent = False

    def on_bar(self, candle: Candle, context: StrategyContext) -> tuple[OrderRequest, ...]:
        if self.sent:
            return ()
        self.sent = True
        return (
            OrderRequest(
                instrument_id=candle.instrument_id,
                side=OrderSide.BUY,
                quantity=Decimal("2"),
            ),
        )


def _candles() -> tuple[Candle, ...]:
    result: list[Candle] = []
    prices = ["100", "101", "99", "105", "103", "110"]
    for index, price_text in enumerate(prices):
        timestamp = START + timedelta(minutes=index)
        price = Decimal(price_text)
        result.append(
            Candle(
                instrument_id=BTC,
                timestamp=timestamp,
                source="determinism",
                timeframe="1m",
                open=price,
                high=price + Decimal("1"),
                low=price - Decimal("1"),
                close=price,
                volume=Decimal("100"),
                close_time=timestamp + timedelta(minutes=1),
            )
        )
    return tuple(result)


def _economic_fill_signature(result: object) -> tuple[tuple[object, ...], ...]:
    fills = result.fills  # type: ignore[attr-defined]
    return tuple(
        (
            fill.instrument_id.key,
            fill.side.value,
            fill.quantity,
            fill.price,
            fill.fee,
            fill.timestamp,
            fill.bar_index,
        )
        for fill in fills
    )


def test_event_driven_backtest_is_economically_deterministic() -> None:
    candles = _candles()
    first = EventDrivenBacktester(initial_cash=Decimal("10000")).run(candles, BuyAndHold())
    second = EventDrivenBacktester(initial_cash=Decimal("10000")).run(candles, BuyAndHold())

    assert first.initial_cash == second.initial_cash
    assert first.final_equity == second.final_equity
    assert first.total_return == second.total_return
    assert first.equity_curve == second.equity_curve
    assert _economic_fill_signature(first) == _economic_fill_signature(second)
    assert first.submitted_orders == second.submitted_orders
    assert first.rejected_orders == second.rejected_orders
    assert first.cancelled_orders == second.cancelled_orders
