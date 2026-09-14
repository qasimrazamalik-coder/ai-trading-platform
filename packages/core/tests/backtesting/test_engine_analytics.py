from decimal import Decimal

import pytest

from quant_os_core.backtesting.analytics import summarize_backtest
from quant_os_core.backtesting.broker import SimulatedBroker
from quant_os_core.backtesting.costs import FinancingModel, ProportionalFeeModel
from quant_os_core.backtesting.engine import EventDrivenBacktester
from quant_os_core.backtesting.models import OrderRequest


class BuyOnce:
    def __init__(self, instrument):
        self.instrument = instrument

    def on_bar(self, candle, context):
        if context.bar_index == 0:
            return (
                OrderRequest(
                    instrument_id=self.instrument,
                    side="buy",
                    quantity=Decimal("1"),
                ),
            )
        return ()


class BuyThenSell:
    def __init__(self, instrument):
        self.instrument = instrument

    def on_bar(self, candle, context):
        if context.bar_index == 0:
            return (OrderRequest(instrument_id=self.instrument, side="buy", quantity=Decimal("1")),)
        if context.bar_index == 2:
            return (
                OrderRequest(
                    instrument_id=self.instrument,
                    side="sell",
                    quantity=Decimal("1"),
                ),
            )
        return ()


def test_event_backtest_is_next_bar_execution(instrument, candles):
    result = EventDrivenBacktester(initial_cash=Decimal("1000")).run(
        candles,
        BuyOnce(instrument),
    )
    assert len(result.fills) == 1
    assert result.fills[0].bar_index == 1
    assert result.fills[0].price == candles[1].open


def test_event_backtest_equity_is_accounting_consistent(instrument, candles):
    result = EventDrivenBacktester(initial_cash=Decimal("1000")).run(
        candles,
        BuyThenSell(instrument),
    )
    assert len(result.fills) == 2
    assert result.fills[0].price == Decimal("103")
    assert result.fills[1].price == Decimal("106")
    assert result.final_equity == Decimal("1003")


def test_event_backtest_includes_fees(instrument, candles):
    broker = SimulatedBroker(fee_model=ProportionalFeeModel(taker_bps=Decimal("10")))
    result = EventDrivenBacktester(initial_cash=Decimal("1000"), broker=broker).run(
        candles,
        BuyThenSell(instrument),
    )
    assert result.final_equity < Decimal("1003")


def test_financing_reduces_equity(instrument, candles):
    baseline = EventDrivenBacktester(initial_cash=Decimal("1000")).run(candles, BuyOnce(instrument))
    financed = EventDrivenBacktester(
        initial_cash=Decimal("1000"),
        financing_model=FinancingModel(
            annual_long_rate=Decimal("0.365"),
            periods_per_year=Decimal("365"),
        ),
    ).run(candles, BuyOnce(instrument))
    assert financed.final_equity < baseline.final_equity


def test_summary_is_derived_from_equity_ledger(instrument, candles):
    result = EventDrivenBacktester(initial_cash=Decimal("1000")).run(
        candles,
        BuyThenSell(instrument),
    )
    summary = summarize_backtest(result, periods_per_year=252)
    assert summary.total_return == pytest.approx(0.003)
    assert summary.fill_count == 2
    assert summary.max_drawdown <= 0


def test_backtest_rejects_unsorted_candles(instrument, candles):
    with pytest.raises(ValueError):
        EventDrivenBacktester(initial_cash=Decimal("1000")).run(
            (candles[1], candles[0]),
            BuyOnce(instrument),
        )


def test_backtester_resets_broker_state_between_runs(instrument, candles):
    backtester = EventDrivenBacktester(initial_cash=Decimal("1000"))
    first = backtester.run(candles, BuyOnce(instrument))
    second = backtester.run(candles, BuyOnce(instrument))
    assert len(first.fills) == len(second.fills) == 1
    assert first.final_equity == second.final_equity


def test_unfilled_final_order_is_cancelled(instrument, candles):
    class LastBarOrder:
        def on_bar(self, candle, context):
            if context.event_index == len(candles) - 1:
                return (
                    OrderRequest(
                        instrument_id=instrument,
                        side="buy",
                        quantity=Decimal("1"),
                    ),
                )
            return ()

    result = EventDrivenBacktester(initial_cash=Decimal("1000")).run(candles, LastBarOrder())
    assert result.submitted_orders == 1
    assert result.cancelled_orders == 1
    assert len(result.fills) == 0
