from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import numpy as np
import pytest

from quant_os_core.backtesting.analytics import equity_returns, summarize_backtest
from quant_os_core.backtesting.benchmark import compare_to_benchmark
from quant_os_core.backtesting.broker import BrokerConfig, SimulatedBroker
from quant_os_core.backtesting.costs import FinancingModel, ProportionalFeeModel
from quant_os_core.backtesting.engine import EventDrivenBacktester
from quant_os_core.backtesting.models import (
    BacktestResult,
    EquityPoint,
    Fill,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    SimulatedOrder,
)
from quant_os_core.backtesting.portfolio import Portfolio, Position
from quant_os_core.backtesting.slippage import BpsSlippageModel
from quant_os_core.backtesting.vectorized import run_vectorized_backtest
from quant_os_core.instruments import InstrumentId


def point(equity: str) -> EquityPoint:
    return EquityPoint(
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        cash=Decimal(equity),
        equity=Decimal(equity),
        gross_exposure=Decimal("0"),
        net_exposure=Decimal("0"),
    )


def result_with_equity(*values: str) -> BacktestResult:
    return BacktestResult(
        initial_cash=Decimal(values[0]),
        final_equity=Decimal(values[-1]),
        equity_curve=tuple(point(value) for value in values),
        fills=(),
        submitted_orders=0,
        rejected_orders=0,
        cancelled_orders=0,
    )


def test_analytics_single_point_and_invalid_equity():
    one = result_with_equity("1000")
    assert equity_returns(one).size == 0
    assert summarize_backtest(one).sharpe is None

    invalid = result_with_equity("1000", "0")
    with pytest.raises(ValueError):
        equity_returns(invalid)


def test_flat_analytics_reports_undefined_ratios():
    summary = summarize_backtest(result_with_equity("1000", "1000", "1000"))
    assert summary.sharpe is None
    assert summary.sortino is None
    assert summary.calmar is None


def test_benchmark_guards_and_single_point():
    one = result_with_equity("1000")
    comparison = compare_to_benchmark(one, [1000])
    assert comparison.beta is None
    assert comparison.benchmark_total_return == 0

    two = result_with_equity("1000", "1001")
    with pytest.raises(ValueError):
        compare_to_benchmark(two, [1000, 0])
    with pytest.raises(ValueError):
        compare_to_benchmark(two, [[1000, 1001]])


def test_broker_config_and_cost_guards():
    with pytest.raises(ValueError):
        BrokerConfig(latency_bars=0)
    with pytest.raises(ValueError):
        BrokerConfig(max_bar_volume_fraction=Decimal("0"))
    with pytest.raises(ValueError):
        ProportionalFeeModel(taker_bps=Decimal("-1"))
    with pytest.raises(ValueError):
        ProportionalFeeModel(maker_bps=Decimal("-1"))
    with pytest.raises(ValueError):
        ProportionalFeeModel().calculate(notional=Decimal("-1"))
    with pytest.raises(ValueError):
        FinancingModel(periods_per_year=Decimal("0"))
    assert FinancingModel().charge(signed_notional=Decimal("0")) == 0


def test_order_and_fill_helpers(instrument):
    with pytest.raises(ValueError):
        OrderRequest(
            instrument_id=instrument,
            side="buy",
            quantity=Decimal("1"),
            order_type="stop",
        )
    with pytest.raises(ValueError):
        OrderRequest(
            instrument_id=instrument,
            side="buy",
            quantity=Decimal("1"),
            stop_price=Decimal("99"),
        )

    request = OrderRequest(instrument_id=instrument, side="buy", quantity=Decimal("2"))
    order = SimulatedOrder(
        request=request,
        submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
        eligible_bar_index=1,
        remaining_quantity=Decimal("1"),
    )
    assert order.filled_quantity == Decimal("1")

    fill = Fill(
        order_id=uuid4(),
        instrument_id=instrument,
        side=OrderSide.SELL,
        quantity=Decimal("2"),
        price=Decimal("10"),
        fee=Decimal("0"),
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        bar_index=1,
    )
    assert fill.signed_quantity == Decimal("-2")
    assert fill.notional == Decimal("20")


def test_portfolio_guards_and_snapshots(instrument):
    with pytest.raises(ValueError):
        Portfolio(Decimal("0"))
    portfolio = Portfolio(Decimal("1000"))
    assert Position(instrument).unrealized_pnl(None) == 0
    with pytest.raises(ValueError):
        portfolio.apply_financing(Decimal("-1"))
    with pytest.raises(ValueError):
        portfolio.mark(instrument, Decimal("0"))
    assert portfolio.snapshots() == ()


def test_slippage_guards():
    with pytest.raises(ValueError):
        BpsSlippageModel(fixed_bps=Decimal("-1"))
    model = BpsSlippageModel()
    with pytest.raises(ValueError):
        model.adjust(
            reference_price=Decimal("0"),
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            bar_volume=Decimal("1"),
        )
    with pytest.raises(ValueError):
        BpsSlippageModel(fixed_bps=Decimal("20000")).adjust(
            reference_price=Decimal("100"),
            side=OrderSide.SELL,
            quantity=Decimal("1"),
            bar_volume=Decimal("1"),
        )


def test_vectorized_shape_and_finite_guards():
    with pytest.raises(ValueError):
        run_vectorized_backtest([[1, 2]], [1, 2])
    with pytest.raises(ValueError):
        run_vectorized_backtest([1], [1])
    with pytest.raises(ValueError):
        run_vectorized_backtest([1, np.nan], [1, 1])
    with pytest.raises(ValueError):
        run_vectorized_backtest([1, 2], [1, 1], signal_delay=-1)
    result = run_vectorized_backtest([1, 2, 3], [1, 1, 1], signal_delay=99)
    assert np.all(result.applied_positions == 0)


def test_event_engine_guards(instrument, candles):
    class Noop:
        def on_bar(self, candle, context):
            return ()

    engine = EventDrivenBacktester(initial_cash=Decimal("1000"))
    with pytest.raises(ValueError):
        engine.run((), Noop())

    altered = candles[1].model_copy(update={"timeframe": "1h"})
    with pytest.raises(ValueError):
        engine.run((candles[0], altered), Noop())
