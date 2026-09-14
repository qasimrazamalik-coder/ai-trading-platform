from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from quant_os_core.backtesting.models import Fill, OrderSide
from quant_os_core.backtesting.portfolio import Portfolio


def make_fill(instrument, side, quantity, price, fee="0"):
    return Fill(
        order_id=uuid4(),
        instrument_id=instrument,
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        fee=Decimal(fee),
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        bar_index=0,
    )


def test_long_accounting_and_mark_to_market(instrument):
    portfolio = Portfolio(Decimal("1000"))
    portfolio.apply_fill(make_fill(instrument, OrderSide.BUY, "2", "100", "1"))
    portfolio.mark(instrument, Decimal("110"))
    assert portfolio.cash == Decimal("799")
    assert portfolio.equity == Decimal("1019")
    assert portfolio.gross_exposure == Decimal("220")


def test_realized_long_profit(instrument):
    portfolio = Portfolio(Decimal("1000"))
    portfolio.apply_fill(make_fill(instrument, OrderSide.BUY, "2", "100"))
    portfolio.apply_fill(make_fill(instrument, OrderSide.SELL, "1", "120"))
    position = portfolio.position_for(instrument)
    assert position.quantity == Decimal("1")
    assert position.realized_pnl == Decimal("20")


def test_short_profit_and_cover(instrument):
    portfolio = Portfolio(Decimal("1000"))
    portfolio.apply_fill(make_fill(instrument, OrderSide.SELL, "2", "100"))
    portfolio.mark(instrument, Decimal("90"))
    assert portfolio.equity == Decimal("1020")
    portfolio.apply_fill(make_fill(instrument, OrderSide.BUY, "2", "90"))
    position = portfolio.position_for(instrument)
    assert position.quantity == 0
    assert position.realized_pnl == Decimal("20")
    assert portfolio.equity == Decimal("1020")


def test_flip_long_to_short_sets_new_average(instrument):
    portfolio = Portfolio(Decimal("1000"))
    portfolio.apply_fill(make_fill(instrument, OrderSide.BUY, "1", "100"))
    portfolio.apply_fill(make_fill(instrument, OrderSide.SELL, "2", "110"))
    position = portfolio.position_for(instrument)
    assert position.quantity == Decimal("-1")
    assert position.average_price == Decimal("110")
    assert position.realized_pnl == Decimal("10")
