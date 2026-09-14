from datetime import UTC, datetime
from decimal import Decimal
from random import Random
from uuid import uuid4

from quant_os_core.backtesting.models import Fill, OrderSide
from quant_os_core.backtesting.portfolio import Portfolio


def test_randomized_single_asset_accounting_identity(instrument):
    rng = Random(260914)
    portfolio = Portfolio(Decimal("100000"))
    total_fees = Decimal("0")

    for index in range(200):
        side = OrderSide.BUY if rng.random() >= 0.5 else OrderSide.SELL
        quantity = Decimal(str(rng.randint(1, 5)))
        price = Decimal(str(rng.randint(80, 120)))
        fee = Decimal("0.05")
        total_fees += fee
        fill = Fill(
            order_id=uuid4(),
            instrument_id=instrument,
            side=side,
            quantity=quantity,
            price=price,
            fee=fee,
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            bar_index=index,
        )
        portfolio.apply_fill(fill)
        mark = Decimal(str(rng.randint(80, 120)))
        portfolio.mark(instrument, mark)

        position = portfolio.position_for(instrument)
        expected_equity = portfolio.cash + position.quantity * mark
        assert portfolio.equity == expected_equity
        assert portfolio.gross_exposure == abs(position.quantity * mark)
        assert portfolio.net_exposure == position.quantity * mark

    assert portfolio.total_fees == total_fees
