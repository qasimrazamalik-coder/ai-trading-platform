from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

from quant_os_api.paper_session import PaperTradingSession
from quant_os_core.backtesting.models import OrderSide
from quant_os_core.execution import BrokerOrderRequest
from quant_os_core.instruments import InstrumentId

BTC = InstrumentId(venue="PAPER", symbol="BTCUSDT")


def test_concurrent_duplicate_submission_remains_idempotent() -> None:
    session = PaperTradingSession(fee_rate=Decimal("0"), slippage_bps=Decimal("0"))
    session.update_mark(BTC, Decimal("100"))
    request = BrokerOrderRequest(
        client_order_id="concurrent-same",
        instrument_id=BTC,
        side=OrderSide.BUY,
        quantity=Decimal("1"),
    )

    with ThreadPoolExecutor(max_workers=16) as pool:
        states = list(pool.map(lambda _: session.submit(request), range(100)))

    assert len({state.venue_order_id for state in states}) == 1
    assert len(session.orders()) == 1
    account = session.account()
    assert account.positions[0].quantity == Decimal("1")


def test_concurrent_unique_orders_preserve_accounting() -> None:
    session = PaperTradingSession(fee_rate=Decimal("0"), slippage_bps=Decimal("0"))
    session.update_mark(BTC, Decimal("100"))

    def submit(index: int) -> None:
        side = OrderSide.BUY if index % 2 == 0 else OrderSide.SELL
        session.submit(
            BrokerOrderRequest(
                client_order_id=f"concurrent-{index}",
                instrument_id=BTC,
                side=side,
                quantity=Decimal("1"),
            )
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(submit, range(200)))

    assert len(session.orders()) == 200
    assert session.account().positions == ()
    assert session.account().equity == Decimal("100000")
