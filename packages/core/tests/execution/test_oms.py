from decimal import Decimal

from quant_os_core.backtesting.models import OrderSide, OrderType
from quant_os_core.execution import BrokerOrderRequest, OrderManagementSystem, PaperBroker
from quant_os_core.instruments import InstrumentId

BTC = InstrumentId(venue="PAPER", symbol="BTCUSDT")


def test_oms_refresh_and_position_reconciliation() -> None:
    broker = PaperBroker(fee_rate=Decimal("0"), slippage_bps=Decimal("0"))
    broker.update_mark(BTC, Decimal("100"))
    oms = OrderManagementSystem(broker)
    oms.submit(
        BrokerOrderRequest(
            client_order_id="a",
            instrument_id=BTC,
            side=OrderSide.BUY,
            quantity=Decimal("2"),
        )
    )
    assert oms.reconcile_orders() == ()
    assert oms.reconcile_positions({BTC.key: Decimal("2")}) == ()
    issues = oms.reconcile_positions({BTC.key: Decimal("1")})
    assert len(issues) == 1
    assert issues[0].category == "position_mismatch"
    assert issues[0].severity == "critical"


def test_tolerance_and_terminal_orders() -> None:
    broker = PaperBroker(fee_rate=Decimal("0"), slippage_bps=Decimal("0"))
    broker.update_mark(BTC, Decimal("100"))
    oms = OrderManagementSystem(broker)
    oms.submit(
        BrokerOrderRequest(
            client_order_id="a",
            instrument_id=BTC,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
        )
    )
    assert len(oms.terminal_orders()) == 1
    assert oms.reconcile_positions({BTC.key: Decimal("1.001")}, tolerance=Decimal("0.01")) == ()


def test_catalog_is_explicitly_live_disabled() -> None:
    from quant_os_core.execution import default_adapter_catalog

    catalog = default_adapter_catalog()
    assert {item.adapter_id for item in catalog} >= {"paper", "psx_broker", "ibkr_style_global"}
    assert all(not item.live_trading for item in catalog)


def test_oms_detects_remote_status_and_fill_drift() -> None:
    from datetime import UTC, datetime

    from quant_os_core.backtesting.models import OrderStatus
    from quant_os_core.execution.models import AccountSnapshot, BrokerOrderState

    class DriftBroker:
        def __init__(self) -> None:
            self.state: BrokerOrderState | None = None

        def submit(self, request: BrokerOrderRequest) -> BrokerOrderState:
            self.state = BrokerOrderState(
                client_order_id=request.client_order_id,
                venue_order_id="REMOTE",
                instrument_id=request.instrument_id,
                status=OrderStatus.PENDING,
                submitted_quantity=request.quantity,
                updated_at=datetime.now(UTC),
            )
            return self.state

        def cancel(self, client_order_id: str) -> BrokerOrderState:
            assert self.state is not None
            self.state = self.state.model_copy(update={"status": OrderStatus.CANCELLED})
            return self.state

        def order(self, client_order_id: str) -> BrokerOrderState | None:
            return self.state

        def open_orders(self) -> tuple[BrokerOrderState, ...]:
            return () if self.state is None else (self.state,)

        def account(self) -> AccountSnapshot:
            return AccountSnapshot(
                cash=Decimal("100"),
                equity=Decimal("100"),
                buying_power=Decimal("100"),
                positions=(),
                timestamp=datetime.now(UTC),
            )

    broker = DriftBroker()
    oms = OrderManagementSystem(broker)
    oms.submit(
        BrokerOrderRequest(
            client_order_id="drift",
            instrument_id=BTC,
            side=OrderSide.BUY,
            quantity=Decimal("2"),
        )
    )
    assert broker.state is not None
    broker.state = broker.state.model_copy(
        update={"status": OrderStatus.PARTIALLY_FILLED, "filled_quantity": Decimal("1")}
    )
    issues = oms.reconcile_orders()
    assert {issue.category for issue in issues} == {
        "order_status_mismatch",
        "fill_quantity_mismatch",
    }
    assert oms.safe_to_trade() is False
    refreshed = oms.refresh("drift")
    assert refreshed.filled_quantity == Decimal("1")


def test_oms_cancel_refresh_and_bad_tolerance() -> None:
    import pytest

    broker = PaperBroker(fee_rate=Decimal("0"), slippage_bps=Decimal("0"))
    broker.update_mark(BTC, Decimal("100"))
    oms = OrderManagementSystem(broker)
    oms.submit(
        BrokerOrderRequest(
            client_order_id="pending-limit",
            instrument_id=BTC,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("90"),
        )
    )
    assert oms.cancel("pending-limit").status.value == "cancelled"
    with pytest.raises(KeyError):
        oms.refresh("missing")
    with pytest.raises(ValueError):
        oms.reconcile_positions({}, tolerance=Decimal("-0.1"))
