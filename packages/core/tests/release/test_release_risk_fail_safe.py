from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from quant_os_core.backtesting.models import OrderSide, OrderStatus
from quant_os_core.events import OrderIntent, Side
from quant_os_core.execution import BrokerOrderRequest, OrderManagementSystem
from quant_os_core.execution.models import AccountSnapshot, BrokerOrderState
from quant_os_core.instruments import InstrumentId
from quant_os_core.risk_engine import RiskAction, RiskContext, RiskEngine

BTC = InstrumentId(venue="FAULT", symbol="BTCUSDT")


class DisconnectingBroker:
    def __init__(self) -> None:
        self.disconnected = False
        self.state: BrokerOrderState | None = None

    def submit(self, request: BrokerOrderRequest) -> BrokerOrderState:
        self.state = BrokerOrderState(
            client_order_id=request.client_order_id,
            venue_order_id="REMOTE-1",
            instrument_id=request.instrument_id,
            status=OrderStatus.PENDING,
            submitted_quantity=request.quantity,
            updated_at=datetime.now(UTC),
        )
        return self.state

    def cancel(self, client_order_id: str) -> BrokerOrderState:
        if self.state is None:
            raise KeyError(client_order_id)
        return self.state

    def order(self, client_order_id: str) -> BrokerOrderState | None:
        if self.disconnected:
            raise ConnectionError("broker offline")
        return self.state

    def open_orders(self) -> tuple[BrokerOrderState, ...]:
        if self.disconnected:
            raise ConnectionError("broker offline")
        return () if self.state is None else (self.state,)

    def account(self) -> AccountSnapshot:
        if self.disconnected:
            raise ConnectionError("broker offline")
        return AccountSnapshot(
            cash=Decimal("10000"),
            equity=Decimal("10000"),
            buying_power=Decimal("10000"),
            positions=(),
            timestamp=datetime.now(UTC),
        )


def test_oms_fails_closed_when_broker_disconnects_with_no_orders() -> None:
    broker = DisconnectingBroker()
    oms = OrderManagementSystem(broker)
    assert oms.safe_to_trade() is True
    broker.disconnected = True
    assert oms.safe_to_trade() is False
    issues = oms.reconcile_positions({})
    assert len(issues) == 1
    assert issues[0].category == "broker_unavailable"
    assert issues[0].severity == "critical"


def test_oms_fails_closed_when_reconciliation_disconnects() -> None:
    broker = DisconnectingBroker()
    oms = OrderManagementSystem(broker)
    oms.submit(
        BrokerOrderRequest(
            client_order_id="disconnect",
            instrument_id=BTC,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
        )
    )
    broker.disconnected = True
    issues = oms.reconcile_orders()
    assert len(issues) == 1
    assert issues[0].category == "broker_unavailable"
    assert issues[0].severity == "critical"
    assert oms.safe_to_trade() is False


def test_risk_engine_halts_on_stale_data_and_kill_switch() -> None:
    intent = OrderIntent(
        source="release-test",
        strategy_id="strategy",
        symbol="BTCUSDT",
        side=Side.BUY,
        quantity=Decimal("1"),
    )
    base = dict(
        equity=Decimal("10000"),
        peak_equity=Decimal("10000"),
        start_of_day_equity=Decimal("10000"),
        positions=(),
        mark_prices={"BTCUSDT": Decimal("100")},
    )
    stale = RiskEngine().evaluate(intent, RiskContext(**base, stale_market_data=True))
    killed = RiskEngine().evaluate(intent, RiskContext(**base, kill_switch=True))
    assert stale.action is RiskAction.HALT
    assert killed.action is RiskAction.HALT
    assert stale.approved_intent is None
    assert killed.approved_intent is None


def test_live_trading_requires_two_key_production_interlock() -> None:
    import pytest
    from pydantic import ValidationError

    from quant_os_core.settings import Settings

    assert Settings(_env_file=None, live_trading_enabled=False).live_trading_enabled is False
    with pytest.raises(ValidationError, match="explicit risk acknowledgement"):
        Settings(_env_file=None, env="development", live_trading_enabled=True)
    with pytest.raises(ValidationError, match="explicit risk acknowledgement"):
        Settings(_env_file=None, env="production", live_trading_enabled=True)
    armed = Settings(
        _env_file=None,
        env="production",
        live_trading_enabled=True,
        live_trading_ack="I_UNDERSTAND_REAL_MONEY_RISK",
    )
    assert armed.live_trading_enabled is True
