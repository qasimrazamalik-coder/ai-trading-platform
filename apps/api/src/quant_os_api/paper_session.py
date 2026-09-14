"""Thread-safe in-memory paper execution session for the local API."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from threading import RLock

from quant_os_core.execution import BrokerOrderRequest, PaperBroker
from quant_os_core.execution.models import AccountSnapshot, BrokerOrderState, ExecutionFill
from quant_os_core.instruments import InstrumentId


class PaperTradingSession:
    """Own a deterministic paper broker behind a process-local lock.

    The session is deliberately in-memory. It is suitable for tutorials, UI verification,
    and paper execution on a single API process. Durable multi-process paper trading belongs
    behind the persistent OMS/event-store layer rather than this convenience surface.
    """

    def __init__(
        self,
        *,
        initial_cash: Decimal = Decimal("100000"),
        fee_rate: Decimal = Decimal("0.0005"),
        slippage_bps: Decimal = Decimal("1"),
    ) -> None:
        self._initial_cash = initial_cash
        self._fee_rate = fee_rate
        self._slippage_bps = slippage_bps
        self._lock = RLock()
        self._broker = self._new_broker()

    def _new_broker(self) -> PaperBroker:
        return PaperBroker(
            initial_cash=self._initial_cash,
            fee_rate=self._fee_rate,
            slippage_bps=self._slippage_bps,
        )

    def reset(self) -> None:
        with self._lock:
            self._broker = self._new_broker()

    def update_mark(
        self,
        instrument_id: InstrumentId,
        price: Decimal,
        *,
        timestamp: datetime | None = None,
    ) -> tuple[ExecutionFill, ...]:
        with self._lock:
            return self._broker.update_mark(instrument_id, price, timestamp=timestamp)

    def submit(self, request: BrokerOrderRequest) -> BrokerOrderState:
        with self._lock:
            return self._broker.submit(request)

    def cancel(self, client_order_id: str) -> BrokerOrderState:
        with self._lock:
            return self._broker.cancel(client_order_id)

    def order(self, client_order_id: str) -> BrokerOrderState | None:
        with self._lock:
            return self._broker.order(client_order_id)

    def orders(self) -> tuple[BrokerOrderState, ...]:
        with self._lock:
            return self._broker.orders()

    def account(self) -> AccountSnapshot:
        with self._lock:
            return self._broker.account()
