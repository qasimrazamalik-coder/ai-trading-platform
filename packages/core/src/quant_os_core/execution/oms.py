"""Order-management and reconciliation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from quant_os_core.backtesting.models import OrderStatus
from quant_os_core.execution.broker import ExecutionBroker
from quant_os_core.execution.models import BrokerOrderRequest, BrokerOrderState


@dataclass(frozen=True, slots=True)
class ReconciliationIssue:
    category: str
    key: str
    expected: str
    actual: str
    severity: str


class OrderManagementSystem:
    def __init__(self, broker: ExecutionBroker) -> None:
        self._broker = broker
        self._local: dict[str, BrokerOrderState] = {}

    def submit(self, request: BrokerOrderRequest) -> BrokerOrderState:
        state = self._broker.submit(request)
        self._local[request.client_order_id] = state
        return state

    def cancel(self, client_order_id: str) -> BrokerOrderState:
        state = self._broker.cancel(client_order_id)
        self._local[client_order_id] = state
        return state

    def refresh(self, client_order_id: str) -> BrokerOrderState:
        remote = self._broker.order(client_order_id)
        if remote is None:
            raise KeyError(client_order_id)
        self._local[client_order_id] = remote
        return remote

    @property
    def local_orders(self) -> tuple[BrokerOrderState, ...]:
        return tuple(self._local.values())

    def reconcile_orders(self) -> tuple[ReconciliationIssue, ...]:
        issues: list[ReconciliationIssue] = []
        for client_id, local in self._local.items():
            try:
                remote = self._broker.order(client_id)
            except Exception as exc:
                issues.append(
                    ReconciliationIssue(
                        "broker_unavailable",
                        client_id,
                        local.status.value,
                        type(exc).__name__,
                        "critical",
                    )
                )
                continue
            if remote is None:
                issues.append(
                    ReconciliationIssue(
                        "missing_remote_order",
                        client_id,
                        local.status.value,
                        "missing",
                        "high",
                    )
                )
                continue
            if remote.status != local.status:
                issues.append(
                    ReconciliationIssue(
                        "order_status_mismatch",
                        client_id,
                        local.status.value,
                        remote.status.value,
                        "high",
                    )
                )
            if remote.filled_quantity != local.filled_quantity:
                issues.append(
                    ReconciliationIssue(
                        "fill_quantity_mismatch",
                        client_id,
                        str(local.filled_quantity),
                        str(remote.filled_quantity),
                        "critical",
                    )
                )
        return tuple(issues)

    def reconcile_positions(
        self,
        expected_quantities: dict[str, Decimal],
        *,
        tolerance: Decimal = Decimal("0"),
    ) -> tuple[ReconciliationIssue, ...]:
        if tolerance < 0:
            raise ValueError("tolerance cannot be negative")
        try:
            account = self._broker.account()
        except Exception as exc:
            return (
                ReconciliationIssue(
                    "broker_unavailable",
                    "account",
                    "reachable",
                    type(exc).__name__,
                    "critical",
                ),
            )
        remote = {p.instrument_id.key: p.quantity for p in account.positions}
        keys = set(expected_quantities) | set(remote)
        issues: list[ReconciliationIssue] = []
        for key in sorted(keys):
            expected = expected_quantities.get(key, Decimal("0"))
            actual = remote.get(key, Decimal("0"))
            if abs(expected - actual) > tolerance:
                issues.append(
                    ReconciliationIssue(
                        "position_mismatch", key, str(expected), str(actual), "critical"
                    )
                )
        return tuple(issues)

    def safe_to_trade(self) -> bool:
        critical = {"critical", "high"}
        if any(issue.severity in critical for issue in self.reconcile_orders()):
            return False
        try:
            self._broker.account()
        except Exception:
            return False
        return True

    def terminal_orders(self) -> tuple[BrokerOrderState, ...]:
        terminal = {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}
        return tuple(state for state in self._local.values() if state.status in terminal)
