"""Paper loop for the diversified core in cost-guard mode (monthly rebalancing).

The core is an allocation, not a set of trade ideas: it has no stop or target, so it cannot
pass the plan and edge-over-cost stages of ``EntryPipeline``. The owner chose (see
``GUARDRAILS.md``, "Diversified core") to replace only those two stages with a cost guard:

* trade at most once per calendar month (until that month's rebalance completes);
* only when a fund is more than ``band`` off its target weight (or its target is zero);
* never an order under ``min_notional``.

Every order still passes the Shariah gate; every buy also passes the protections and the
risk engine. The kill switch, fail-closed broker breaker and position checks stay in force.
Run it in its own Alpaca paper account: any position outside the core universe pauses it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import ROUND_DOWN, Decimal
from functools import partial

from quant_os_core.agents.exits import Bar
from quant_os_core.backtesting.models import OrderSide, OrderStatus
from quant_os_core.events import OrderIntent, Side
from quant_os_core.execution.broker import ExecutionBroker
from quant_os_core.execution.models import AccountSnapshot, BrokerOrderRequest
from quant_os_core.ops.alerts import AlertSink
from quant_os_core.ops.orders import client_order_id
from quant_os_core.ops.paper_loop import StepReport, UniverseItem
from quant_os_core.ops.resilience import CircuitBreaker
from quant_os_core.ops.state import StateStore
from quant_os_core.risk_engine import ProtectionManager, RiskContext, RiskEngine, RiskPosition
from quant_os_core.shariah import ShariahGate, ShariahOrderContext
from quant_os_core.strategies.diversified_core import (
    CoreMode,
    RebalanceOrder,
    is_rebalance_day,
    rebalance_orders,
    target_weights,
)

_LAST_KEY = "core:last_rebalance"
_PENDING_KEY = "core:pending"
_QTY_STEP = Decimal("0.0001")
_OPEN = {OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED}
MISS_LIMIT = 3  # steps an order id may be unknown to the broker before it is dropped


@dataclass(slots=True)
class CoreRebalancer:
    broker: ExecutionBroker
    store: StateStore
    universe: Mapping[str, UniverseItem]
    shariah: ShariahGate
    protections: ProtectionManager
    risk: RiskEngine
    alerts: AlertSink
    mode: CoreMode = CoreMode.DEFENSIVE
    band: Decimal = Decimal("0.02")
    min_notional: Decimal = Decimal("1.00")
    cash_buffer: Decimal = Decimal("0.01")  # keep 1% of cash unspent for fill slippage
    flatten_on_kill: bool = True
    broker_breaker: CircuitBreaker = field(
        default_factory=lambda: CircuitBreaker("broker", failure_threshold=3)
    )

    # ---- helpers ----------------------------------------------------------------------
    def _alert(self, level: str, title: str, message: str, report: StepReport) -> None:
        self.store.log(level, "core", f"{title}: {message}")
        if level != "info":
            report.issues.append(f"{title}: {message}")
        self.alerts.send(level, title, message)

    def _account(self, report: StepReport) -> AccountSnapshot | None:
        try:
            return self.broker_breaker.call(self.broker.account)
        except Exception as exc:
            self._alert("critical", "broker unavailable", repr(exc), report)
            report.unknown_state = True
            return None

    def _pending_ids(self) -> list[str]:
        raw = self.store.get_value(_PENDING_KEY)
        return [cid for cid in raw.split(",") if cid]

    def _orders_still_open(self, report: StepReport) -> bool:
        still: list[str] = []
        for cid in self._pending_ids():
            try:
                state = self.broker_breaker.call(partial(self.broker.order, cid))
            except Exception as exc:
                self._alert("warning", "order lookup failed", f"{cid}: {exc!r}", report)
                still.append(cid)
                continue
            if state is None:
                # unknown to the broker: maybe a POST that timed out and never landed, maybe a
                # lookup glitch. Stay paused until the broker has denied it MISS_LIMIT times.
                misses = int(self.store.get_value(f"core:miss:{cid}", "0")) + 1
                self.store.set_value(f"core:miss:{cid}", str(misses))
                if misses < MISS_LIMIT:
                    still.append(cid)
                else:
                    self._alert("warning", "order never seen", f"{cid}: dropped", report)
            elif state.status in _OPEN:
                still.append(cid)
            elif state.status is OrderStatus.FILLED:
                report.fills.append(cid)
        self.store.set_value(_PENDING_KEY, ",".join(still))
        return bool(still)

    def _submit(
        self, symbol: str, side: OrderSide, quantity: Decimal, now: datetime, report: StepReport
    ) -> bool:
        item = self.universe[symbol]
        request = BrokerOrderRequest(
            client_order_id=client_order_id(symbol, side, now),
            instrument_id=item.instrument.instrument_id,
            side=side,
            quantity=quantity,
        )
        # record the intent first: a POST that times out may still create a live order, and
        # the next step must not trade again until the broker says what happened to it
        self.store.set_value(
            _PENDING_KEY, ",".join([*self._pending_ids(), request.client_order_id])
        )
        try:
            self.broker_breaker.call(partial(self.broker.submit, request))
        except Exception as exc:
            self._alert("critical", "order failed", f"{side.value} {symbol}: {exc!r}", report)
            return False
        return True

    # ---- the step ---------------------------------------------------------------------
    def step(self, now: datetime, bars: Mapping[str, Sequence[Bar]]) -> StepReport:
        report = StepReport(now)
        if self.store.kill_switch_on():
            report.killed = True
            for cid in self._pending_ids():
                try:
                    self.broker.cancel(cid)
                except Exception as exc:
                    self._alert("warning", "cancel failed", repr(exc), report)
            # an order whose cancel failed may still fill: flatten only once none is open,
            # or the flatten could sell shares a live order is already selling
            if self._orders_still_open(report):
                return report
            account = self._account(report)
            if self.flatten_on_kill and account is not None:
                for position in account.positions:
                    symbol = position.instrument_id.symbol
                    if (
                        symbol in self.universe
                        and position.quantity > 0
                        and self._submit(symbol, OrderSide.SELL, position.quantity, now, report)
                    ):
                        report.exits.append(symbol)
            return report

        account = self._account(report)
        if account is None:
            return report
        held = {p.instrument_id.symbol: p.quantity for p in account.positions}
        strangers = sorted(s for s, q in held.items() if q != 0 and s not in self.universe)
        shorts = sorted(s for s, q in held.items() if q < 0)
        if strangers or shorts or account.cash < 0:
            problem = (
                (f"not in the core universe: {strangers}; " if strangers else "")
                + (f"short: {shorts}; " if shorts else "")
                + (f"negative cash {account.cash}" if account.cash < 0 else "")
            )
            self._alert("critical", "unexpected account state", f"{problem} core paused", report)
            report.unknown_state = True
            return report
        if self._orders_still_open(report):
            self._record(now, report)
            return report

        last_raw = self.store.get_value(_LAST_KEY)
        last = date.fromisoformat(last_raw) if last_raw else None
        if self.broker_breaker.healthy and is_rebalance_day(now.date(), last):
            self._rebalance(now, account, held, bars, report)
        self._record(now, report)
        return report

    def _rebalance(
        self,
        now: datetime,
        account: AccountSnapshot,
        held: Mapping[str, Decimal],
        bars: Mapping[str, Sequence[Bar]],
        report: StepReport,
    ) -> None:
        missing = [s for s in self.universe if not bars.get(s)]
        if missing:  # fail closed: weights over a partial universe would be wrong
            self._alert("warning", "rebalance skipped", f"no bars for {missing}", report)
            return
        prices = {s: bars[s][-1].close for s in self.universe}
        try:
            targets = target_weights(
                {s: [float(b.close) for b in bars[s]] for s in self.universe}, self.mode
            )
        except ValueError as exc:
            self._alert("warning", "rebalance skipped", str(exc), report)
            return
        holdings = {s: held.get(s, Decimal(0)) * prices[s] for s in self.universe}
        orders = rebalance_orders(
            targets,
            holdings,
            account.buying_power * (1 - self.cash_buffer),  # slack for fills above last close
            band=self.band,
            min_notional=self.min_notional,
        )
        sold = [o.symbol for o in orders if o.side == "sell"]
        for order in orders:
            if order.side == "sell":
                self._sell(order, prices, held, targets, now, report)
        if any(s in report.exits for s in sold):  # buy next step, once proceeds are cash
            return
        failed = 0
        for order in orders:
            if order.side == "buy" and not self._buy(order, prices, held, account, now, report):
                failed += 1
        # gate blocks are final for the month; a failed submit is retried on the next step
        if failed == 0:
            self.store.set_value(_LAST_KEY, now.date().isoformat())

    def _sell(
        self,
        order: RebalanceOrder,
        prices: Mapping[str, Decimal],
        held: Mapping[str, Decimal],
        targets: Mapping[str, Decimal],
        now: datetime,
        report: StepReport,
    ) -> None:
        owned = held.get(order.symbol, Decimal(0))
        if targets.get(order.symbol, Decimal(0)) == 0:
            quantity = owned
        else:
            quantity = (order.notional / prices[order.symbol] / _QTY_STEP).to_integral_value(
                ROUND_DOWN
            ) * _QTY_STEP
            quantity = min(quantity, owned)
        if quantity <= 0:
            return
        item = self.universe[order.symbol]
        verdict = self.shariah.check_order(
            item.instrument,
            is_sell=True,
            quantity=quantity,
            notional=quantity * prices[order.symbol],
            context=ShariahOrderContext(
                owned_quantity=owned, free_cash=Decimal(0), settled_quantity=owned
            ),
            financials=item.financials,
            crypto_asset=item.crypto_asset,
        )
        self._decide(now, order.symbol, quantity, [("shariah", verdict.allowed, verdict.reason)])
        if not verdict.allowed:
            report.blocked[order.symbol] = f"shariah {verdict.rule_id}"
            return
        if self._submit(order.symbol, OrderSide.SELL, quantity, now, report):
            report.exits.append(order.symbol)

    def _buy(
        self,
        order: RebalanceOrder,
        prices: Mapping[str, Decimal],
        held: Mapping[str, Decimal],
        account: AccountSnapshot,
        now: datetime,
        report: StepReport,
    ) -> bool:
        """True when the buy was sent or finally blocked; False when the submit failed."""

        symbol, price = order.symbol, prices[order.symbol]
        quantity = (order.notional / price / _QTY_STEP).to_integral_value(ROUND_DOWN) * _QTY_STEP
        if quantity <= 0:
            return True
        item = self.universe[symbol]
        stages: list[tuple[str, bool, str]] = [
            ("cost_guard", True, f"monthly, drift > {self.band:.0%}, >= ${self.min_notional}")
        ]
        verdict = self.shariah.check_order(
            item.instrument,
            is_sell=False,
            quantity=quantity,
            notional=quantity * price,
            context=ShariahOrderContext(
                owned_quantity=held.get(symbol, Decimal(0)), free_cash=account.buying_power
            ),
            financials=item.financials,
            crypto_asset=item.crypto_asset,
        )
        stages.append(("shariah", verdict.allowed, f"{verdict.rule_id}: {verdict.reason}"))
        if not verdict.allowed:
            self._block(now, symbol, quantity, stages, f"shariah {verdict.rule_id}", report)
            return True
        lock = self.protections.blocking_lock(symbol, self.store.closed_trades(), now)
        stages.append(("protections", lock is None, "no active locks" if lock is None else ""))
        if lock is not None:
            reason = f"{lock.protection}: {lock.reason}"
            self._block(now, symbol, quantity, stages, reason, report)
            return True
        marks = {s: prices[s] for s in held if s in prices}
        peak = max([account.equity, *(e for _, e in self.store.equity_curve(500))])
        evaluation = self.risk.evaluate(
            OrderIntent(
                source="diversified-core",
                strategy_id="diversified-core",
                symbol=symbol,
                side=Side.BUY,
                quantity=quantity,
                limit_price=price,
            ),
            RiskContext(
                equity=account.equity,
                peak_equity=peak,
                start_of_day_equity=self._start_of_day_equity(now, account.equity),
                positions=tuple(
                    RiskPosition(s, q, marks[s]) for s, q in held.items() if q > 0 and s in marks
                ),
                mark_prices={**marks, symbol: price},
                stop_prices={},
            ),
        )
        detail = "; ".join(evaluation.reasons) or str(evaluation.action)
        stages.append(("risk", evaluation.approved, detail))
        if not evaluation.approved:
            self._block(now, symbol, quantity, stages, "risk", report)
            return True
        approved = evaluation.approved_quantity
        self._decide(now, symbol, approved, stages)
        if approved <= 0:
            return True
        if not self._submit(symbol, OrderSide.BUY, approved, now, report):
            return False
        report.entries_submitted.append(symbol)
        return True

    def _start_of_day_equity(self, now: datetime, equity: Decimal) -> Decimal:
        """First equity seen each day, so the daily-loss halt can fire for core buys."""

        key = f"core:sod:{now.date().isoformat()}"
        stored = self.store.get_value(key)
        if stored:
            return Decimal(stored)
        self.store.set_value(key, str(equity))
        return equity

    def _block(
        self,
        now: datetime,
        symbol: str,
        quantity: Decimal,
        stages: list[tuple[str, bool, str]],
        reason: str,
        report: StepReport,
    ) -> None:
        self._decide(now, symbol, quantity, stages)
        report.blocked[symbol] = reason

    def _decide(
        self, now: datetime, symbol: str, quantity: Decimal, stages: list[tuple[str, bool, str]]
    ) -> None:
        blocked = next((name for name, ok, _ in stages if not ok), None)
        self.store.record_decision(
            now,
            symbol,
            blocked is None,
            blocked,
            quantity if blocked is None else Decimal(0),
            [{"stage": n, "passed": ok, "detail": d} for n, ok, d in stages],
        )

    def _record(self, now: datetime, report: StepReport) -> None:
        account = self._account(report)
        if account is not None:
            self.store.record_equity(now, account.equity, account.cash)
