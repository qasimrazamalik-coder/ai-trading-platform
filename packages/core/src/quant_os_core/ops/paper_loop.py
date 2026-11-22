"""Paper-trading loop: the full decision path against a broker, with durable state.

One ``step`` per bar (e.g. every 5 minutes for day trading, once a day for swing):

1. kill switch -> cancel pending entries, optionally flatten, stop;
2. account snapshot through a circuit breaker; any failure -> no new entries (fail closed);
3. reconcile stored positions with the broker; a mismatch -> no new entries, critical alert;
4. pending entries: fills become positions, stale orders are cancelled;
5. exits: the exit manager on the latest bar; exits are market sells;
6. entries: signal -> research tasks for the analyst board -> pipeline -> patient limit order;
7. equity snapshot and a once-a-day summary alert.

Works with the deterministic ``PaperBroker`` (offline tests) or ``AlpacaPaperBroker``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from functools import partial

from quant_os_core.agents.board import AgentBoard, AnalystRole
from quant_os_core.agents.day_trading import Market, flatten_time, settlement_time
from quant_os_core.agents.exits import Bar, ExitManager, OpenPosition
from quant_os_core.agents.pipeline import EntryPipeline
from quant_os_core.agents.simulator import ExitPolicy
from quant_os_core.agents.trade_plan import TradePlan
from quant_os_core.backtesting.models import OrderStatus
from quant_os_core.execution.broker import ExecutionBroker
from quant_os_core.execution.models import AccountSnapshot
from quant_os_core.instruments import Instrument
from quant_os_core.ops.alerts import AlertSink
from quant_os_core.ops.ledger import Ledger, record_fill_safely
from quant_os_core.ops.orders import entry_order, exit_order
from quant_os_core.ops.resilience import CircuitBreaker
from quant_os_core.ops.state import StateStore
from quant_os_core.risk_engine import ClosedTrade, ExitReason, RiskContext, RiskPosition
from quant_os_core.shariah import CompanyFinancials, CryptoAsset, ShariahOrderContext

Signal = Callable[[str, Sequence[Bar]], TradePlan | None]
DEFAULT_REVIEW_ROLES = (
    AnalystRole.NEWS_SENTIMENT,
    AnalystRole.RISK_OFFICER,
    AnalystRole.BEAR_ADVOCATE,
)


@dataclass(frozen=True, slots=True)
class UniverseItem:
    instrument: Instrument
    financials: CompanyFinancials | None = None
    crypto_asset: CryptoAsset | None = None
    market: Market = Market.US


@dataclass(slots=True)
class StepReport:
    time: datetime
    killed: bool = False
    unknown_state: bool = False
    entries_submitted: list[str] = field(default_factory=list)
    fills: list[str] = field(default_factory=list)
    exits: list[str] = field(default_factory=list)
    blocked: dict[str, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PaperTrader:
    broker: ExecutionBroker
    store: StateStore
    pipeline: EntryPipeline
    universe: Mapping[str, UniverseItem]
    signal: Signal
    alerts: AlertSink
    exit_policy: ExitPolicy = field(default_factory=ExitPolicy)
    board: AgentBoard | None = None
    review_roles: tuple[AnalystRole, ...] = DEFAULT_REVIEW_ROLES
    pending_timeout: timedelta = timedelta(minutes=30)
    flatten_on_kill: bool = True
    quantity_tolerance: Decimal = Decimal("0.000001")
    broker_breaker: CircuitBreaker = field(
        default_factory=lambda: CircuitBreaker("broker", failure_threshold=3)
    )
    ledger: Ledger | None = None  # trade and fee ledger; fills are recorded automatically
    _exits: ExitManager = field(default_factory=ExitManager)

    # ---- helpers ----------------------------------------------------------------------
    def _alert(self, level: str, title: str, message: str, report: StepReport) -> None:
        self.store.log(level, "loop", f"{title}: {message}")
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

    def _reconcile(self, account: AccountSnapshot, report: StepReport) -> set[str]:
        """Symbols whose quantity differs between the bot and the broker."""

        broker_qty = {p.instrument_id.symbol: p.quantity for p in account.positions}
        mismatched: set[str] = set()
        stored = {s.position.plan.symbol: s.position.quantity for s in self.store.open_positions()}
        for symbol in set(broker_qty) | set(stored):
            ours, theirs = stored.get(symbol, Decimal("0")), broker_qty.get(symbol, Decimal("0"))
            if abs(ours - theirs) > self.quantity_tolerance:
                mismatched.add(symbol)
                self._alert(
                    "critical",
                    "position mismatch",
                    f"{symbol}: bot thinks {ours}, broker has {theirs}; new entries paused",
                    report,
                )
        report.unknown_state = report.unknown_state or bool(mismatched)
        return mismatched

    def _refresh_pending(self, now: datetime, report: StepReport) -> None:
        for item in self.store.pending():
            cid, plan = item["client_order_id"], item["plan"]
            try:
                state = self.broker_breaker.call(partial(self.broker.order, cid))
            except Exception as exc:
                self._alert("warning", "order lookup failed", f"{cid}: {exc!r}", report)
                continue
            if state is None:
                self.store.remove_pending(cid)
                continue
            if state.status is OrderStatus.FILLED:
                universe = self.universe[plan.symbol]
                asset = universe.instrument.asset_class
                strict = not self.pipeline.shariah.profile.allow_pre_settlement_resale
                fill_time = now
                position = OpenPosition(
                    plan,
                    state.filled_quantity,
                    fill_time,
                    rules=self.exit_policy.rules,
                    trail_fraction=self.exit_policy.trail_fraction,
                    max_holding=self.exit_policy.max_holding,
                    flatten_at=(
                        flatten_time(asset, fill_time, market=universe.market)
                        if self.exit_policy.flatten_before_close
                        else None
                    ),
                    earliest_exit=(
                        settlement_time(asset, fill_time, market=universe.market)
                        if strict
                        else None
                    ),
                )
                self.store.save_position(position, asset.value)
                self.store.remove_pending(cid)
                report.fills.append(plan.symbol)
                price = state.average_fill_price or plan.entry_price
                record_fill_safely(
                    self.ledger,
                    self.store.log,
                    now,
                    plan.symbol,
                    "buy",
                    state.filled_quantity,
                    price,
                )
                self._alert(
                    "info", "filled", f"BUY {state.filled_quantity} {plan.symbol} @ {price}", report
                )
            elif state.status in {OrderStatus.CANCELLED, OrderStatus.REJECTED}:
                self.store.remove_pending(cid)
            elif now - item["submitted"] >= self.pending_timeout:
                try:
                    self.broker.cancel(cid)
                finally:
                    self.store.remove_pending(cid)
                self.store.log("info", "loop", f"cancelled stale entry {cid}")

    def _close(
        self, position: OpenPosition, price: Decimal, trade: ClosedTrade, report: StepReport
    ) -> None:
        item = self.universe.get(position.plan.symbol)
        if item is None:
            self._alert("critical", "unknown symbol", position.plan.symbol, report)
            return
        request = exit_order(
            position.plan.symbol, item.instrument.instrument_id, position.quantity, trade.exit_time
        )
        try:
            state = self.broker_breaker.call(lambda: self.broker.submit(request))
        except Exception as exc:
            self._alert("critical", "exit failed", f"{position.plan.symbol}: {exc!r}", report)
            return
        fill = state.average_fill_price or price
        record_fill_safely(
            self.ledger,
            self.store.log,
            trade.exit_time,
            position.plan.symbol,
            "sell",
            position.quantity,
            fill,
        )
        profit = fill / position.plan.entry_price - Decimal("1")
        closed = ClosedTrade(trade.symbol, trade.exit_time, profit, trade.exit_reason)
        self.store.record_close(closed, position.quantity, position.plan.entry_price, fill)
        self.store.remove_position(position.plan.symbol)
        if self.board is not None:  # grade the analysts who called this trade
            self.board.grade_trade(position.plan.symbol, position.entry_time, profit)
        report.exits.append(position.plan.symbol)
        self._alert(
            "info" if profit >= 0 else "warning",
            "closed",
            f"SELL {position.quantity} {trade.symbol} @ {fill} ({profit:+.2%}, "
            f"{trade.exit_reason.value})",
            report,
        )

    # ---- the step ---------------------------------------------------------------------
    def step(self, now: datetime, bars: Mapping[str, Sequence[Bar]]) -> StepReport:
        report = StepReport(now)
        if self.store.kill_switch_on():
            report.killed = True
            for item in self.store.pending():
                try:
                    self.broker.cancel(item["client_order_id"])
                except Exception as exc:
                    self._alert("warning", "cancel failed", repr(exc), report)
                self.store.remove_pending(item["client_order_id"])
            if self.flatten_on_kill:
                for stored in self.store.open_positions():
                    series = bars.get(stored.position.plan.symbol) or ()
                    price = series[-1].close if series else stored.position.plan.entry_price
                    trade = ClosedTrade(
                        stored.position.plan.symbol, now, Decimal("0"), ExitReason.SIGNAL
                    )
                    self._close(stored.position, price, trade, report)
            return report

        account = self._account(report)
        if account is None:
            return report
        self._refresh_pending(now, report)
        mismatched = self._reconcile(account, report)
        healthy = not mismatched

        for stored in self.store.open_positions():
            position = stored.position
            series = bars.get(position.plan.symbol) or ()
            if not series:
                continue
            closes = tuple(b.close for b in series)
            event = self._exits.on_bar(position, series[-1], closes)
            # Exits still run for positions the broker agrees on: a known stop must fire.
            if event is not None and position.plan.symbol not in mismatched:
                self._close(position, event.price, event.trade, report)
            else:
                self.store.save_position(position, stored.asset_class)  # persist trailed stop

        if healthy and self.broker_breaker.healthy:
            self._entries(now, bars, report)
            self._refresh_pending(now, report)  # instant fills become positions now

        refreshed = self._account(report)
        if refreshed is not None:
            self.store.record_equity(now, refreshed.equity, refreshed.cash)
            self._daily_summary(now, refreshed, report)
        return report

    def _entries(
        self, now: datetime, bars: Mapping[str, Sequence[Bar]], report: StepReport
    ) -> None:
        held = {s.position.plan.symbol for s in self.store.open_positions()}
        pending = {p["symbol"] for p in self.store.pending()}
        positions = self.store.open_positions()
        for symbol, item in self.universe.items():
            series = bars.get(symbol) or ()
            if symbol in held or symbol in pending or not series:
                continue
            plan = self.signal(symbol, series)
            if plan is None:
                continue
            self._request_reviews(plan, now)
            account = self.broker.account()
            marks = {
                s.position.plan.symbol: (bars.get(s.position.plan.symbol) or series)[-1].close
                for s in positions
            }
            equity = account.equity
            peak = max([equity, *(e for _, e in self.store.equity_curve(500))])
            decision = self.pipeline.decide(
                plan,
                instrument=item.instrument,
                financials=item.financials,
                crypto_asset=item.crypto_asset,
                account=ShariahOrderContext(
                    owned_quantity=Decimal("0"), free_cash=account.buying_power
                ),
                risk_context=RiskContext(
                    equity=equity,
                    peak_equity=peak,
                    start_of_day_equity=self._start_of_day_equity(now, equity),
                    positions=tuple(
                        RiskPosition(
                            s.position.plan.symbol,
                            s.position.quantity,
                            marks[s.position.plan.symbol],
                            s.position.stop,
                        )
                        for s in positions
                    ),
                    mark_prices={**marks, symbol: plan.entry_price},
                    stop_prices={symbol: plan.stop_loss},
                ),
                closed_trades=self.store.closed_trades(),
                open_symbols=held | pending,
                now=now,
            )
            stages = [
                {"stage": s.stage, "passed": s.passed, "detail": s.detail} for s in decision.stages
            ]
            self.store.record_decision(
                now, symbol, decision.approved, decision.blocked_by, decision.quantity, stages
            )
            if not decision.approved:
                report.blocked[symbol] = decision.blocked_by or "unknown"
                continue
            request = entry_order(
                plan, item.instrument.instrument_id, decision.quantity, series[-1].close, now
            )
            if request is None:
                report.blocked[symbol] = "price ran past entry"
                continue
            try:
                self.broker_breaker.call(partial(self.broker.submit, request))
            except Exception as exc:
                self._alert("critical", "entry failed", f"{symbol}: {exc!r}", report)
                continue
            self.store.add_pending(
                request.client_order_id,
                plan,
                item.instrument.asset_class.value,
                decision.quantity,
                now,
            )
            self.pipeline.governor.record_entry(now)
            report.entries_submitted.append(symbol)

    def _request_reviews(self, plan: TradePlan, now: datetime) -> None:
        """Bot -> analysts: one review request per symbol per day on the agent board."""

        if self.board is None:
            return
        key = f"reviews:{plan.symbol}:{now.date().isoformat()}"
        if self.store.get_value(key):
            return
        for role in self.review_roles:
            self.board.post_task(
                plan.symbol,
                role,
                f"Long {plan.symbol} planned: entry {plan.entry_price}, stop {plan.stop_loss}, "
                f"target {plan.profit_target}. Review from the {role.value} seat.",
            )
        self.store.set_value(key, "posted")

    def _start_of_day_equity(self, now: datetime, equity: Decimal) -> Decimal:
        key = f"sod:{now.date().isoformat()}"
        stored = self.store.get_value(key)
        if stored:
            return Decimal(stored)
        self.store.set_value(key, str(equity))
        return equity

    def _daily_summary(self, now: datetime, account: AccountSnapshot, report: StepReport) -> None:
        key = f"summary:{now.date().isoformat()}"
        if self.store.get_value(key) or now.hour < 21:  # after the US close, in UTC
            return
        start = Decimal(self.store.get_value(f"sod:{now.date().isoformat()}") or account.equity)
        change = account.equity / start - Decimal("1") if start > 0 else Decimal("0")
        trades = [t for t in self.store.closed_trades(200) if t.exit_time.date() == now.date()]
        self.alerts.send(
            "info",
            "daily summary",
            f"equity {account.equity:.2f} ({change:+.2%}), trades closed today {len(trades)}, "
            f"open positions {len(account.positions)}",
        )
        self.store.set_value(key, "sent")
