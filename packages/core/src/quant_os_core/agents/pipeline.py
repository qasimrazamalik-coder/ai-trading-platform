"""The entry decision pipeline, in the order the research report specifies.

plan validation -> Shariah gate -> protections -> frequency governor -> analyst council
-> sizing -> independent risk engine. Every stage can stop the trade; none can enlarge
what an earlier stage allowed. Each stage's verdict is kept for the audit log.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_DOWN, Decimal

from quant_os_core.agents.council import AnalystCouncil
from quant_os_core.agents.trade_plan import PlanGovernor, TradePlan, size_plan, validate_plan
from quant_os_core.events import OrderIntent, Side
from quant_os_core.instruments import Instrument
from quant_os_core.risk_engine import (
    ClosedTrade,
    ProtectionManager,
    RiskContext,
    RiskEngine,
)
from quant_os_core.shariah import (
    CompanyFinancials,
    ShariahGate,
    ShariahOrderContext,
)


@dataclass(frozen=True, slots=True)
class StageResult:
    stage: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class EntryDecision:
    approved: bool
    quantity: Decimal
    intent: OrderIntent | None
    stages: tuple[StageResult, ...]

    @property
    def blocked_by(self) -> str | None:
        return next((s.stage for s in self.stages if not s.passed), None)


@dataclass(slots=True)
class EntryPipeline:
    shariah: ShariahGate
    protections: ProtectionManager
    governor: PlanGovernor
    council: AnalystCouncil
    risk: RiskEngine
    strategy_id: str = "halal-council"
    _log: list[EntryDecision] = field(default_factory=list)

    def decide(
        self,
        plan: TradePlan,
        *,
        instrument: Instrument,
        financials: CompanyFinancials | None,
        account: ShariahOrderContext,
        risk_context: RiskContext,
        closed_trades: Sequence[ClosedTrade],
        open_symbols: set[str],
        now: datetime,
    ) -> EntryDecision:
        stages: list[StageResult] = []

        def stop(stage: str, detail: str) -> EntryDecision:
            stages.append(StageResult(stage, False, detail))
            decision = EntryDecision(False, Decimal("0"), None, tuple(stages))
            self._log.append(decision)
            return decision

        verdict = validate_plan(plan, self.governor.policy)
        if not verdict.accepted:
            return stop("plan", verdict.reason)
        stages.append(StageResult("plan", True, verdict.reason))

        quantity = size_plan(plan, self.governor.policy, risk_context.equity, account.free_cash)
        if quantity <= 0:
            return stop("sizing", "size rounds to zero for this account")
        shariah = self.shariah.check_order(
            instrument,
            is_sell=False,
            quantity=quantity,
            notional=quantity * plan.entry_price,
            context=account,
            financials=financials,
        )
        if not shariah.allowed:
            return stop("shariah", f"{shariah.rule_id}: {shariah.reason}")
        stages.append(StageResult("shariah", True, shariah.reason))

        lock = self.protections.blocking_lock(plan.symbol, closed_trades, now)
        if lock is not None:
            return stop("protections", f"{lock.protection}: {lock.reason} until {lock.until}")
        stages.append(StageResult("protections", True, "no active locks"))

        frequency = self.governor.check(plan, open_symbols, now)
        if not frequency.accepted:
            return stop("governor", frequency.reason)
        stages.append(StageResult("governor", True, frequency.reason))

        council = self.council.deliberate(plan)
        if not council.approved:
            return stop("council", council.reason)
        stages.append(
            StageResult("council", True, f"{council.reason} x{council.size_multiplier:.2f}")
        )
        step = self.governor.policy.quantity_step
        quantity = (quantity * council.size_multiplier / step).to_integral_value(ROUND_DOWN) * step
        if quantity <= 0:
            return stop("sizing", "council multiplier reduced size to zero")

        intent = OrderIntent(
            source=self.strategy_id,
            strategy_id=self.strategy_id,
            symbol=plan.symbol,
            side=Side.BUY,
            quantity=quantity,
            limit_price=plan.entry_price,
        )
        evaluation = self.risk.evaluate(intent, risk_context)
        if not evaluation.approved:
            return stop("risk", "; ".join(evaluation.reasons) or str(evaluation.action))
        stages.append(StageResult("risk", True, str(evaluation.action)))

        final = evaluation.approved_intent
        self.governor.record_entry(now)
        decision = EntryDecision(True, evaluation.approved_quantity, final, tuple(stages))
        self._log.append(decision)
        return decision

    @property
    def audit_log(self) -> tuple[EntryDecision, ...]:
        return tuple(self._log)
