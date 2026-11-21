"""Agent-board tools for the MCP server: the two-way link between the bot and AI assistants.

Writes go only to the agent board (SQLite). Proposals are evaluated by the full entry
pipeline in paper mode and recorded; no tool can reach a broker.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import ValidationError

from quant_os_core.agents.board import (
    AgentBoard,
    AnalystRole,
    BoardAnalyst,
    distinct_sources,
)
from quant_os_core.agents.council import AnalystCouncil, Stance
from quant_os_core.agents.pipeline import EntryPipeline
from quant_os_core.agents.profit_engine import CostModel, RiskDial, dial_policy
from quant_os_core.agents.trade_plan import PlanGovernor, TradePlan
from quant_os_core.instruments import AssetClass, Instrument, InstrumentId, InstrumentType
from quant_os_core.risk_engine import ProtectionManager, RiskContext, RiskEngine
from quant_os_core.shariah import (
    CompanyFinancials,
    CryptoAsset,
    ShariahGate,
    ShariahOrderContext,
    ShariahProfile,
    halal_risk_limits,
)

JSON = dict[str, Any]
CONSENSUS_WINDOW = timedelta(hours=24)
MIN_SOURCES_FOR_PROPOSAL = 2  # e.g. claude + chatgpt, or one of them + the local model


class ToolError(Exception):
    """A tool-level failure, reported to the model inside the result (isError)."""


def decimal_arg(args: Mapping[str, Any], key: str) -> Decimal:
    try:
        value = Decimal(str(args[key]))
    except (KeyError, InvalidOperation) as exc:
        raise ToolError(f"'{key}' must be a number") from exc
    if not value.is_finite():
        raise ToolError(f"'{key}' must be finite")
    return value


_ROLES = [r.value for r in AnalystRole]
_PLAN_PROPS: JSON = {
    "symbol": {"type": "string", "description": "e.g. AAPL or BTC/USD"},
    "asset_class": {"type": "string", "enum": ["equity", "etf", "crypto"]},
    "entry_price": {"type": "number", "exclusiveMinimum": 0},
    "stop_loss": {"type": "number", "exclusiveMinimum": 0},
    "profit_target": {"type": "number", "exclusiveMinimum": 0},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "invalidation_condition": {"type": "string", "minLength": 8},
    "justification": {"type": "string"},
    "token_category": {"type": "string", "description": "crypto only, e.g. layer1"},
    "financials": {
        "type": "object",
        "description": "equity/etf only: AAOIFI screen inputs from the latest filing",
        "properties": {
            k: {"type": "number" if k != "sector" else "string"}
            for k in (
                "sector",
                "market_cap",
                "interest_bearing_debt",
                "interest_bearing_cash_and_securities",
                "total_revenue",
                "noncompliant_revenue",
                "total_assets",
                "tangible_assets",
                "shares_outstanding",
            )
        },
    },
}
_PLAN_REQUIRED = [
    "symbol",
    "asset_class",
    "entry_price",
    "stop_loss",
    "profit_target",
    "confidence",
    "invalidation_condition",
]

BOARD_TOOLS: list[JSON] = [
    {
        "name": "get_research_tasks",
        "description": "Open research tasks the bot wants analysed. Take one, research it with "
        "your own tools, then answer with submit_analyst_report including its task_id.",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}},
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "post_research_task",
        "description": "Ask the other analysts (another assistant or the local model) to "
        "research a symbol from a given role.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "role": {"type": "string", "enum": _ROLES},
                "question": {"type": "string", "minLength": 10},
            },
            "required": ["symbol", "role", "question"],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False},
    },
    {
        "name": "submit_analyst_report",
        "description": "Submit your analysis as one council seat. Reports can veto or shrink a "
        "trade; they can never enlarge or approve one. Name yourself in 'source'.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "role": {"type": "string", "enum": _ROLES},
                "source": {"type": "string", "description": "claude, chatgpt, qwen-local..."},
                "stance": {"type": "string", "enum": [s.value for s in Stance]},
                "conviction": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string", "minLength": 20, "maxLength": 800},
                "red_flag": {"type": "boolean"},
                "task_id": {"type": "integer"},
            },
            "required": ["symbol", "role", "source", "stance", "conviction", "rationale"],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False},
    },
    {
        "name": "get_analyst_board",
        "description": "Recent analyst reports for a symbol from every source.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "hours": {"type": "integer", "minimum": 1, "maximum": 720},
            },
            "required": ["symbol"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "evaluate_trade_plan",
        "description": "Dry run: send a trade plan through the full pipeline (plan rules, edge "
        "over cost, Shariah gate, protections, governor, council, risk engine) and see which "
        "stage passes or blocks it. Nothing is recorded or traded.",
        "inputSchema": {"type": "object", "properties": _PLAN_PROPS, "required": _PLAN_REQUIRED},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "submit_trade_proposal",
        "description": "Record a PAPER trade proposal after the full pipeline. Needs fresh "
        "analyst reports for the symbol from at least two different sources. Never trades live.",
        "inputSchema": {
            "type": "object",
            "properties": {**_PLAN_PROPS, "source": {"type": "string"}},
            "required": [*_PLAN_REQUIRED, "source"],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False},
    },
    {
        "name": "list_proposals",
        "description": "Recent paper proposals and the stage that approved or blocked each.",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
        },
        "annotations": {"readOnlyHint": True},
    },
]


class BoardTools:
    def __init__(
        self,
        board: AgentBoard,
        *,
        paper_equity: Decimal = Decimal("50"),
        dial: RiskDial = RiskDial.BALANCED,
        profile: ShariahProfile | None = None,
    ) -> None:
        self.board = board
        self.paper_equity = paper_equity
        self.dial = dial
        self.profile = profile or ShariahProfile.spot_crypto_and_day_trading()

    def handlers(self) -> dict[str, Callable[[Mapping[str, Any]], str]]:
        return {
            "get_research_tasks": self.get_research_tasks,
            "post_research_task": self.post_research_task,
            "submit_analyst_report": self.submit_analyst_report,
            "get_analyst_board": self.get_analyst_board,
            "evaluate_trade_plan": lambda a: self._evaluate(a, record=False),
            "submit_trade_proposal": lambda a: self._evaluate(a, record=True),
            "list_proposals": self.list_proposals,
        }

    @staticmethod
    def _role(args: Mapping[str, Any]) -> AnalystRole:
        try:
            return AnalystRole(str(args["role"]))
        except (KeyError, ValueError) as exc:
            raise ToolError(f"role must be one of {_ROLES}") from exc

    def get_research_tasks(self, args: Mapping[str, Any]) -> str:
        tasks = self.board.open_tasks(int(args.get("limit", 10)))
        return json.dumps(tasks) if tasks else "No open research tasks."

    def post_research_task(self, args: Mapping[str, Any]) -> str:
        try:
            task_id = self.board.post_task(
                str(args.get("symbol", "")), self._role(args), str(args.get("question", ""))
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return f"task {task_id} posted"

    def submit_analyst_report(self, args: Mapping[str, Any]) -> str:
        try:
            report_id = self.board.submit_report(
                symbol=str(args["symbol"]),
                role=self._role(args),
                source=str(args["source"]),
                stance=Stance(str(args["stance"])),
                conviction=decimal_arg(args, "conviction"),
                rationale=str(args["rationale"]),
                red_flag=bool(args.get("red_flag", False)),
                task_id=int(args["task_id"]) if args.get("task_id") is not None else None,
            )
        except (KeyError, ValueError, ValidationError) as exc:
            raise ToolError(f"invalid report: {exc}") from exc
        return f"report {report_id} recorded"

    def get_analyst_board(self, args: Mapping[str, Any]) -> str:
        hours = max(1, min(720, int(args.get("hours", 24))))
        reports = self.board.recent_reports(str(args.get("symbol", "")), timedelta(hours=hours))
        if not reports:
            return "No recent reports."
        return json.dumps(
            [
                {
                    "role": r.role.value,
                    "source": r.source,
                    "stance": r.report.stance.value,
                    "conviction": str(r.report.conviction),
                    "red_flag": r.report.red_flag,
                    "rationale": r.report.rationale,
                    "created": r.created.isoformat(),
                }
                for r in reports
            ]
        )

    def list_proposals(self, args: Mapping[str, Any]) -> str:
        items = self.board.proposals(int(args.get("limit", 20)))
        return json.dumps(items) if items else "No proposals yet."

    # ---- evaluation ---------------------------------------------------------------------
    def _plan(self, args: Mapping[str, Any]) -> tuple[TradePlan, Instrument]:
        symbol = str(args.get("symbol", "")).strip().upper()
        try:
            asset = AssetClass(str(args.get("asset_class")))
        except ValueError as exc:
            raise ToolError("asset_class must be equity, etf or crypto") from exc
        if asset not in {AssetClass.EQUITY, AssetClass.ETF, AssetClass.CRYPTO}:
            raise ToolError("only spot equity, etf and crypto can be proposed")
        try:
            plan = TradePlan(
                symbol=symbol,
                entry_price=decimal_arg(args, "entry_price"),
                profit_target=decimal_arg(args, "profit_target"),
                stop_loss=decimal_arg(args, "stop_loss"),
                invalidation_condition=str(args.get("invalidation_condition", "")),
                confidence=decimal_arg(args, "confidence"),
                justification=str(args.get("justification", "assistant proposal"))[:400],
            )
        except ValidationError as exc:
            raise ToolError(f"invalid plan: {exc.errors()[0]['msg']}") from exc
        base, _, quote = symbol.partition("/")
        instrument = Instrument(
            instrument_id=InstrumentId(venue="PAPER", symbol=symbol),
            asset_class=asset,
            instrument_type=InstrumentType.SPOT,
            base_currency=(base or symbol)[:12].ljust(3, "X"),
            quote_currency=(quote or "USD")[:12],
            price_precision=8,
            quantity_precision=8,
        )
        return plan, instrument

    def _financials(self, args: Mapping[str, Any]) -> CompanyFinancials | None:
        raw = args.get("financials")
        if not raw:
            return None
        if not isinstance(raw, dict):
            raise ToolError("financials must be an object")
        try:
            numbers = {k: Decimal(str(v)) for k, v in raw.items() if k != "sector"}
            return CompanyFinancials(sector=str(raw.get("sector", "")), **numbers)
        except (TypeError, ValueError, InvalidOperation) as exc:
            raise ToolError(f"invalid financials: {exc}") from exc

    def _pipeline(self) -> EntryPipeline:
        seats = [BoardAnalyst(self.board, role, CONSENSUS_WINDOW) for role in AnalystRole]
        return EntryPipeline(
            shariah=ShariahGate(self.profile),
            protections=ProtectionManager([]),
            governor=PlanGovernor(dial_policy(self.dial)),
            council=AnalystCouncil(seats),
            risk=RiskEngine(halal_risk_limits()),
            strategy_id="assistant-proposal",
            cost_model=CostModel(),
        )

    def _evaluate(self, args: Mapping[str, Any], *, record: bool) -> str:
        plan, instrument = self._plan(args)
        source = str(args.get("source", "")).strip().lower()
        if record:
            if not source:
                raise ToolError("'source' is required for proposals")
            sources = distinct_sources(self.board, plan.symbol, CONSENSUS_WINDOW)
            if len(sources) < MIN_SOURCES_FOR_PROPOSAL:
                raise ToolError(
                    f"need fresh reports from {MIN_SOURCES_FOR_PROPOSAL} different sources for "
                    f"{plan.symbol}; have {sorted(sources) or 'none'}"
                )
        token = args.get("token_category")
        crypto = CryptoAsset(plan.symbol, str(token)) if token else None
        equity = self.paper_equity
        decision = self._pipeline().decide(
            plan,
            instrument=instrument,
            financials=self._financials(args),
            crypto_asset=crypto,
            account=ShariahOrderContext(owned_quantity=Decimal("0"), free_cash=equity),
            risk_context=RiskContext(
                equity=equity,
                peak_equity=equity,
                start_of_day_equity=equity,
                positions=(),
                mark_prices={plan.symbol: plan.entry_price},
                stop_prices={plan.symbol: plan.stop_loss},
            ),
            closed_trades=[],
            open_symbols=set(),
            now=datetime.now(UTC),
        )
        stages = [
            {"stage": s.stage, "passed": s.passed, "detail": s.detail} for s in decision.stages
        ]
        verdict = "APPROVED (paper)" if decision.approved else f"BLOCKED at {decision.blocked_by}"
        summary: JSON = {"verdict": verdict, "quantity": str(decision.quantity), "stages": stages}
        if record:
            summary["proposal_id"] = self.board.record_proposal(
                source=source,
                plan=plan,
                asset_class=instrument.asset_class.value,
                approved=decision.approved,
                blocked_by=decision.blocked_by,
                quantity=decision.quantity,
                stages=stages,
            )
        return json.dumps(summary)
