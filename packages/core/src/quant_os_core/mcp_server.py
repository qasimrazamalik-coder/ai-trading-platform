"""MCP server (stdio or HTTP) linking Quant OS to AI assistants such as Claude and ChatGPT.

Dual-era, per the MCP versioning page (https://modelcontextprotocol.io/specification/latest):

* modern clients (2026-07-28) send ``io.modelcontextprotocol/protocolVersion`` and
  ``clientCapabilities`` in every request's ``_meta``; ``server/discover`` is supported;
* legacy clients (2025-11-25 and earlier) open with ``initialize``.

Guardrail: no tool places, modifies or cancels live orders or reads credentials. The only
writes go to the agent board (research tasks, analyst reports, paper proposals), and every
proposal passes the full rule pipeline. See ``mcp_board_tools`` and ``mcp_http``.

Run:  python -m quant_os_core.mcp_server --docs docs --state state/board.db
      python -m quant_os_core.mcp_server --docs docs --state state/board.db --http 127.0.0.1:8765
      add --state-db state/core.db --screening examples/paper_core.json for the read-only
      status, decision, screening and signal tools.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, TextIO

from quant_os_core.agents.board import AgentBoard
from quant_os_core.agents.profit_engine import RiskDial, deposit_plan
from quant_os_core.agents.signals import (
    LAST_REBALANCE_KEY,
    MAX_SCREEN_AGE,
    SIGNAL_TITLE,
    Signal,
    load_screening,
    summarize,
)
from quant_os_core.instruments import AssetClass, Instrument, InstrumentId, InstrumentType
from quant_os_core.knowledge.rag import LANGUAGES, KnowledgeBase, answer_multilingual
from quant_os_core.mcp_board_tools import BOARD_TOOLS, BoardTools, ToolError, decimal_arg
from quant_os_core.ops.state import StateStore
from quant_os_core.shariah import CompanyFinancials, CryptoAsset, ShariahGate, ShariahProfile

SERVER_INFO = {"name": "quant-os", "version": "0.3.0"}
MODERN_VERSIONS = ("2026-07-28",)
LEGACY_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
META_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CAPS = "io.modelcontextprotocol/clientCapabilities"
META_SERVER = "io.modelcontextprotocol/serverInfo"
INSTRUCTIONS = (
    "Quant OS: halal trading knowledge, calculators and the agent board. No tool can trade "
    "live. Use knowledge_search before answering trading, quant-math, psychology or halal-rule "
    "questions and cite the sources. As an analyst: call get_research_tasks, research each task "
    "with your own tools, answer with submit_analyst_report (name yourself in 'source'). Use "
    "evaluate_trade_plan for dry runs; submit_trade_proposal records paper proposals only."
)

JSON = dict[str, Any]
Rates = Callable[[str, str], Decimal]  # units of the target currency per unit of the source
Generate = Callable[[Sequence[dict[str, str]]], str]


_decimal = decimal_arg


_PROFILES = {
    "pk_hanafi": ShariahProfile(),
    "owner": ShariahProfile.spot_crypto_and_day_trading(),
}

TOOLS: list[JSON] = [
    {
        "name": "knowledge_search",
        "description": "Search the Quant OS knowledge base (trading basics, quant formulas, "
        "strategies and backtesting, psychology, halal rules). Returns cited passages.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "k": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
            },
            "required": ["query"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "shariah_check",
        "description": "Check whether an instrument type is allowed by the Shariah gate. "
        "Profiles: pk_hanafi (crypto off) or owner (spot crypto and day trading on).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "asset_class": {"type": "string", "enum": [a.value for a in AssetClass]},
                "instrument_type": {"type": "string", "enum": [t.value for t in InstrumentType]},
                "profile": {"type": "string", "enum": list(_PROFILES), "default": "owner"},
                "token_category": {"type": "string", "description": "crypto only, e.g. layer1"},
            },
            "required": ["asset_class", "instrument_type"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "position_size",
        "description": "Fixed-fractional size: equity x risk_fraction / (entry - stop), "
        "capped at 20% of equity, no leverage.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "equity": {"type": "number", "exclusiveMinimum": 0},
                "entry": {"type": "number", "exclusiveMinimum": 0},
                "stop": {"type": "number", "exclusiveMinimum": 0},
                "risk_fraction": {"type": "number", "default": 0.01, "maximum": 0.05},
            },
            "required": ["equity", "entry", "stop"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "deposit_projection",
        "description": "Balance after months of compounding at an annual return with a "
        "monthly deposit. Illustrative only; returns are never constant.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "start": {"type": "number", "minimum": 0},
                "monthly_deposit": {"type": "number", "minimum": 0},
                "annual_return": {"type": "number", "exclusiveMinimum": -1},
                "months": {"type": "integer", "minimum": 1, "maximum": 600},
            },
            "required": ["start", "monthly_deposit", "annual_return", "months"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "growth_projection",
        "description": "Periods needed to grow start to target at a constant return per "
        "period: ln(target/start) / ln(1 + r). Illustrative only; returns are never constant.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "start": {"type": "number", "exclusiveMinimum": 0},
                "target": {"type": "number", "exclusiveMinimum": 0},
                "return_per_period": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["start", "target", "return_per_period"],
        },
        "annotations": {"readOnlyHint": True},
    },
]

_READ_ONLY = {"readOnlyHint": True, "destructiveHint": False}
_POSITIVE: JSON = {"type": "number", "exclusiveMinimum": 0}
_NON_NEGATIVE: JSON = {"type": "number", "minimum": 0}

# Read-only tools over the bot's state, screening, signals, calculators and multilingual
# knowledge. None of them writes anywhere; a missing data source is a tool error.
ADVANCED_TOOLS: list[JSON] = [
    {
        "name": "bot_status",
        "description": "Paper bot status from the state store: equity, peak, drawdown, open "
        "positions, pending orders, kill switch and last core rebalance. Read only.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": _READ_ONLY,
    },
    {
        "name": "explain_decision",
        "description": "The latest pipeline decisions with every gate stage (passed or "
        "blocked, and why). Optionally filter by symbol. Read only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
        },
        "annotations": _READ_ONLY,
    },
    {
        "name": "screening_results",
        "description": "Shariah screening per fund: look-through financials, AAOIFI ratio "
        "verdict, source and age of the data (stale after 120 days). Read only.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
        },
        "annotations": _READ_ONLY,
    },
    {
        "name": "signals",
        "description": "Current diversified-core signals (rebalance due, drift, 200-day trend "
        "crossings, stale screens). Information only; nothing executes from them.",
        "inputSchema": {
            "type": "object",
            "properties": {"lang": {"type": "string", "enum": list(LANGUAGES), "default": "en"}},
        },
        "annotations": _READ_ONLY,
    },
    {
        "name": "reward_risk",
        "description": "Reward-to-risk multiple R = (target - entry) / (entry - stop) for a "
        "long trade, and the break-even win rate 1 / (1 + R) before costs.",
        "inputSchema": {
            "type": "object",
            "properties": {"entry": _POSITIVE, "stop": _POSITIVE, "target": _POSITIVE},
            "required": ["entry", "stop", "target"],
        },
        "annotations": _READ_ONLY,
    },
    {
        "name": "zakat",
        "description": "Zakat at 2.5% of net zakatable wealth when it reaches the nisab you "
        "supply (gold or silver basis). General calculation, not a fatwa.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cash": _NON_NEGATIVE,
                "gold_silver": _NON_NEGATIVE,
                "investments": _NON_NEGATIVE,
                "receivables": _NON_NEGATIVE,
                "debts_due": _NON_NEGATIVE,
                "nisab": _POSITIVE,
            },
            "required": ["nisab"],
        },
        "annotations": _READ_ONLY,
    },
    {
        "name": "purification",
        "description": "Amount of income to give away: dividend x non-compliant revenue / "
        "total revenue, or non-compliant income / shares outstanding x shares held (AAOIFI "
        "SS21). Not a fatwa.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dividend": _NON_NEGATIVE,
                "noncompliant_revenue": _NON_NEGATIVE,
                "total_revenue": _POSITIVE,
                "noncompliant_income": _NON_NEGATIVE,
                "shares_outstanding": _POSITIVE,
                "shares_held": _NON_NEGATIVE,
            },
        },
        "annotations": _READ_ONLY,
    },
    {
        "name": "drawdown_recovery",
        "description": "Gain needed to recover from a drawdown: d / (1 - d). Give drawdown as "
        "a fraction (0.3 for -30%) or give peak and current equity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "drawdown": {"type": "number", "minimum": 0, "exclusiveMaximum": 1},
                "peak": _POSITIVE,
                "current": _POSITIVE,
            },
        },
        "annotations": _READ_ONLY,
    },
    {
        "name": "currency_convert",
        "description": "Convert an amount between currencies with the configured rates "
        "provider, or with an explicit rate (target units per source unit). Numbers only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "amount": _NON_NEGATIVE,
                "from": {"type": "string", "minLength": 3, "maxLength": 3},
                "to": {"type": "string", "minLength": 3, "maxLength": 3},
                "rate": _POSITIVE,
            },
            "required": ["amount", "from", "to"],
        },
        "annotations": _READ_ONLY,
    },
    {
        "name": "knowledge_ask",
        "description": "Ask the knowledge base in English, Urdu or Arabic. Answers cite their "
        "sources; with no local model, Urdu/Arabic questions get the English passages and a "
        "note. Never gives buy or sell calls.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "minLength": 2},
                "lang": {"type": "string", "enum": list(LANGUAGES), "default": "en"},
            },
            "required": ["question"],
        },
        "annotations": _READ_ONLY,
    },
]

_PCT = Decimal("0.01")


def _tool_errors(
    handler: Callable[[Mapping[str, Any]], str],
) -> Callable[[Mapping[str, Any]], str]:
    """Report bad input or unreadable data as a tool error instead of an internal error."""

    @functools.wraps(handler)
    def wrapped(args: Mapping[str, Any]) -> str:
        try:
            return handler(args)
        except ToolError:
            raise
        except (ValueError, KeyError, TypeError, ArithmeticError, OSError) as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapped


def _external(module: str, name: str) -> Callable[..., Any] | None:
    """A function from an optional sibling module (e.g. ops.calculators), if installed."""

    try:
        found = importlib.import_module(f"quant_os_core.ops.{module}")
    except ImportError:
        return None
    candidate = getattr(found, name, None)
    return candidate if callable(candidate) else None


def _calc(name: str, field: str, *args: Decimal) -> Decimal | None:
    """One figure from the shared ``ops.calculators`` function when it is installed.

    Returns ``None`` (use the local formula) if the module, the function or the field is
    missing, or if it rejects the input, so both give the same answer either way.
    """

    function = _external("calculators", name)
    if function is None:
        return None
    try:
        return Decimal(str(function(*args)[field]))
    except Exception:
        return None


def cached_fx_rates(cache_path: Path = Path("state/fx_rates.json")) -> Rates | None:
    """Read-only rates from ``ops.fx``'s saved table: never fetches, never writes."""

    fx = importlib.import_module("quant_os_core.ops.fx") if _external("fx", "FxService") else None
    if fx is None:
        return None

    def no_fetch(_url: str) -> Any:
        raise OSError("the MCP server does not download rates")

    def rate(source: str, target: str) -> Decimal:
        service = fx.FxService(fetch=no_fetch, cache_path=cache_path)
        return Decimal(str(service.rates().rate(source, target)))

    return rate


def _optional_decimal(args: Mapping[str, Any], key: str) -> Decimal:
    return _decimal(args, key) if args.get(key) is not None else Decimal(0)


class QuantOsMcp:
    def __init__(
        self,
        kb: KnowledgeBase | None,
        board: BoardTools | None = None,
        *,
        docs: Path | None = None,
        store: StateStore | None = None,
        screening: Path | None = None,
        signals: Callable[[], Sequence[Signal]] | None = None,
        rates: Rates | None = None,
        llm: Generate | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.kb = kb
        self.board = board
        self.docs = docs
        self.store = store
        self.screening = screening
        self.signals_provider = signals
        self.rates = rates
        self.llm = llm
        self.clock = clock
        self.legacy_version: str | None = None
        self.tools: list[JSON] = [*TOOLS, *ADVANCED_TOOLS, *(BOARD_TOOLS if board else [])]
        self._tools: dict[str, Callable[[Mapping[str, Any]], str]] = {
            "knowledge_search": self._knowledge_search,
            "shariah_check": self._shariah_check,
            "position_size": self._position_size,
            "growth_projection": self._growth_projection,
            "deposit_projection": self._deposit_projection,
        }
        advanced: dict[str, Callable[[Mapping[str, Any]], str]] = {
            "bot_status": self._bot_status,
            "explain_decision": self._explain_decision,
            "screening_results": self._screening_results,
            "signals": self._signals,
            "reward_risk": self._reward_risk,
            "zakat": self._zakat,
            "purification": self._purification,
            "drawdown_recovery": self._drawdown_recovery,
            "currency_convert": self._currency_convert,
            "knowledge_ask": self._knowledge_ask,
        }
        self._tools.update({name: _tool_errors(fn) for name, fn in advanced.items()})
        if board is not None:
            self._tools.update(board.handlers())

    # ---- read-only state tools -----------------------------------------------------------
    def _need_store(self) -> StateStore:
        if self.store is None:
            raise ToolError("state store not configured; start the server with --state-db")
        return self.store

    def _bot_status(self, _args: Mapping[str, Any]) -> str:
        store = self._need_store()
        curve = store.equity_curve()
        status: JSON = {
            "kill_switch": "on" if store.kill_switch_on() else "off",
            "last_rebalance": store.get_value(LAST_REBALANCE_KEY) or None,
            "pending_orders": len(store.pending()),
        }
        if curve:
            when, equity = curve[-1]
            peak = max(e for _, e in curve)
            drawdown = (peak - equity) / peak * 100 if peak > 0 else Decimal(0)
            status.update(
                equity=str(equity),
                peak_equity=str(peak),
                drawdown_pct=str(drawdown.quantize(_PCT)),
                as_of=when.isoformat(),
            )
        else:
            status["equity"] = None
        status["positions"] = [
            {
                "symbol": s.position.plan.symbol,
                "asset_class": s.asset_class,
                "quantity": str(s.position.quantity),
                "entry": str(s.position.plan.entry_price),
                "stop": str(s.position.stop),
                "target": str(s.position.plan.profit_target),
            }
            for s in store.open_positions()
        ]
        return json.dumps(status)

    def _explain_decision(self, args: Mapping[str, Any]) -> str:
        store = self._need_store()
        limit = max(1, min(20, int(args.get("limit", 5))))
        symbol = str(args.get("symbol", "")).strip().upper()
        rows = store.recent("decisions", 500 if symbol else limit)
        if symbol:
            rows = [r for r in rows if str(r["symbol"]).upper() == symbol][:limit]
        if not rows:
            return "No decisions recorded yet."
        out = [
            {
                "time": r["time"],
                "symbol": r["symbol"],
                "verdict": "approved" if r["approved"] else f"blocked at {r['blocked_by']}",
                "quantity": r["quantity"],
                "stages": json.loads(r["stages"]),
            }
            for r in rows
        ]
        return json.dumps(out)

    def _screening_results(self, args: Mapping[str, Any]) -> str:
        if self.screening is None:
            raise ToolError("screening config not configured; start with --screening")
        records = load_screening(self.screening)
        wanted = str(args.get("symbol", "")).strip().upper()
        today = self.clock().date()
        gate = ShariahGate(ShariahProfile())
        out = []
        for symbol, record in sorted(records.items()):
            if wanted and symbol != wanted:
                continue
            item: JSON = {"symbol": symbol, "source": record.source or None}
            if record.financials is None:
                item.update(verdict="blocked (R6): no financials on file", stale=True)
                out.append(item)
                continue
            numbers = {k: Decimal(v) for k, v in record.financials.items() if k != "sector"}
            financials = CompanyFinancials(sector=record.financials.get("sector", ""), **numbers)
            decision = gate.screen_company(financials)
            cap = financials.market_cap
            item["ratios"] = {
                "interest_debt_to_market_cap": f"{financials.interest_bearing_debt / cap:.2%}",
                "interest_cash_to_market_cap": (
                    f"{financials.interest_bearing_cash_and_securities / cap:.2%}"
                ),
                "noncompliant_revenue": (
                    f"{financials.noncompliant_revenue / financials.total_revenue:.2%}"
                ),
                "tangible_assets": f"{financials.tangible_assets / financials.total_assets:.2%}",
            }
            item["verdict"] = (
                "pass" if decision.allowed else f"blocked ({decision.rule_id}): {decision.reason}"
            )
            age = (today - record.as_of).days if record.as_of else None
            item.update(
                as_of=record.as_of.isoformat() if record.as_of else None,
                age_days=age,
                stale=age is None or age > MAX_SCREEN_AGE.days,
            )
            out.append(item)
        if not out:
            raise ToolError(f"no screening record for {wanted or 'any fund'}")
        return json.dumps(out)

    def _signals(self, args: Mapping[str, Any]) -> str:
        lang = str(args.get("lang", "en"))
        if lang not in LANGUAGES:
            raise ToolError(f"lang must be one of {list(LANGUAGES)}")
        if self.signals_provider is not None:
            current = list(self.signals_provider())
            body = {"summary": summarize(current, lang), "signals": [s.as_dict() for s in current]}
            return json.dumps(body, ensure_ascii=False)
        if self.store is not None:  # latest summary the signal agent sent to the store sink
            prefix = f"{SIGNAL_TITLE}:"
            for event in self.store.recent("events", 500):
                if event["kind"] == "alert" and str(event["message"]).startswith(prefix):
                    return f"latest published at {event['time']}:\n{event['message']}"
            return "No signals published yet."
        raise ToolError("signals not configured; start with --state-db")

    # ---- calculators ---------------------------------------------------------------------
    def _reward_risk(self, args: Mapping[str, Any]) -> str:
        entry, stop, target = (_decimal(args, k) for k in ("entry", "stop", "target"))
        if min(entry, stop, target) <= 0:
            raise ToolError("entry, stop and target must be positive")
        if not stop < entry < target:
            raise ToolError("long-only: need stop < entry < target")
        shared = _calc("reward_risk", "reward_risk", entry, stop, target)
        multiple = shared if shared is not None else (target - entry) / (entry - stop)
        breakeven = 1 / (1 + multiple) * 100
        return (
            f"reward:risk {multiple:.2f}R (risk {entry - stop:.2f}, reward {target - entry:.2f} "
            f"per unit); break-even win rate {breakeven:.1f}% before costs"
        )

    def _zakat(self, args: Mapping[str, Any]) -> str:
        nisab = _decimal(args, "nisab")
        parts = {
            k: _optional_decimal(args, k)
            for k in ("cash", "gold_silver", "investments", "receivables", "debts_due")
        }
        if nisab <= 0 or any(v < 0 for v in parts.values()):
            raise ToolError("nisab must be positive and amounts non-negative")
        net = sum((v for k, v in parts.items() if k != "debts_due"), Decimal(0))
        net -= parts["debts_due"]
        note = "General calculation, not a fatwa; confirm the method with a scholar."
        if net < nisab:
            return f"net zakatable wealth {net:.2f} is below the nisab {nisab:.2f}: zakat 0. {note}"
        shared = _calc("zakat", "due", net, nisab)
        due = shared if shared is not None else net * Decimal("0.025")
        return f"net zakatable wealth {net:.2f}; zakat at 2.5% = {due:.2f}. {note}"

    def _purification(self, args: Mapping[str, Any]) -> str:
        note = "Give it to charity; it is not zakat. Not a fatwa."
        if args.get("dividend") is not None:
            dividend = _decimal(args, "dividend")
            bad, total = _decimal(args, "noncompliant_revenue"), _decimal(args, "total_revenue")
            if total <= 0 or dividend < 0 or bad < 0 or bad > total:
                raise ToolError("need dividend >= 0 and 0 <= noncompliant_revenue <= total_revenue")
            shared = _calc("purification", "purify", dividend, bad / total * 100)
            amount = shared if shared is not None else dividend * bad / total
            return f"purify {amount:.2f} of the {dividend:.2f} dividend ({bad / total:.2%}). {note}"
        if args.get("noncompliant_income") is not None:
            income = _decimal(args, "noncompliant_income")
            shares, held = _decimal(args, "shares_outstanding"), _decimal(args, "shares_held")
            if shares <= 0 or held < 0 or income < 0:
                raise ToolError("need shares_outstanding > 0, shares_held >= 0, income >= 0")
            return f"purify {income / shares * held:.4f} (AAOIFI SS21 3/4/6/4). {note}"
        raise ToolError("give dividend + revenues, or noncompliant_income + share counts")

    def _drawdown_recovery(self, args: Mapping[str, Any]) -> str:
        if args.get("drawdown") is not None:
            drawdown = _decimal(args, "drawdown")
        elif args.get("peak") is not None and args.get("current") is not None:
            peak, current = _decimal(args, "peak"), _decimal(args, "current")
            if peak <= 0 or current <= 0 or current > peak:
                raise ToolError("need 0 < current <= peak")
            drawdown = (peak - current) / peak
        else:
            raise ToolError("give drawdown (fraction) or peak and current")
        if not Decimal(0) <= drawdown < 1:
            raise ToolError("drawdown must be a fraction in [0, 1)")
        shared = _calc("drawdown_recovery", "required_gain_pct", drawdown * 100)
        needed = shared if shared is not None else drawdown / (1 - drawdown) * 100
        return f"a {drawdown:.2%} drawdown needs a {needed:.2f}% gain to recover"

    def _rate(self, source: str, target: str) -> Decimal:
        if self.rates is None:
            raise ToolError("no rates provider configured; pass 'rate' explicitly")
        try:
            return Decimal(str(self.rates(source, target)))
        except Exception as exc:  # offline, unknown currency, no saved table
            raise ToolError(f"rates unavailable: {exc}") from exc

    def _currency_convert(self, args: Mapping[str, Any]) -> str:
        amount = _decimal(args, "amount")
        source = str(args.get("from", "")).strip().upper()
        target = str(args.get("to", "")).strip().upper()
        if amount < 0 or len(source) != 3 or len(target) != 3:
            raise ToolError("need amount >= 0 and 3-letter currency codes")
        if source == target:
            rate = Decimal(1)
        elif args.get("rate") is not None:
            rate = _decimal(args, "rate")
        else:
            rate = self._rate(source, target)
        if rate <= 0:
            raise ToolError("rate must be positive")
        return f"{amount} {source} = {amount * rate:.2f} {target} at {rate} {target}/{source}"

    def _knowledge_ask(self, args: Mapping[str, Any]) -> str:
        if self.docs is None:
            raise ToolError("knowledge folder not configured; start the server with --docs")
        question = str(args.get("question", "")).strip()
        if len(question) < 2:
            raise ToolError("'question' is required")
        lang = str(args.get("lang", "en"))
        if lang not in LANGUAGES:
            raise ToolError(f"lang must be one of {list(LANGUAGES)}")
        reply = answer_multilingual(question, lang, docs=self.docs, llm=self.llm)
        sources = "\n".join(
            f"  [{i}] {s['source']} - {s['heading']}"
            for i, s in enumerate(reply["sources"], start=1)
        )
        return f"{reply['answer']}\n{sources}".rstrip()

    # ---- tools -------------------------------------------------------------------------
    def _knowledge_search(self, args: Mapping[str, Any]) -> str:
        if self.kb is None:
            raise ToolError("knowledge base not loaded; start the server with --docs")
        query = str(args.get("query", "")).strip()
        if len(query) < 2:
            raise ToolError("'query' is required")
        k = int(args.get("k", 4))
        hits = self.kb.search(query, max(1, min(8, k)))
        if not hits:
            return "No passages found."
        return "\n\n".join(
            f"[{i}] {h.chunk.source} - {h.chunk.heading}\n{h.chunk.text}"
            for i, h in enumerate(hits, start=1)
        )

    def _shariah_check(self, args: Mapping[str, Any]) -> str:
        try:
            asset = AssetClass(str(args["asset_class"]))
            kind = InstrumentType(str(args["instrument_type"]))
            profile = _PROFILES[str(args.get("profile", "owner"))]
        except (KeyError, ValueError) as exc:
            raise ToolError("unknown asset_class, instrument_type or profile") from exc
        gate = ShariahGate(profile)
        instrument = Instrument(
            instrument_id=InstrumentId(venue="CHECK", symbol="X"),
            asset_class=asset,
            instrument_type=kind,
            base_currency="XXX",
            quote_currency="USD",
            price_precision=2,
            quantity_precision=4,
        )
        decision = gate.check_instrument(instrument)
        category = args.get("token_category")
        if decision.allowed and asset is AssetClass.CRYPTO and category:
            decision = gate.screen_crypto(CryptoAsset("X", str(category)))
        verdict = "allowed" if decision.allowed else "blocked"
        return f"{verdict} ({decision.rule_id}): {decision.reason}"

    def _position_size(self, args: Mapping[str, Any]) -> str:
        equity, entry, stop = (_decimal(args, k) for k in ("equity", "entry", "stop"))
        risk = Decimal(str(args.get("risk_fraction", "0.01")))
        if min(equity, entry, stop) <= 0 or not Decimal("0") < risk <= Decimal("0.05"):
            raise ToolError("equity, entry, stop must be positive; risk_fraction in (0, 0.05]")
        if stop >= entry:
            raise ToolError("long-only: stop must be below entry")
        risk_qty = equity * risk / (entry - stop)
        cap_qty = equity * Decimal("0.20") / entry
        qty = min(risk_qty, cap_qty)
        capped = " (capped at 20% of equity)" if cap_qty < risk_qty else ""
        return (
            f"quantity {qty:.6f}{capped}; cost {qty * entry:.2f}; "
            f"loss at stop {qty * (entry - stop):.2f}"
        )

    def _growth_projection(self, args: Mapping[str, Any]) -> str:
        start, target, rate = (_decimal(args, k) for k in ("start", "target", "return_per_period"))
        if min(start, target, rate) <= 0:
            raise ToolError("all inputs must be positive")
        if target <= start:
            return "0 periods: target already reached"
        periods = math.log(float(target / start)) / math.log1p(float(rate))
        return f"{periods:.1f} periods at a constant {float(rate):.2%} per period"

    def _deposit_projection(self, args: Mapping[str, Any]) -> str:
        start, monthly, rate = (
            _decimal(args, k) for k in ("start", "monthly_deposit", "annual_return")
        )
        try:
            months = int(args["months"])
            path = deposit_plan(start, monthly, rate, months)
        except (KeyError, ValueError) as exc:
            raise ToolError(f"invalid inputs: {exc}") from exc
        end = path[-1]
        return (
            f"after {months} months: balance {end.balance}, deposited {end.deposited}, "
            f"growth {end.balance - end.deposited:.2f}"
        )

    # ---- protocol ----------------------------------------------------------------------
    def _result(self, body: JSON) -> JSON:
        return {"resultType": "complete", **body, "_meta": {META_SERVER: SERVER_INFO}}

    @staticmethod
    def _error(msg_id: Any, code: int, message: str, data: Any = None) -> JSON:
        error: JSON = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": msg_id, "error": error}

    def handle(self, message: Mapping[str, Any]) -> JSON | None:
        msg_id = message.get("id")
        method = message.get("method")
        if not isinstance(method, str):
            return self._error(msg_id, -32600, "invalid request")
        if msg_id is None:
            return None  # notification, e.g. notifications/initialized
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return self._error(msg_id, -32602, "params must be an object")

        if method == "initialize":
            requested = str(params.get("protocolVersion", ""))
            # Legacy negotiation: echo a supported version, else offer our latest legacy one.
            self.legacy_version = requested if requested in LEGACY_VERSIONS else LEGACY_VERSIONS[0]
            result = {
                "protocolVersion": self.legacy_version,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
                "instructions": INSTRUCTIONS,
            }
            return {"jsonrpc": "2.0", "id": msg_id, "result": result}

        meta = params.get("_meta") or {}
        version = meta.get(META_VERSION) if isinstance(meta, dict) else None
        if version is not None:
            if version not in MODERN_VERSIONS:
                supported = [*MODERN_VERSIONS, *LEGACY_VERSIONS[:1]]
                data = {"supported": supported, "requested": version}
                return self._error(msg_id, -32022, "Unsupported protocol version", data)
            if META_CAPS not in meta:
                return self._error(msg_id, -32602, f"missing _meta '{META_CAPS}'")
        elif self.legacy_version is None and method != "ping":
            return self._error(
                msg_id, -32602, f"missing _meta '{META_VERSION}' (or send initialize first)"
            )

        body: JSON
        if method == "server/discover":
            body = {
                "supportedVersions": [*MODERN_VERSIONS, *LEGACY_VERSIONS],
                "capabilities": {"tools": {}},
                "instructions": INSTRUCTIONS,
            }
            return {"jsonrpc": "2.0", "id": msg_id, "result": self._result(body)}
        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": self._result({})}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": self._result({"tools": self.tools})}
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            tool = self._tools.get(str(name))
            if tool is None or not isinstance(arguments, dict):
                return self._error(msg_id, -32602, f"unknown tool or bad arguments: {name}")
            try:
                text, is_error = tool(arguments), False
            except ToolError as exc:
                text, is_error = str(exc), True
            body = {"content": [{"type": "text", "text": text}], "isError": is_error}
            return {"jsonrpc": "2.0", "id": msg_id, "result": self._result(body)}
        return self._error(msg_id, -32601, f"method not found: {method}")

    def serve(self, stdin: TextIO, stdout: TextIO) -> None:
        """Newline-delimited JSON-RPC over stdio; logs go to stderr only."""

        for line in stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                reply: JSON | None = self._error(None, -32700, "parse error")
            else:
                try:
                    reply = (
                        self.handle(message)
                        if isinstance(message, dict)
                        else self._error(None, -32600, "invalid request")
                    )
                except Exception as exc:  # never crash the server on one bad request
                    print(f"quant-os mcp: internal error: {exc!r}", file=sys.stderr)
                    reply = self._error(message.get("id"), -32603, "internal error")
            if reply is not None:
                stdout.write(json.dumps(reply) + "\n")
                stdout.flush()


def build_server(
    docs: Path,
    state: Path | None,
    paper_equity: str,
    dial: str,
    *,
    state_db: Path | None = None,
    screening: Path | None = None,
) -> QuantOsMcp:
    kb = KnowledgeBase.from_directory(docs) if docs.is_dir() else None
    if kb is None:
        print(f"quant-os mcp: docs folder {docs} not found; search disabled", file=sys.stderr)
    board = (
        BoardTools(AgentBoard(state), paper_equity=Decimal(paper_equity), dial=RiskDial(dial))
        if state is not None
        else None
    )
    store = None
    if state_db is not None:
        if state_db.is_file():  # never create a trading-state file from here
            store = StateStore(state_db)
        else:
            print(f"quant-os mcp: state db {state_db} not found; status disabled", file=sys.stderr)
    return QuantOsMcp(
        kb,
        board,
        docs=docs if kb is not None else None,
        store=store,
        screening=screening,
        rates=cached_fx_rates(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quant OS MCP server (stdio or HTTP)")
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--state", type=Path, help="agent board SQLite file; enables board tools")
    parser.add_argument("--paper-equity", default="50")
    parser.add_argument("--dial", default="balanced", choices=[d.value for d in RiskDial])
    parser.add_argument("--http", metavar="HOST:PORT", help="serve Streamable HTTP instead")
    parser.add_argument("--state-db", type=Path, help="paper loop StateStore (read only)")
    parser.add_argument("--screening", type=Path, help="config with fund financials")
    args = parser.parse_args(argv)
    server = build_server(
        args.docs,
        args.state,
        args.paper_equity,
        args.dial,
        state_db=args.state_db,
        screening=args.screening,
    )
    if args.http:
        from quant_os_core.mcp_http import serve_http

        host, _, port = args.http.rpartition(":")
        return serve_http(server, host or "127.0.0.1", int(port))
    server.serve(sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
