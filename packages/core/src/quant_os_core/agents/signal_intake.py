"""External trade-signal intake: text -> TradePlan candidate -> paper entry pipeline.

A signal such as ``BUY BTC/USD 65000 SL 62000 TP 72000`` becomes a long-only spot
:class:`TradePlan`. Anything that smells of shorting, leverage or derivatives (``SHORT``,
``10x``, ``PERP``, ``futures``, ``margin``, ...) is rejected with a reason and counted
against the provider's halal compliance. Opening ``SELL`` signals are rejected too: exits
come from our own registered exit plan, and without a position a sell would be a short.

Every accepted candidate goes through ``EntryPipeline.decide`` in paper mode, with provider
attribution, so the marketplace can grade the provider. The provider's text never sets the
size or the confidence: confidence is a fixed prior chosen by the owner.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Protocol

from pydantic import ValidationError

from quant_os_core.agents.marketplace import Marketplace, ProviderKind, SignalOutcome
from quant_os_core.agents.pipeline import EntryDecision
from quant_os_core.agents.trade_plan import TradePlan
from quant_os_core.instruments import Instrument
from quant_os_core.risk_engine import ClosedTrade, RiskContext
from quant_os_core.shariah import CompanyFinancials, CryptoAsset, ShariahOrderContext

PAPER = "paper"
MAX_SIGNAL_CHARS = 500

_FORBIDDEN: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bshort(s|ing)?\b", re.I), "short selling is not allowed (R4)"),
    (
        re.compile(r"\bleverage[ds]?\b|\b\d+(\.\d+)?\s*x\b|\bx\s*\d+\b", re.I),
        "leverage is not allowed (R3); leverage stays 1.0",
    ),
    (re.compile(r"\bperp(etual)?s?\b", re.I), "perpetual contracts are blocked (R1)"),
    (re.compile(r"\bfutures?\b", re.I), "futures are blocked (R1)"),
    (re.compile(r"\bmargin\b", re.I), "margin trading is not allowed (R3)"),
    (
        re.compile(r"\boptions?\b|\bcfds?\b|\bbinar(y|ies)\b", re.I),
        "options, CFDs and binaries are blocked (R1, R2)",
    ),
)
_NUMBER = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_HEAD = re.compile(r"^\s*(BUY|LONG|SELL)\s+([A-Z][A-Z0-9]{0,11}(?:[/\-.][A-Z0-9]{1,12})?)\b", re.I)
_ENTRY = re.compile(r"^\s*(?:@|AT|ENTRY|PRICE)?\s*[:=]?\s*" + _NUMBER, re.I)
_STOP = re.compile(r"\b(?:SL|STOP(?:[\s-]?LOSS)?)\s*[:=@]?\s*" + _NUMBER, re.I)
_TARGET = re.compile(r"\b(?:TP1?|TARGET|TAKE[\s-]?PROFIT)\s*[:=@]?\s*" + _NUMBER, re.I)


class SignalRejectedError(ValueError):
    """The text is not an acceptable long-only spot entry."""

    def __init__(self, reason: str, *, haram: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.haram = haram  # True: blocked on Shariah grounds; counts against compliance


@dataclass(frozen=True, slots=True)
class ExternalSignal:
    provider_id: str
    message_id: str
    text: str
    received_at: datetime
    provider_kind: ProviderKind = ProviderKind.DISCORD_CHANNEL

    def __post_init__(self) -> None:
        if self.received_at.tzinfo is None:
            raise ValueError("received_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class ParsedSignal:
    symbol: str
    entry: Decimal
    stop: Decimal
    target: Decimal


def _dec(raw: str) -> Decimal:
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation as exc:  # pragma: no cover - the regex only admits digits
        raise SignalRejectedError(f"bad number: {raw}") from exc


def parse_signal(text: str) -> ParsedSignal:
    """Parse one signal line; raise :class:`SignalRejectedError` with a reason."""

    if len(text) > MAX_SIGNAL_CHARS:
        raise SignalRejectedError("signal text too long")
    for pattern, reason in _FORBIDDEN:
        if pattern.search(text):
            raise SignalRejectedError(reason, haram=True)
    head = _HEAD.match(text)
    if head is None:
        raise SignalRejectedError("not a signal: expected 'BUY <SYMBOL> <entry> SL <x> TP <y>'")
    if head.group(1).upper() == "SELL":
        raise SignalRejectedError(
            "opening SELL signals are not accepted; exits follow our own exit plan "
            "and selling without a position would be a short (R4)",
            haram=True,
        )
    entry = _ENTRY.match(text[head.end() :])
    stop = _STOP.search(text)
    target = _TARGET.search(text)
    if entry is None or stop is None or target is None:
        raise SignalRejectedError("signal needs an entry price, a stop (SL) and a target (TP)")
    return ParsedSignal(
        symbol=head.group(2).upper(),
        entry=_dec(entry.group(1)),
        stop=_dec(stop.group(1)),
        target=_dec(target.group(1)),
    )


def to_trade_plan(signal: ExternalSignal, parsed: ParsedSignal, confidence: Decimal) -> TradePlan:
    try:
        return TradePlan(
            symbol=parsed.symbol,
            entry_price=parsed.entry,
            profit_target=parsed.target,
            stop_loss=parsed.stop,
            invalidation_condition=f"price trades below the stop at {parsed.stop}",
            confidence=confidence,
            justification=(
                f"external signal from {signal.provider_id} (message {signal.message_id})"
            )[:1000],
        )
    except ValidationError as exc:
        first = exc.errors()[0].get("msg", "invalid plan")
        raise SignalRejectedError(f"invalid plan: {first}") from exc


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """Everything ``EntryPipeline.decide`` needs besides the plan."""

    instrument: Instrument
    financials: CompanyFinancials | None
    account: ShariahOrderContext
    risk_context: RiskContext
    closed_trades: Sequence[ClosedTrade]
    open_symbols: set[str]
    now: datetime
    crypto_asset: CryptoAsset | None = None


class EntryDecider(Protocol):
    def decide(
        self,
        plan: TradePlan,
        *,
        instrument: Instrument,
        financials: CompanyFinancials | None,
        account: ShariahOrderContext,
        crypto_asset: CryptoAsset | None = None,
        risk_context: RiskContext,
        closed_trades: Sequence[ClosedTrade],
        open_symbols: set[str],
        now: datetime,
    ) -> EntryDecision: ...


ContextProvider = Callable[[TradePlan], DecisionContext | None]


@dataclass(frozen=True, slots=True)
class IntakeResult:
    provider_id: str
    message_id: str
    accepted: bool
    reason: str
    plan: TradePlan | None = None
    decision: EntryDecision | None = None


@dataclass(slots=True)
class SignalIntake:
    """Paper-only bridge from external signals to the entry pipeline."""

    pipeline: EntryDecider
    context_for: ContextProvider
    marketplace: Marketplace | None = None
    confidence_prior: Decimal = Decimal("0.60")
    mode: str = PAPER

    def __post_init__(self) -> None:
        if self.mode != PAPER:
            raise ValueError("external signals are paper-only; live trading is not available")
        if not Decimal("0") <= self.confidence_prior <= Decimal("1"):
            raise ValueError("confidence_prior must be in [0, 1]")

    def _record(self, signal: ExternalSignal, shariah_passed: bool | None, approved: bool) -> None:
        if self.marketplace is None:
            return
        self.marketplace.ensure_provider(signal.provider_id, signal.provider_kind)
        self.marketplace.record(
            SignalOutcome(
                provider_id=signal.provider_id,
                signal_id=signal.message_id,
                at=signal.received_at,
                shariah_passed=shariah_passed,
                approved=approved,
            )
        )

    def process(self, signal: ExternalSignal) -> IntakeResult:
        def reject(reason: str, plan: TradePlan | None = None) -> IntakeResult:
            return IntakeResult(signal.provider_id, signal.message_id, False, reason, plan)

        try:
            plan = to_trade_plan(signal, parse_signal(signal.text), self.confidence_prior)
        except SignalRejectedError as exc:
            if exc.haram:
                self._record(signal, shariah_passed=False, approved=False)
            return reject(exc.reason)

        context = self.context_for(plan)
        if context is None:
            return reject("no market context for this symbol; unknown state blocks entries", plan)

        decision = self.pipeline.decide(
            plan,
            instrument=context.instrument,
            financials=context.financials,
            account=context.account,
            crypto_asset=context.crypto_asset,
            risk_context=context.risk_context,
            closed_trades=context.closed_trades,
            open_symbols=context.open_symbols,
            now=context.now,
        )
        reached = {stage.stage: stage.passed for stage in decision.stages}
        self._record(signal, shariah_passed=reached.get("shariah"), approved=decision.approved)
        if decision.approved:
            return IntakeResult(
                signal.provider_id, signal.message_id, True, "approved for paper", plan, decision
            )
        detail = next((s.detail for s in decision.stages if not s.passed), "blocked")
        return IntakeResult(
            signal.provider_id,
            signal.message_id,
            False,
            f"pipeline blocked at {decision.blocked_by}: {detail}",
            plan,
            decision,
        )
