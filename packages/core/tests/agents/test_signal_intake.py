from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_os_core.agents import (
    AnalystCouncil,
    EntryPipeline,
    PlanGovernor,
    PlanPolicy,
    RuleAnalyst,
    TradePlan,
)
from quant_os_core.agents.marketplace import Marketplace, ProviderKind
from quant_os_core.agents.pipeline import EntryDecision, StageResult
from quant_os_core.agents.signal_intake import (
    DecisionContext,
    ExternalSignal,
    SignalIntake,
    SignalRejectedError,
    parse_signal,
)
from quant_os_core.instruments import AssetClass, Instrument, InstrumentId, InstrumentType
from quant_os_core.risk_engine import ClosedTrade, ProtectionManager, RiskContext, RiskEngine
from quant_os_core.shariah import (
    CompanyFinancials,
    CryptoAsset,
    ShariahGate,
    ShariahOrderContext,
    halal_risk_limits,
)

NOW = datetime(2026, 11, 22, 15, tzinfo=UTC)


def _signal(text: str, message_id: str = "1") -> ExternalSignal:
    return ExternalSignal("discord:42", message_id, text, NOW)


def _instrument(symbol: str = "AAPL", asset_class: AssetClass = AssetClass.EQUITY) -> Instrument:
    return Instrument(
        instrument_id=InstrumentId(venue="alpaca", symbol=symbol),
        asset_class=asset_class,
        instrument_type=InstrumentType.SPOT,
        base_currency=symbol.split("/")[0],
        quote_currency="USD",
        price_precision=2,
        quantity_precision=4,
    )


def _context(plan: TradePlan, asset_class: AssetClass = AssetClass.EQUITY) -> DecisionContext:
    financials = CompanyFinancials(
        sector="technology",
        market_cap=Decimal("1000"),
        interest_bearing_debt=Decimal("100"),
        interest_bearing_cash_and_securities=Decimal("50"),
        total_revenue=Decimal("400"),
        noncompliant_revenue=Decimal("4"),
        total_assets=Decimal("900"),
        tangible_assets=Decimal("600"),
        shares_outstanding=Decimal("100"),
    )
    return DecisionContext(
        instrument=_instrument(plan.symbol, asset_class),
        financials=financials,
        account=ShariahOrderContext(owned_quantity=Decimal("0"), free_cash=Decimal("500")),
        risk_context=RiskContext(
            equity=Decimal("500"),
            peak_equity=Decimal("500"),
            start_of_day_equity=Decimal("500"),
            positions=(),
            mark_prices={plan.symbol: plan.entry_price},
            stop_prices={plan.symbol: plan.stop_loss},
        ),
        closed_trades=(),
        open_symbols=set(),
        now=NOW,
    )


@dataclass
class FakePipeline:
    approve: bool = True
    blocked_stage: str = "shariah"
    calls: list[tuple[TradePlan, dict[str, object]]] = field(default_factory=list)

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
    ) -> EntryDecision:
        self.calls.append((plan, {"instrument": instrument, "now": now}))
        if self.approve:
            stages = (StageResult("plan", True, "ok"), StageResult("shariah", True, "ok"))
            return EntryDecision(True, Decimal("1"), None, stages)
        stages = (StageResult("plan", True, "ok"), StageResult(self.blocked_stage, False, "no"))
        return EntryDecision(False, Decimal("0"), None, stages)


def test_parse_examples() -> None:
    parsed = parse_signal("BUY BTC/USD 65000 SL 62000 TP 72000")
    assert (parsed.symbol, parsed.entry, parsed.stop, parsed.target) == (
        "BTC/USD",
        Decimal("65000"),
        Decimal("62000"),
        Decimal("72000"),
    )
    parsed = parse_signal("long aapl @ 190.50 stop loss: 185 take profit 1,200.25")
    assert parsed.symbol == "AAPL"
    assert parsed.entry == Decimal("190.50")
    assert parsed.target == Decimal("1200.25")
    assert parse_signal("BUY ETH-USD entry 3000 SL=2900 TP1 3300 TP2 3600").target == Decimal(
        "3300"
    )


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("SHORT BTC/USD 65000 SL 68000 TP 60000", "short selling"),
        ("BUY BTC/USD 10x 65000 SL 62000 TP 72000", "leverage"),
        ("BUY BTC/USD x20 65000 SL 62000 TP 72000", "leverage"),
        ("BUY BTC PERP 65000 SL 62000 TP 72000", "perpetual"),
        ("BUY ES futures 5000 SL 4900 TP 5200", "futures"),
        ("BUY AAPL on margin 190 SL 185 TP 200", "margin"),
        ("BUY AAPL options 190 SL 185 TP 200", "options"),
        ("SELL BTC/USD 65000 SL 68000 TP 60000", "opening SELL"),
    ],
)
def test_haram_signals_are_rejected_with_reason(text: str, fragment: str) -> None:
    with pytest.raises(SignalRejectedError, match=fragment) as info:
        parse_signal(text)
    assert info.value.haram


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("gm everyone, big day today", "not a signal"),
        ("BUY AAPL 190 TP 200", "needs an entry price"),
        ("BUY AAPL " + "9" * 600, "too long"),
    ],
)
def test_malformed_signals(text: str, fragment: str) -> None:
    with pytest.raises(SignalRejectedError, match=fragment) as info:
        parse_signal(text)
    assert not info.value.haram


def test_accepted_signal_runs_pipeline_and_is_attributed() -> None:
    market = Marketplace()
    pipe = FakePipeline()
    intake = SignalIntake(pipe, _context, market)
    result = intake.process(_signal("BUY AAPL 100 SL 95 TP 110", "m1"))
    assert result.accepted and result.reason == "approved for paper"
    assert result.plan is not None
    assert result.plan.confidence == Decimal("0.60")
    assert "discord:42" in result.plan.justification
    assert len(pipe.calls) == 1
    record = market.provider("discord:42")
    assert record.kind is ProviderKind.DISCORD_CHANNEL
    (outcome,) = market.outcomes("discord:42")
    assert outcome.signal_id == "m1"
    assert outcome.approved and outcome.shariah_passed is True


def test_haram_signal_counts_against_compliance_and_never_reaches_pipeline() -> None:
    market = Marketplace()
    pipe = FakePipeline()
    result = SignalIntake(pipe, _context, market).process(
        _signal("BUY BTC/USD 10x 65000 SL 62000 TP 72000")
    )
    assert not result.accepted
    assert "leverage" in result.reason
    assert pipe.calls == []
    (outcome,) = market.outcomes("discord:42")
    assert outcome.shariah_passed is False and not outcome.approved


def test_chatter_and_bad_geometry_are_not_graded() -> None:
    market = Marketplace()
    intake = SignalIntake(FakePipeline(), _context, market)
    assert "not a signal" in intake.process(_signal("hello")).reason
    bad = intake.process(_signal("BUY AAPL 100 SL 105 TP 110", "2"))
    assert bad.reason.startswith("invalid plan")
    assert "discord:42" not in [r.provider_id for r in market.providers()]


def test_missing_context_blocks_entry() -> None:
    pipe = FakePipeline()
    result = SignalIntake(pipe, lambda plan: None).process(_signal("BUY AAPL 100 SL 95 TP 110"))
    assert not result.accepted
    assert "unknown state" in result.reason
    assert pipe.calls == []


def test_pipeline_block_is_reported_and_recorded() -> None:
    market = Marketplace()
    intake = SignalIntake(FakePipeline(approve=False), _context, market)
    result = intake.process(_signal("BUY AAPL 100 SL 95 TP 110"))
    assert not result.accepted
    assert result.reason == "pipeline blocked at shariah: no"
    assert market.outcomes("discord:42")[0].shariah_passed is False

    early = SignalIntake(FakePipeline(approve=False, blocked_stage="plan"), _context, market)
    early.process(_signal("BUY AAPL 100 SL 95 TP 110", "2"))
    assert market.outcomes("discord:42")[1].shariah_passed is None


def test_paper_only_and_prior_validation() -> None:
    with pytest.raises(ValueError, match="paper-only"):
        SignalIntake(FakePipeline(), _context, mode="live")
    with pytest.raises(ValueError, match="confidence_prior"):
        SignalIntake(FakePipeline(), _context, confidence_prior=Decimal("2"))
    with pytest.raises(ValueError, match="timezone-aware"):
        ExternalSignal("p", "1", "x", datetime(2026, 1, 1))


def test_real_pipeline_blocks_crypto_without_profile() -> None:
    pipe = EntryPipeline(
        shariah=ShariahGate(),
        protections=ProtectionManager([]),
        governor=PlanGovernor(PlanPolicy()),
        council=AnalystCouncil(
            [RuleAnalyst("trend", Decimal("0.6")), RuleAnalyst("sent", Decimal("0.3"))]
        ),
        risk=RiskEngine(halal_risk_limits()),
    )
    market = Marketplace()
    intake = SignalIntake(pipe, lambda plan: _context(plan, AssetClass.CRYPTO), market)
    result = intake.process(_signal("BUY BTC/USD 65000 SL 62000 TP 72000"))
    assert not result.accepted
    assert result.decision is not None and result.decision.blocked_by == "shariah"
    assert market.outcomes("discord:42")[0].shariah_passed is False

    equity = SignalIntake(pipe, _context, market).process(_signal("BUY AAPL 100 SL 95 TP 110", "2"))
    assert equity.accepted, equity.reason
    assert equity.decision is not None and equity.decision.intent is not None
