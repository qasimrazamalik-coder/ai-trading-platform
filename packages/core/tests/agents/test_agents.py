from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError
from quant_os_core.agents import (
    AnalystCouncil,
    EntryPipeline,
    PlanGovernor,
    PlanPolicy,
    RuleAnalyst,
    TradePlan,
    parse_analyst_json,
    size_plan,
    validate_plan,
)
from quant_os_core.instruments import AssetClass, Instrument, InstrumentId, InstrumentType
from quant_os_core.risk_engine import (
    ClosedTrade,
    CooldownPeriod,
    ExitReason,
    ProtectionManager,
    RiskContext,
    RiskEngine,
)
from quant_os_core.shariah import (
    CompanyFinancials,
    ShariahGate,
    ShariahOrderContext,
    halal_risk_limits,
)

NOW = datetime(2026, 11, 21, 15, tzinfo=UTC)


def _plan(**kwargs: object) -> TradePlan:
    params: dict[str, object] = {
        "symbol": "AAPL",
        "entry_price": Decimal("100"),
        "profit_target": Decimal("110"),
        "stop_loss": Decimal("95"),
        "invalidation_condition": "daily close below the 50-day average",
        "confidence": Decimal("0.7"),
        "justification": "trend and momentum aligned",
    }
    params.update(kwargs)
    return TradePlan(**params)  # type: ignore[arg-type]


def test_plan_requires_exit_geometry_and_no_floats() -> None:
    with pytest.raises(ValidationError):
        _plan(stop_loss=Decimal("101"))
    with pytest.raises(ValidationError):
        _plan(confidence=0.7)
    with pytest.raises(ValidationError):
        _plan(invalidation_condition="")
    plan = _plan()
    assert plan.reward_to_risk == Decimal("2")
    assert plan.stop_distance == Decimal("0.05")


def test_plan_policy_rejections() -> None:
    policy = PlanPolicy()
    assert validate_plan(_plan(), policy).accepted
    assert not validate_plan(_plan(confidence=Decimal("0.5")), policy).accepted
    assert not validate_plan(_plan(profit_target=Decimal("105")), policy).accepted
    assert not validate_plan(_plan(stop_loss=Decimal("99.5")), policy).accepted
    assert not validate_plan(
        _plan(stop_loss=Decimal("80"), profit_target=Decimal("160")), policy
    ).accepted


def test_sizing_is_risk_based_capped_and_unlevered() -> None:
    policy = PlanPolicy()
    qty = size_plan(_plan(), policy, Decimal("50"), Decimal("50"))
    # risk budget: 50 * 1% * scale(0.7 -> 0.5+0.5*0.333=0.6667) / 5 = 0.0666; cap: 50*20%/100 = 0.1
    assert qty == Decimal("0.0666")
    assert qty * Decimal("100") <= Decimal("50")
    assert size_plan(_plan(), policy, Decimal("50"), Decimal("0")) == Decimal("0")
    with pytest.raises(ValueError):
        size_plan(_plan(), policy, Decimal("0"), Decimal("1"))


def test_governor_frequency_and_no_pyramiding() -> None:
    governor = PlanGovernor(PlanPolicy(max_trades_per_day=2))
    assert not governor.check(_plan(), {"aapl"}, NOW).accepted
    for hour in (1, 2):
        assert governor.check(_plan(), set(), NOW + timedelta(hours=hour)).accepted
        governor.record_entry(NOW + timedelta(hours=hour))
    assert not governor.check(_plan(), set(), NOW + timedelta(hours=3)).accepted
    assert governor.check(_plan(), set(), NOW + timedelta(days=1, hours=2)).accepted


def test_council_veto_shrink_and_bear_case() -> None:
    plan = _plan()
    strong_bear = AnalystCouncil(
        [RuleAnalyst("tech", Decimal("0.6")), RuleAnalyst("news", Decimal("-0.8"))]
    )
    assert not strong_bear.deliberate(plan).approved
    flagged = AnalystCouncil([RuleAnalyst("news", Decimal("0.9"), flag=True)])
    assert flagged.deliberate(plan).reason == "red flag raised"
    all_bull = AnalystCouncil([RuleAnalyst("a", Decimal("0.9")), RuleAnalyst("b", Decimal("0.9"))])
    decision = all_bull.deliberate(plan)
    assert decision.approved and decision.size_multiplier == Decimal("0.9")
    assert decision.bear_case  # devil's advocate always present
    mixed = AnalystCouncil([RuleAnalyst("a", Decimal("0.4")), RuleAnalyst("b", Decimal("-0.5"))])
    assert mixed.deliberate(plan).size_multiplier <= Decimal("1")
    with pytest.raises(ValueError):
        AnalystCouncil([])


def test_llm_reply_parsing_fails_closed() -> None:
    good = '{"analyst":"news","stance":"bearish","conviction":"0.4","rationale":"guidance cut"}'
    report = parse_analyst_json(good)
    assert report is not None and report.conviction == Decimal("0.4")
    assert parse_analyst_json("not json") is None
    assert (
        parse_analyst_json('{"analyst":"x","stance":"moon","conviction":"2","rationale":"y"}')
        is None
    )


def _pipeline(protections: ProtectionManager | None = None) -> EntryPipeline:
    return EntryPipeline(
        shariah=ShariahGate(),
        protections=protections or ProtectionManager([]),
        governor=PlanGovernor(PlanPolicy()),
        council=AnalystCouncil(
            [RuleAnalyst("trend", Decimal("0.6")), RuleAnalyst("sent", Decimal("0.3"))]
        ),
        risk=RiskEngine(halal_risk_limits()),
    )


def _call(
    pipe: EntryPipeline, instrument: Instrument, trades: list[ClosedTrade] | None = None
) -> object:
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
    return pipe.decide(
        _plan(),
        instrument=instrument,
        financials=financials,
        account=ShariahOrderContext(owned_quantity=Decimal("0"), free_cash=Decimal("50")),
        risk_context=RiskContext(
            equity=Decimal("50"),
            peak_equity=Decimal("50"),
            start_of_day_equity=Decimal("50"),
            positions=(),
            mark_prices={"AAPL": Decimal("100")},
            stop_prices={"AAPL": Decimal("95")},
        ),
        closed_trades=trades or [],
        open_symbols=set(),
        now=NOW,
    )


def _instrument(asset_class: AssetClass = AssetClass.EQUITY) -> Instrument:
    return Instrument(
        instrument_id=InstrumentId(venue="alpaca", symbol="AAPL"),
        asset_class=asset_class,
        instrument_type=InstrumentType.SPOT,
        base_currency="AAPL",
        quote_currency="USD",
        price_precision=2,
        quantity_precision=4,
    )


def test_pipeline_happy_path_produces_small_unlevered_order() -> None:
    pipe = _pipeline()
    decision = _call(pipe, _instrument())
    assert decision.approved, decision.stages  # type: ignore[attr-defined]
    assert decision.intent is not None  # type: ignore[attr-defined]
    assert decision.quantity * Decimal("100") <= Decimal("50")  # type: ignore[attr-defined]
    assert [s.stage for s in decision.stages] == [  # type: ignore[attr-defined]
        "plan",
        "shariah",
        "protections",
        "governor",
        "council",
        "risk",
    ]
    assert len(pipe.audit_log) == 1


def test_pipeline_blocks_crypto_and_cooldown() -> None:
    assert _call(_pipeline(), _instrument(AssetClass.CRYPTO)).blocked_by == "shariah"  # type: ignore[attr-defined]
    recent = [
        ClosedTrade("AAPL", NOW - timedelta(minutes=30), Decimal("0.01"), ExitReason.TAKE_PROFIT)
    ]
    cooled = _pipeline(ProtectionManager([CooldownPeriod(stop_duration=timedelta(hours=2))]))
    assert _call(cooled, _instrument(), recent).blocked_by == "protections"  # type: ignore[attr-defined]
