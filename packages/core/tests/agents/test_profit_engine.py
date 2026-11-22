from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest

from quant_os_core.agents import (
    AnalystCouncil,
    CostModel,
    EntryPipeline,
    PlanGovernor,
    RiskDial,
    RuleAnalyst,
    TradePlan,
    allocate_sleeves,
    check_edge,
    deposit_plan,
    dial_policy,
    expected_edge,
)
from quant_os_core.instruments import AssetClass, Instrument, InstrumentId, InstrumentType
from quant_os_core.risk_engine import ProtectionManager, RiskContext, RiskEngine
from quant_os_core.shariah import (
    CryptoAsset,
    ShariahGate,
    ShariahOrderContext,
    ShariahProfile,
    halal_risk_limits,
)


def _plan(entry: str, stop: str, target: str, confidence: str) -> TradePlan:
    return TradePlan(
        symbol="BTC/USD",
        entry_price=Decimal(entry),
        profit_target=Decimal(target),
        stop_loss=Decimal(stop),
        invalidation_condition="closes back inside the range",
        confidence=Decimal(confidence),
        justification="test",
    )


def test_expected_edge_formula() -> None:
    plan = _plan("100", "98", "104", "0.5")  # 0.5*4 - 0.5*2 = 1 -> 1% of entry
    assert expected_edge(plan) == Decimal("0.01")


def test_edge_filter_blocks_trades_that_lose_to_costs() -> None:
    model = CostModel()
    crypto_rt = model.round_trip(AssetClass.CRYPTO)  # maker entry + taker exit
    assert crypto_rt == Decimal("0.0015") + Decimal("0.0025") + Decimal("0.0008")
    stock_rt = model.round_trip(AssetClass.EQUITY)
    assert stock_rt == Decimal("0.0005")
    scalp = _plan("100", "99.7", "100.45", "0.6")  # edge 0.6*0.45-0.4*0.3 = 0.15%
    assert not check_edge(scalp, AssetClass.CRYPTO, model).accepted
    assert check_edge(scalp, AssetClass.EQUITY, model).accepted  # cheap venue, same idea
    losing = _plan("100", "95", "105", "0.4")
    assert check_edge(losing, AssetClass.EQUITY, model).reason.startswith("negative")
    taker_only = CostModel(prefer_maker=False)
    assert taker_only.round_trip(AssetClass.CRYPTO) > crypto_rt
    with pytest.raises(ValueError):
        CostModel(min_edge_to_cost=Decimal("0.5"))


def test_pipeline_edge_stage_blocks_before_shariah() -> None:
    pipeline = EntryPipeline(
        shariah=ShariahGate(ShariahProfile.spot_crypto_and_day_trading()),
        protections=ProtectionManager([]),
        governor=PlanGovernor(dial_policy(RiskDial.BALANCED)),
        council=AnalystCouncil([RuleAnalyst("trend", Decimal("0.6"))]),
        risk=RiskEngine(halal_risk_limits()),
        cost_model=CostModel(),
    )
    instrument = Instrument(
        instrument_id=InstrumentId(venue="ALPACA", symbol="BTC/USD"),
        asset_class=AssetClass.CRYPTO,
        instrument_type=InstrumentType.SPOT,
        base_currency="BTC",
        quote_currency="USD",
        price_precision=2,
        quantity_precision=6,
    )
    decision = pipeline.decide(
        _plan("100", "98.9", "101.7", "0.6"),
        instrument=instrument,
        financials=None,
        crypto_asset=CryptoAsset("BTC", "layer1"),
        account=ShariahOrderContext(owned_quantity=Decimal("0"), free_cash=Decimal("50")),
        risk_context=RiskContext(
            equity=Decimal("50"),
            peak_equity=Decimal("50"),
            start_of_day_equity=Decimal("50"),
            positions=(),
            mark_prices={"BTC/USD": Decimal("100")},
            stop_prices={"BTC/USD": Decimal("98.9")},
        ),
        closed_trades=[],
        open_symbols=set(),
        now=datetime(2026, 11, 21, tzinfo=UTC),
    )
    assert decision.blocked_by == "edge"


def test_risk_dial_presets() -> None:
    growth = dial_policy(RiskDial.GROWTH)
    assert growth.risk_per_trade == Decimal("0.02") and growth.max_position_fraction == Decimal(
        "0.20"
    )
    assert dial_policy(RiskDial.CONSERVATIVE).risk_per_trade == Decimal("0.005")


def test_sleeve_allocator_risk_parity_cap_and_no_leverage() -> None:
    vols = np.array([0.15, 0.22, 0.45, 0.15])
    corr = np.array(
        [
            [1.0, 0.4, 0.3, 0.0],
            [0.4, 1.0, 0.2, 0.0],
            [0.3, 0.2, 1.0, 0.1],
            [0.0, 0.0, 0.1, 1.0],
        ]
    )
    cov = np.outer(vols, vols) * corr
    names = ["us_stocks", "psx", "crypto", "gold"]
    low = allocate_sleeves(names, cov, vol_target=0.05)
    assert low.cash > 0 and low.expected_vol == pytest.approx(0.05)
    high = allocate_sleeves(names, cov, vol_target=0.50)
    assert high.cash == pytest.approx(0.0, abs=1e-9)  # fully invested, never levered
    assert sum(high.weights.values()) == pytest.approx(1.0)
    assert high.weights["crypto"] < high.weights["gold"]  # risk parity: volatile sleeve smaller
    capped = allocate_sleeves(names, cov, vol_target=0.5, max_sleeve=0.3)
    assert max(capped.weights.values()) <= 0.3 + 1e-9
    with pytest.raises(ValueError):
        allocate_sleeves(names[:3], cov, vol_target=0.1)


def test_deposit_plan_matches_closed_form() -> None:
    path = deposit_plan(Decimal("50"), Decimal("50"), Decimal("0.10"), 36)
    m = (1.10) ** (1 / 12) - 1
    closed = 50 * (1 + m) ** 36 + 50 * ((1 + m) ** 36 - 1) / m
    assert float(path[-1].balance) == pytest.approx(closed, abs=0.05)
    assert path[-1].deposited == Decimal("1850")
    with pytest.raises(ValueError):
        deposit_plan(Decimal("50"), Decimal("50"), Decimal("0.1"), 0)
