from decimal import Decimal

import pytest

from quant_os_core.events import OrderIntent, Side
from quant_os_core.risk_engine import (
    RiskAction,
    RiskContext,
    RiskEngine,
    RiskLimits,
    RiskPosition,
)


def intent(side: Side, quantity: str = "1", symbol: str = "BTC") -> OrderIntent:
    return OrderIntent(
        source="strategy",
        strategy_id="test",
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
    )


def context(
    *,
    positions: tuple[RiskPosition, ...] = (),
    equity: str = "1000",
    peak: str = "1000",
    start_day: str = "1000",
    marks: dict[str, Decimal] | None = None,
    **kwargs: object,
) -> RiskContext:
    return RiskContext(
        equity=Decimal(equity),
        peak_equity=Decimal(peak),
        start_of_day_equity=Decimal(start_day),
        positions=positions,
        mark_prices=marks or {"BTC": Decimal("100")},
        **kwargs,
    )


def test_approves_order_inside_limits() -> None:
    result = RiskEngine().evaluate(intent(Side.BUY), context())
    assert result.action is RiskAction.APPROVE
    assert result.approved_quantity == Decimal("1")


def test_resizes_order_to_position_limit() -> None:
    engine = RiskEngine(RiskLimits(max_position_fraction=Decimal("0.10")))
    result = engine.evaluate(intent(Side.BUY, "5"), context())
    assert result.action is RiskAction.RESIZE
    assert result.approved_quantity == Decimal("1")


def test_halts_when_kill_switch_or_stale_data_active() -> None:
    engine = RiskEngine()
    assert engine.evaluate(intent(Side.BUY), context(kill_switch=True)).action is RiskAction.HALT
    stale_result = engine.evaluate(intent(Side.BUY), context(stale_market_data=True))
    assert stale_result.action is RiskAction.HALT


def test_halts_at_daily_loss_and_drawdown_limits() -> None:
    limits = RiskLimits(
        max_daily_loss_fraction=Decimal("0.03"),
        max_drawdown_fraction=Decimal("0.10"),
    )
    engine = RiskEngine(limits)
    daily = context(equity="960", peak="1000", start_day="1000")
    drawdown = context(equity="890", peak="1000", start_day="890")
    assert engine.evaluate(intent(Side.BUY), daily).action is RiskAction.HALT
    assert engine.evaluate(intent(Side.BUY), drawdown).action is RiskAction.HALT


def test_risk_reducing_order_allowed_while_gross_limit_already_breached() -> None:
    position = RiskPosition("BTC", Decimal("20"), Decimal("100"))
    limits = RiskLimits(
        max_position_fraction=Decimal("1"),
        max_gross_leverage=Decimal("1.5"),
        max_net_leverage=Decimal("1.5"),
        max_correlated_fraction=Decimal("1"),
        max_group_fraction=Decimal("1"),
    )
    ctx = context(positions=(position,))
    result = RiskEngine(limits).evaluate(intent(Side.SELL, "5"), ctx)
    assert result.approved
    assert result.approved_quantity == Decimal("5")
    assert result.projected_gross_leverage < Decimal("2")


def test_shorting_can_be_disabled() -> None:
    engine = RiskEngine(RiskLimits(allow_short=False))
    result = engine.evaluate(intent(Side.SELL), context())
    assert result.action is RiskAction.REJECT
    assert "shorting" in result.reasons[0]


def test_portfolio_heat_can_resize_order() -> None:
    limits = RiskLimits(
        max_position_fraction=Decimal("1"),
        max_portfolio_heat=Decimal("0.02"),
        max_correlated_fraction=Decimal("1"),
        max_group_fraction=Decimal("1"),
    )
    ctx = context(stop_prices={"BTC": Decimal("90")})
    result = RiskEngine(limits).evaluate(intent(Side.BUY, "5"), ctx)
    assert result.action is RiskAction.RESIZE
    assert result.approved_quantity == pytest.approx(Decimal("2"), abs=Decimal("0.000001"))


def test_group_and_correlation_concentration_limit_capacity() -> None:
    eth = RiskPosition("ETH", Decimal("2"), Decimal("100"), group="CRYPTO")
    limits = RiskLimits(
        max_position_fraction=Decimal("1"),
        max_group_fraction=Decimal("0.25"),
        max_correlated_fraction=Decimal("0.25"),
    )
    ctx = context(
        positions=(eth,),
        marks={"BTC": Decimal("100"), "ETH": Decimal("100")},
        symbol_groups={"BTC": "CRYPTO", "ETH": "CRYPTO"},
        correlations={(("BTC", "ETH")): 0.95},
    )
    result = RiskEngine(limits).evaluate(intent(Side.BUY, "2"), ctx)
    assert result.action is RiskAction.RESIZE
    assert result.approved_quantity == pytest.approx(Decimal("0.5"), abs=Decimal("0.000001"))
