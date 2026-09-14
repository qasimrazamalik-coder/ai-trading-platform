from decimal import Decimal

import pytest

from quant_os_core.events import OrderIntent, Side
from quant_os_core.risk_engine import RiskAction, RiskContext, RiskEngine, RiskLimits, RiskPosition


def _intent(quantity: str = "1") -> OrderIntent:
    return OrderIntent(
        source="guard",
        strategy_id="guard",
        symbol="BTC",
        side=Side.BUY,
        quantity=Decimal(quantity),
    )


def _context(**kwargs: object) -> RiskContext:
    params: dict[str, object] = {
        "equity": Decimal("1000"),
        "peak_equity": Decimal("1000"),
        "start_of_day_equity": Decimal("1000"),
        "positions": (),
        "mark_prices": {"BTC": Decimal("100")},
    }
    params.update(kwargs)
    return RiskContext(**params)  # type: ignore[arg-type]


def test_risk_models_reject_invalid_limits_and_prices() -> None:
    with pytest.raises(ValueError):
        RiskLimits(max_position_fraction=Decimal("0"))
    with pytest.raises(ValueError):
        RiskLimits(correlation_threshold=2.0)
    with pytest.raises(ValueError):
        RiskPosition("BTC", Decimal("1"), Decimal("0"))
    with pytest.raises(ValueError):
        _context(mark_prices={"BTC": Decimal("0")})


def test_missing_mark_price_rejects_order() -> None:
    result = RiskEngine().evaluate(_intent(), _context(mark_prices={"ETH": Decimal("10")}))
    assert result.action is RiskAction.REJECT
    assert result.approved_intent is None


def test_resized_intent_carries_approved_quantity() -> None:
    result = RiskEngine().evaluate(_intent("5"), _context())
    assert result.action is RiskAction.RESIZE
    assert result.approved_intent is not None
    assert result.approved_intent.quantity == result.approved_quantity
    assert result.approved_intent.event_id == result.order_intent.event_id


def test_equity_below_minimum_halts() -> None:
    limits = RiskLimits(min_equity=Decimal("100"))
    ctx = _context(
        equity=Decimal("50"),
        peak_equity=Decimal("50"),
        start_of_day_equity=Decimal("50"),
    )
    assert RiskEngine(limits).evaluate(_intent(), ctx).action is RiskAction.HALT


def test_additional_risk_model_validation_guards() -> None:
    with pytest.raises(ValueError):
        RiskLimits(max_gross_leverage=Decimal("0"))
    with pytest.raises(ValueError):
        RiskLimits(min_equity=Decimal("0"))
    with pytest.raises(ValueError):
        RiskLimits(correlation_threshold=float("nan"))
    with pytest.raises(ValueError):
        RiskPosition(" ", Decimal("1"), Decimal("10"))
    with pytest.raises(ValueError):
        RiskPosition("BTC", Decimal("1"), Decimal("10"), stop_price=Decimal("0"))
    with pytest.raises(ValueError):
        _context(equity=Decimal("0"))


def test_position_and_portfolio_capacity_rejections() -> None:
    at_cap = RiskPosition("BTC", Decimal("1"), Decimal("100"))
    result = RiskEngine().evaluate(
        _intent("1"),
        _context(positions=(at_cap,)),
    )
    assert result.action is RiskAction.REJECT
    assert "position limit" in result.reasons[0]

    limits = RiskLimits(
        max_position_fraction=Decimal("1"),
        max_gross_leverage=Decimal("0.05"),
        max_net_leverage=Decimal("0.05"),
        max_correlated_fraction=Decimal("1"),
        max_group_fraction=Decimal("1"),
    )
    resized = RiskEngine(limits).evaluate(_intent("1"), _context())
    assert resized.action is RiskAction.RESIZE
    assert resized.approved_quantity <= Decimal("0.5")
