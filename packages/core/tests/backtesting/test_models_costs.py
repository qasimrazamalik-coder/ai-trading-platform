from decimal import Decimal

import pytest

from quant_os_core.backtesting.costs import FinancingModel, ProportionalFeeModel
from quant_os_core.backtesting.models import OrderRequest, OrderSide, OrderType


def test_order_rejects_float(instrument):
    with pytest.raises(ValueError):
        OrderRequest(instrument_id=instrument, side="buy", quantity=1.2)


def test_limit_requires_limit_price(instrument):
    with pytest.raises(ValueError):
        OrderRequest(
            instrument_id=instrument,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            order_type=OrderType.LIMIT,
        )


def test_market_rejects_limit_price(instrument):
    with pytest.raises(ValueError):
        OrderRequest(
            instrument_id=instrument,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            limit_price=Decimal("99"),
        )


def test_fee_model_basis_points():
    model = ProportionalFeeModel(taker_bps=Decimal("10"))
    assert model.calculate(notional=Decimal("1000")) == Decimal("1")


def test_fee_model_uses_maker_rate():
    model = ProportionalFeeModel(taker_bps=Decimal("10"), maker_bps=Decimal("2"))
    assert model.calculate(notional=Decimal("1000"), is_maker=True) == Decimal("0.2")


def test_fee_model_minimum():
    model = ProportionalFeeModel(taker_bps=Decimal("1"), minimum_fee=Decimal("0.50"))
    assert model.calculate(notional=Decimal("100")) == Decimal("0.50")


def test_financing_long_and_short():
    model = FinancingModel(
        annual_long_rate=Decimal("0.365"),
        annual_short_rate=Decimal("0.73"),
        periods_per_year=Decimal("365"),
    )
    assert model.charge(signed_notional=Decimal("1000")) == Decimal("1.000")
    assert model.charge(signed_notional=Decimal("-1000")) == Decimal("2.00")
