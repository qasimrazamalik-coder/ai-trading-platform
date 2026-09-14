from decimal import Decimal

from quant_os_core.backtesting.models import OrderSide
from quant_os_core.backtesting.slippage import BpsSlippageModel


def test_fixed_slippage_is_adverse():
    model = BpsSlippageModel(fixed_bps=Decimal("10"))
    buy = model.adjust(
        reference_price=Decimal("100"),
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        bar_volume=Decimal("100"),
    )
    sell = model.adjust(
        reference_price=Decimal("100"),
        side=OrderSide.SELL,
        quantity=Decimal("1"),
        bar_volume=Decimal("100"),
    )
    assert buy == Decimal("100.100")
    assert sell == Decimal("99.900")


def test_market_impact_increases_with_participation():
    model = BpsSlippageModel(impact_bps_at_full_volume=Decimal("100"))
    small = model.adjust(
        reference_price=Decimal("100"),
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        bar_volume=Decimal("100"),
    )
    large = model.adjust(
        reference_price=Decimal("100"),
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        bar_volume=Decimal("100"),
    )
    assert large > small > Decimal("100")
