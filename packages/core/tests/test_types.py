from decimal import Decimal

import pytest

from quant_os_core.types import Money, Price, Quantity


def test_float_is_rejected_for_price() -> None:
    with pytest.raises(TypeError):
        Price(1.1)  # type: ignore[arg-type]


def test_money_normalizes_currency() -> None:
    money = Money("12.50", "usd")
    assert money.amount == Decimal("12.50")
    assert money.currency == "USD"


@pytest.mark.parametrize(
    "value",
    [
        Decimal("0.00000001"),
        Decimal("1"),
        Decimal("123.456789"),
        Decimal("1000000000"),
    ],
)
def test_positive_decimal_price_round_trips(value: Decimal) -> None:
    assert Price(value).value == value


def test_quantity_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Quantity("0")
