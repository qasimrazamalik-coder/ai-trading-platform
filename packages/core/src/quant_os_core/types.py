"""Financial value objects.

Floats are deliberately rejected for money-like values to avoid binary floating-point
surprises in the execution and accounting layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

type DecimalLike = Decimal | int | str


def _decimal(value: DecimalLike, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool")
    if isinstance(value, float):
        raise TypeError(f"{field} must use Decimal, int, or str; floats are forbidden")
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"{field} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class Price:
    value: Decimal

    def __init__(self, value: DecimalLike) -> None:
        parsed = _decimal(value, field="price")
        if parsed <= 0:
            raise ValueError("price must be > 0")
        object.__setattr__(self, "value", parsed)


@dataclass(frozen=True, slots=True)
class Quantity:
    value: Decimal

    def __init__(self, value: DecimalLike) -> None:
        parsed = _decimal(value, field="quantity")
        if parsed <= 0:
            raise ValueError("quantity must be > 0")
        object.__setattr__(self, "value", parsed)


@dataclass(frozen=True, slots=True)
class Money:
    amount: Decimal
    currency: str

    def __init__(self, amount: DecimalLike, currency: str) -> None:
        parsed = _decimal(amount, field="amount")
        normalized = currency.strip().upper()
        if len(normalized) != 3 or not normalized.isalpha():
            raise ValueError("currency must be a 3-letter alphabetic code")
        object.__setattr__(self, "amount", parsed)
        object.__setattr__(self, "currency", normalized)
