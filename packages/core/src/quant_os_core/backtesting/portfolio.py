"""Decimal-based cash, position and mark-to-market accounting."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from quant_os_core.backtesting.models import Fill, PositionSnapshot
from quant_os_core.instruments import InstrumentId


@dataclass(slots=True)
class Position:
    instrument_id: InstrumentId
    quantity: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    def apply_fill(self, fill: Fill) -> None:
        signed = fill.signed_quantity
        old_qty = self.quantity
        new_qty = old_qty + signed

        if old_qty == 0 or (old_qty > 0 and signed > 0) or (old_qty < 0 and signed < 0):
            old_notional = abs(old_qty) * self.average_price
            add_notional = abs(signed) * fill.price
            total_abs = abs(old_qty) + abs(signed)
            self.average_price = (old_notional + add_notional) / total_abs
            self.quantity = new_qty
            return

        closing_qty = min(abs(old_qty), abs(signed))
        direction = Decimal("1") if old_qty > 0 else Decimal("-1")
        self.realized_pnl += closing_qty * (fill.price - self.average_price) * direction
        self.quantity = new_qty

        if new_qty == 0:
            self.average_price = Decimal("0")
        elif old_qty * new_qty < 0:
            self.average_price = fill.price

    def unrealized_pnl(self, market_price: Decimal | None) -> Decimal:
        if market_price is None or self.quantity == 0:
            return Decimal("0")
        direction = Decimal("1") if self.quantity > 0 else Decimal("-1")
        return abs(self.quantity) * (market_price - self.average_price) * direction


class Portfolio:
    def __init__(self, initial_cash: Decimal) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be > 0")
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.marks: dict[str, Decimal] = {}
        self.total_fees = Decimal("0")
        self.total_financing = Decimal("0")

    def position_for(self, instrument_id: InstrumentId) -> Position:
        key = instrument_id.key
        if key not in self.positions:
            self.positions[key] = Position(instrument_id=instrument_id)
        return self.positions[key]

    def apply_fill(self, fill: Fill) -> None:
        cash_delta = -fill.signed_quantity * fill.price - fill.fee
        self.cash += cash_delta
        self.total_fees += fill.fee
        self.position_for(fill.instrument_id).apply_fill(fill)

    def apply_financing(self, charge: Decimal) -> None:
        if charge < 0:
            raise ValueError("financing charge cannot be negative")
        self.cash -= charge
        self.total_financing += charge

    def mark(self, instrument_id: InstrumentId, price: Decimal) -> None:
        if price <= 0:
            raise ValueError("mark price must be > 0")
        self.marks[instrument_id.key] = price

    @property
    def equity(self) -> Decimal:
        value = self.cash
        for key, position in self.positions.items():
            mark = self.marks.get(key, position.average_price)
            value += position.quantity * mark
        return value

    @property
    def gross_exposure(self) -> Decimal:
        return sum(
            (abs(position.quantity * self.marks.get(key, position.average_price))
             for key, position in self.positions.items()),
            start=Decimal("0"),
        )

    @property
    def net_exposure(self) -> Decimal:
        return sum(
            (position.quantity * self.marks.get(key, position.average_price)
             for key, position in self.positions.items()),
            start=Decimal("0"),
        )

    def snapshots(self) -> tuple[PositionSnapshot, ...]:
        result: list[PositionSnapshot] = []
        for key, position in sorted(self.positions.items()):
            mark = self.marks.get(key)
            result.append(
                PositionSnapshot(
                    instrument_id=position.instrument_id,
                    quantity=position.quantity,
                    average_price=position.average_price,
                    realized_pnl=position.realized_pnl,
                    market_price=mark,
                    unrealized_pnl=position.unrealized_pnl(mark),
                )
            )
        return tuple(result)
