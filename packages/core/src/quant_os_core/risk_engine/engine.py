"""Independent pre-trade portfolio risk engine."""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

from quant_os_core.events import OrderIntent, Side
from quant_os_core.risk_engine.models import (
    RiskAction,
    RiskContext,
    RiskEvaluation,
    RiskLimits,
    RiskPosition,
)

ZERO = Decimal("0")
ONE = Decimal("1")


def _signed_quantity(intent: OrderIntent, quantity: Decimal | None = None) -> Decimal:
    qty = intent.quantity if quantity is None else quantity
    return qty if intent.side is Side.BUY else -qty


def _position_map(context: RiskContext) -> dict[str, RiskPosition]:
    return {position.symbol: position for position in context.positions}


def _projected_position_notional(
    intent: OrderIntent,
    context: RiskContext,
    quantity: Decimal,
) -> tuple[Decimal, Decimal, Decimal]:
    price = context.mark_prices.get(intent.symbol)
    if price is None:
        raise ValueError(f"missing mark price for {intent.symbol}")
    positions = _position_map(context)
    current = positions.get(intent.symbol)
    current_quantity = ZERO if current is None else current.quantity
    new_quantity = current_quantity + _signed_quantity(intent, quantity)
    current_notional = current_quantity * price
    new_notional = new_quantity * price
    return current_notional, new_notional, price


def _projected_exposures(
    intent: OrderIntent,
    context: RiskContext,
    quantity: Decimal,
) -> tuple[Decimal, Decimal]:
    current_notional, new_notional, _ = _projected_position_notional(intent, context, quantity)
    gross = context.gross_exposure - abs(current_notional) + abs(new_notional)
    net = context.net_exposure - current_notional + new_notional
    return gross, net


def _projected_heat(
    intent: OrderIntent,
    context: RiskContext,
    quantity: Decimal,
) -> Decimal:
    current_notional, new_notional, price = _projected_position_notional(intent, context, quantity)
    del current_notional
    positions = _position_map(context)
    current = positions.get(intent.symbol)
    old_heat = ZERO if current is None else current.heat
    stop = context.stop_prices.get(intent.symbol)
    if stop is None and current is not None:
        stop = current.stop_price
    if stop is None:
        new_heat = ZERO
    else:
        new_quantity = new_notional / price
        new_heat = abs(new_quantity) * abs(price - stop)
    return context.portfolio_heat - old_heat + new_heat


def _group_exposure(
    intent: OrderIntent,
    context: RiskContext,
    quantity: Decimal,
) -> Decimal:
    group = context.symbol_groups.get(intent.symbol)
    if group is None:
        current = _position_map(context).get(intent.symbol)
        group = None if current is None else current.group
    if group is None:
        return ZERO
    _, new_notional, _ = _projected_position_notional(intent, context, quantity)
    total = ZERO
    for position in context.positions:
        position_group = context.symbol_groups.get(position.symbol, position.group)
        if position_group == group and position.symbol != intent.symbol:
            total += position.absolute_notional
    return total + abs(new_notional)


def _correlated_exposure(
    intent: OrderIntent,
    context: RiskContext,
    quantity: Decimal,
    threshold: float,
) -> Decimal:
    _, new_notional, _ = _projected_position_notional(intent, context, quantity)
    total = abs(new_notional)
    for position in context.positions:
        if position.symbol == intent.symbol:
            continue
        correlation = context.correlations.get((intent.symbol, position.symbol))
        if correlation is None:
            correlation = context.correlations.get((position.symbol, intent.symbol))
        if correlation is not None and abs(correlation) >= threshold:
            total += position.absolute_notional
    return total



class RiskEngine:
    """Deterministic independent risk gate.

    Strategy intent is advisory. This engine may approve, resize, reject, or halt it.
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def evaluate(self, intent: OrderIntent, context: RiskContext) -> RiskEvaluation:
        reasons: list[str] = []
        if context.kill_switch:
            return self._terminal(intent, context, RiskAction.HALT, "kill switch active")
        if context.stale_market_data:
            return self._terminal(intent, context, RiskAction.HALT, "market data is stale")
        if context.equity < self.limits.min_equity:
            return self._terminal(intent, context, RiskAction.HALT, "equity below minimum")
        if context.drawdown_fraction >= self.limits.max_drawdown_fraction:
            return self._terminal(intent, context, RiskAction.HALT, "drawdown limit breached")
        if context.daily_loss_fraction >= self.limits.max_daily_loss_fraction:
            return self._terminal(intent, context, RiskAction.HALT, "daily loss limit breached")
        if intent.symbol not in context.mark_prices:
            return self._terminal(intent, context, RiskAction.REJECT, "missing mark price")

        positions = _position_map(context)
        current = positions.get(intent.symbol)
        current_qty = ZERO if current is None else current.quantity
        projected_qty = current_qty + _signed_quantity(intent)
        if not self.limits.allow_short and projected_qty < ZERO:
            return self._terminal(intent, context, RiskAction.REJECT, "shorting disabled")

        quantity = intent.quantity
        max_position_notional = context.equity * self.limits.max_position_fraction
        cap = self._position_quantity_cap(intent, context, max_position_notional)
        if cap < quantity:
            quantity = cap
            reasons.append("resized for max position limit")
        if quantity <= ZERO:
            return self._terminal(
                intent, context, RiskAction.REJECT, "position limit leaves no capacity"
            )

        quantity = self._cap_by_portfolio_constraints(intent, context, quantity, reasons)
        if quantity <= ZERO:
            return self._terminal(
                intent, context, RiskAction.REJECT, "portfolio limits leave no capacity"
            )

        gross, net = _projected_exposures(intent, context, quantity)
        heat = _projected_heat(intent, context, quantity)
        action = RiskAction.RESIZE if quantity < intent.quantity else RiskAction.APPROVE
        if not reasons:
            reasons.append("within configured risk limits")
        return RiskEvaluation(
            action=action,
            approved_quantity=quantity,
            reasons=tuple(reasons),
            order_intent=intent,
            projected_gross_leverage=gross / context.equity,
            projected_net_leverage=abs(net) / context.equity,
            projected_heat_fraction=heat / context.equity,
        )

    def _position_quantity_cap(
        self,
        intent: OrderIntent,
        context: RiskContext,
        max_position_notional: Decimal,
    ) -> Decimal:
        price = context.mark_prices[intent.symbol]
        current = _position_map(context).get(intent.symbol)
        current_qty = ZERO if current is None else current.quantity
        signed = _signed_quantity(intent)
        projected = current_qty + signed
        if abs(projected) <= abs(current_qty):
            return intent.quantity
        if abs(projected * price) <= max_position_notional:
            return intent.quantity
        direction = ONE if intent.side is Side.BUY else -ONE
        target = direction * max_position_notional / price
        cap = (target - current_qty) / direction
        return max(ZERO, cap)

    def _cap_by_portfolio_constraints(
        self,
        intent: OrderIntent,
        context: RiskContext,
        quantity: Decimal,
        reasons: list[str],
    ) -> Decimal:
        if self._constraints_pass(intent, context, quantity):
            return quantity
        lower = ZERO
        upper = quantity
        for _ in range(80):
            midpoint = (lower + upper) / Decimal("2")
            if self._constraints_pass(intent, context, midpoint):
                lower = midpoint
            else:
                upper = midpoint
        result = lower
        if result < quantity * Decimal("0.999999"):
            reasons.append("resized for portfolio-level limits")
        return result.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)

    def _constraints_pass(
        self,
        intent: OrderIntent,
        context: RiskContext,
        quantity: Decimal,
    ) -> bool:
        if quantity <= ZERO:
            return True
        gross, net = _projected_exposures(intent, context, quantity)
        current_gross = context.gross_exposure / context.equity
        projected_gross = gross / context.equity
        if not self._within_or_reducing(
            projected_gross, current_gross, self.limits.max_gross_leverage
        ):
            return False
        current_net = abs(context.net_exposure) / context.equity
        projected_net = abs(net) / context.equity
        if not self._within_or_reducing(
            projected_net, current_net, self.limits.max_net_leverage
        ):
            return False
        heat = _projected_heat(intent, context, quantity)
        current_heat = context.portfolio_heat / context.equity
        projected_heat = heat / context.equity
        if not self._within_or_reducing(
            projected_heat, current_heat, self.limits.max_portfolio_heat
        ):
            return False
        group = _group_exposure(intent, context, quantity) / context.equity
        current_group = self._current_group_exposure(intent, context) / context.equity
        if not self._within_or_reducing(
            group, current_group, self.limits.max_group_fraction
        ):
            return False
        correlated = _correlated_exposure(
            intent,
            context,
            quantity,
            self.limits.correlation_threshold,
        ) / context.equity
        current_correlated = self._current_correlated_exposure(intent, context) / context.equity
        return self._within_or_reducing(
            correlated, current_correlated, self.limits.max_correlated_fraction
        )

    @staticmethod
    def _within_or_reducing(
        projected: Decimal, current: Decimal, limit: Decimal
    ) -> bool:
        return projected <= limit or (current > limit and projected <= current)

    def _current_group_exposure(
        self, intent: OrderIntent, context: RiskContext
    ) -> Decimal:
        group = context.symbol_groups.get(intent.symbol)
        if group is None:
            current = _position_map(context).get(intent.symbol)
            group = None if current is None else current.group
        if group is None:
            return ZERO
        return sum(
            (
                position.absolute_notional
                for position in context.positions
                if context.symbol_groups.get(position.symbol, position.group) == group
            ),
            ZERO,
        )

    def _current_correlated_exposure(
        self, intent: OrderIntent, context: RiskContext
    ) -> Decimal:
        total = ZERO
        for position in context.positions:
            if position.symbol == intent.symbol:
                total += position.absolute_notional
                continue
            correlation = context.correlations.get((intent.symbol, position.symbol))
            if correlation is None:
                correlation = context.correlations.get((position.symbol, intent.symbol))
            if correlation is not None and abs(correlation) >= self.limits.correlation_threshold:
                total += position.absolute_notional
        return total

    def _terminal(
        self,
        intent: OrderIntent,
        context: RiskContext,
        action: RiskAction,
        reason: str,
    ) -> RiskEvaluation:
        gross = context.gross_exposure / context.equity
        net = abs(context.net_exposure) / context.equity
        heat = context.portfolio_heat / context.equity
        return RiskEvaluation(action, ZERO, (reason,), intent, gross, net, heat)
