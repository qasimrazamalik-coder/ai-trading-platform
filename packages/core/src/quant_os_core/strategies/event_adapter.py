"""Adapters for running target-exposure strategies in the event-driven backtester."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

from quant_os_core.backtesting.engine import StrategyContext
from quant_os_core.backtesting.models import OrderRequest, OrderSide, OrderType
from quant_os_core.market_data.models import Candle

SignalFunction = Callable[[tuple[Decimal, ...]], float]


class SingleAssetTargetAdapter:
    """Translate a close-history target signal into market orders.

    The target signal must be inside [-1, 1]. ``max_units`` converts normalized exposure
    into instrument units. Actual filled position from ``StrategyContext`` is used, so partial
    fills do not silently desynchronize the adapter from portfolio accounting.
    """

    def __init__(
        self,
        *,
        signal_function: SignalFunction,
        max_units: Decimal,
        minimum_history: int = 2,
        rebalance_threshold_units: Decimal = Decimal("0"),
        tag: str = "strategy_factory",
    ) -> None:
        if max_units <= 0:
            raise ValueError("max_units must be > 0")
        if minimum_history <= 0:
            raise ValueError("minimum_history must be > 0")
        if rebalance_threshold_units < 0:
            raise ValueError("rebalance_threshold_units cannot be negative")
        self.signal_function = signal_function
        self.max_units = max_units
        self.minimum_history = minimum_history
        self.rebalance_threshold_units = rebalance_threshold_units
        self.tag = tag
        self._instrument_key: str | None = None
        self._closes: list[Decimal] = []

    def reset(self) -> None:
        self._instrument_key = None
        self._closes.clear()

    def on_bar(self, candle: Candle, context: StrategyContext) -> tuple[OrderRequest, ...]:
        key = candle.instrument_id.key
        if self._instrument_key is None:
            self._instrument_key = key
        elif key != self._instrument_key:
            raise ValueError("SingleAssetTargetAdapter cannot consume multiple instruments")
        self._closes.append(candle.close)
        if len(self._closes) < self.minimum_history:
            return ()
        signal = float(self.signal_function(tuple(self._closes)))
        if not -1.0 <= signal <= 1.0:
            raise ValueError("signal_function must return a value inside [-1, 1]")
        target = self.max_units * Decimal(str(signal))
        effective_position = context.current_position_quantity + context.pending_order_quantity
        delta = target - effective_position
        if abs(delta) <= self.rebalance_threshold_units:
            return ()
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        return (
            OrderRequest(
                instrument_id=candle.instrument_id,
                side=side,
                quantity=abs(delta),
                order_type=OrderType.MARKET,
                tag=self.tag,
            ),
        )
