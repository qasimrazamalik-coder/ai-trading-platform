"""Event-driven multi-asset backtesting orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from quant_os_core.backtesting.broker import SimulatedBroker
from quant_os_core.backtesting.costs import FinancingModel
from quant_os_core.backtesting.models import BacktestResult, EquityPoint, OrderRequest
from quant_os_core.backtesting.portfolio import Portfolio
from quant_os_core.market_data.models import Candle


@dataclass(frozen=True, slots=True)
class StrategyContext:
    event_index: int
    instrument_bar_index: int
    cash: Decimal
    equity: Decimal
    gross_exposure: Decimal
    net_exposure: Decimal
    current_position_quantity: Decimal = Decimal("0")
    pending_order_quantity: Decimal = Decimal("0")

    @property
    def bar_index(self) -> int:
        """Backward-compatible alias for the current instrument-local bar index."""
        return self.instrument_bar_index


class BacktestStrategy(Protocol):
    def on_bar(self, candle: Candle, context: StrategyContext) -> tuple[OrderRequest, ...]: ...


class EventDrivenBacktester:
    def __init__(
        self,
        *,
        initial_cash: Decimal,
        broker: SimulatedBroker | None = None,
        financing_model: FinancingModel | None = None,
    ) -> None:
        self.initial_cash = initial_cash
        self.broker = broker or SimulatedBroker()
        self.financing_model = financing_model

    def run(self, candles: tuple[Candle, ...], strategy: BacktestStrategy) -> BacktestResult:
        if not candles:
            raise ValueError("candles cannot be empty")
        self._validate_sequence(candles)
        self.broker.reset()
        portfolio = Portfolio(self.initial_cash)
        equity_curve: list[EquityPoint] = []
        latest_bar_index: dict[str, int] = {}

        for event_index, candle in enumerate(candles):
            key = candle.instrument_id.key
            local_index = latest_bar_index.get(key, -1) + 1
            latest_bar_index[key] = local_index

            for fill in self.broker.process_bar(candle, bar_index=local_index):
                portfolio.apply_fill(fill)

            portfolio.mark(candle.instrument_id, candle.close)
            self._apply_financing(portfolio, key)
            equity_curve.append(
                EquityPoint(
                    timestamp=candle.close_time,
                    cash=portfolio.cash,
                    equity=portfolio.equity,
                    gross_exposure=portfolio.gross_exposure,
                    net_exposure=portfolio.net_exposure,
                )
            )

            context = StrategyContext(
                event_index=event_index,
                instrument_bar_index=local_index,
                cash=portfolio.cash,
                equity=portfolio.equity,
                gross_exposure=portfolio.gross_exposure,
                net_exposure=portfolio.net_exposure,
                current_position_quantity=portfolio.position_for(candle.instrument_id).quantity,
                pending_order_quantity=self.broker.pending_signed_quantity(candle.instrument_id),
            )
            for request in strategy.on_bar(candle, context):
                target_index = latest_bar_index.get(request.instrument_id.key, -1)
                self.broker.submit(
                    request,
                    submitted_at=candle.close_time,
                    bar_index=target_index,
                )

        self.broker.cancel_open_orders()
        return BacktestResult(
            initial_cash=self.initial_cash,
            final_equity=portfolio.equity,
            equity_curve=tuple(equity_curve),
            fills=tuple(self.broker.fills),
            submitted_orders=len(self.broker.orders),
            rejected_orders=self.broker.rejected_orders,
            cancelled_orders=self.broker.cancelled_orders,
        )

    def _apply_financing(self, portfolio: Portfolio, instrument_key: str) -> None:
        if self.financing_model is None:
            return
        position = portfolio.positions.get(instrument_key)
        if position is None or position.quantity == 0:
            return
        mark = portfolio.marks[instrument_key]
        charge = self.financing_model.charge(signed_notional=position.quantity * mark)
        portfolio.apply_financing(charge)

    @staticmethod
    def _validate_sequence(candles: tuple[Candle, ...]) -> None:
        previous_close = candles[0].close_time
        latest_by_instrument: dict[str, Candle] = {candles[0].instrument_id.key: candles[0]}
        timeframe_by_instrument = {candles[0].instrument_id.key: candles[0].timeframe}

        for candle in candles[1:]:
            if candle.close_time < previous_close:
                raise ValueError("candles must be non-decreasing by close_time")
            previous_close = candle.close_time
            key = candle.instrument_id.key
            previous = latest_by_instrument.get(key)
            if previous is not None and candle.timestamp <= previous.timestamp:
                raise ValueError("each instrument must be strictly increasing by timestamp")
            timeframe = timeframe_by_instrument.get(key)
            if timeframe is not None and candle.timeframe != timeframe:
                raise ValueError("each instrument must use one timeframe per backtest")
            latest_by_instrument[key] = candle
            timeframe_by_instrument[key] = candle.timeframe
