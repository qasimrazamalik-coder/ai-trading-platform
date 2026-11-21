"""Trade-level simulator that runs the live decision path over historical bars.

Freqtrade only applies Protections in backtests when asked (``--enable-protections``). Here the
same :class:`EntryPipeline` (plan check, Shariah gate, protections, governor, council, risk
engine) and :class:`ExitManager` used live are replayed bar by bar, and every closed trade is
fed back to the protections, so a backtest cannot quietly skip the safety rules.

Single-symbol, cash account, fills at the plan's entry price on the signal bar's close.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal

from quant_os_core.agents.day_trading import flatten_time, settlement_time
from quant_os_core.agents.exits import Bar, ExitManager, InvalidationRule, OpenPosition
from quant_os_core.agents.pipeline import EntryDecision, EntryPipeline
from quant_os_core.agents.trade_plan import TradePlan
from quant_os_core.instruments import Instrument
from quant_os_core.risk_engine import ClosedTrade, RiskContext, RiskPosition
from quant_os_core.shariah import CompanyFinancials, CryptoAsset, ShariahOrderContext

SignalFn = Callable[[Sequence[Bar]], TradePlan | None]


@dataclass(frozen=True, slots=True)
class ExitPolicy:
    rules: tuple[InvalidationRule, ...] = ()
    trail_fraction: Decimal | None = Decimal("0.08")
    max_holding: timedelta | None = timedelta(days=60)
    flatten_before_close: bool = False  # day trading: no position held past the session


@dataclass(slots=True)
class SimulationResult:
    starting_cash: Decimal
    cash: Decimal
    trades: list[ClosedTrade] = field(default_factory=list)
    decisions: list[EntryDecision] = field(default_factory=list)
    equity_curve: list[Decimal] = field(default_factory=list)

    @property
    def win_rate(self) -> Decimal:
        if not self.trades:
            return Decimal("0")
        wins = sum(1 for t in self.trades if t.profit_ratio > 0)
        return Decimal(wins) / Decimal(len(self.trades))

    @property
    def total_return(self) -> Decimal:
        final = self.equity_curve[-1] if self.equity_curve else self.cash
        return final / self.starting_cash - Decimal("1")

    @property
    def blocked_by(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for decision in self.decisions:
            stage = decision.blocked_by
            if stage is not None:
                counts[stage] = counts.get(stage, 0) + 1
        return counts


def simulate(
    bars: Sequence[Bar],
    signal: SignalFn,
    pipeline: EntryPipeline,
    *,
    instrument: Instrument,
    financials: CompanyFinancials | None,
    starting_cash: Decimal,
    exit_policy: ExitPolicy | None = None,
    cost_per_side: Decimal = Decimal("0.001"),
    crypto_asset: CryptoAsset | None = None,
) -> SimulationResult:
    if starting_cash <= 0:
        raise ValueError("starting_cash must be positive")
    policy = exit_policy or ExitPolicy()
    asset_class = instrument.asset_class
    hold_to_settlement = not pipeline.shariah.profile.allow_pre_settlement_resale
    if hold_to_settlement and policy.flatten_before_close:
        flat = flatten_time(asset_class, bars[0].time) if bars else None
        if flat is not None:
            raise ValueError("this profile forbids selling before settlement, so no day trades")
    exits = ExitManager()
    result = SimulationResult(starting_cash, starting_cash)
    peak = starting_cash
    day_start = starting_cash
    position: OpenPosition | None = None
    closes: list[Decimal] = []
    symbol = instrument.instrument_id.symbol

    for index, bar in enumerate(bars):
        closes.append(bar.close)
        if index > 0 and bar.time.date() != bars[index - 1].time.date():
            day_start = result.equity_curve[-1] if result.equity_curve else result.cash

        if position is not None:
            event = exits.on_bar(position, bar, tuple(closes))
            if event is not None:
                proceeds = position.quantity * event.price
                result.cash += proceeds - proceeds * cost_per_side
                result.trades.append(event.trade)
                position = None

        if position is None:
            plan = signal(bars[: index + 1])
            if plan is not None and plan.symbol == symbol:
                equity = result.cash
                peak = max(peak, equity)
                decision = pipeline.decide(
                    plan,
                    instrument=instrument,
                    financials=financials,
                    crypto_asset=crypto_asset,
                    account=ShariahOrderContext(owned_quantity=Decimal("0"), free_cash=result.cash),
                    risk_context=RiskContext(
                        equity=equity,
                        peak_equity=peak,
                        start_of_day_equity=max(day_start, Decimal("0.0001")),
                        positions=(),
                        mark_prices={symbol: plan.entry_price},
                        stop_prices={symbol: plan.stop_loss},
                    ),
                    closed_trades=result.trades,
                    open_symbols=set(),
                    now=bar.time,
                )
                result.decisions.append(decision)
                if decision.approved and decision.quantity > 0:
                    cost = decision.quantity * plan.entry_price
                    result.cash -= cost + cost * cost_per_side
                    position = OpenPosition(
                        plan,
                        decision.quantity,
                        bar.time,
                        rules=policy.rules,
                        trail_fraction=policy.trail_fraction,
                        max_holding=policy.max_holding,
                        flatten_at=(
                            flatten_time(asset_class, bar.time)
                            if policy.flatten_before_close
                            else None
                        ),
                        earliest_exit=(
                            settlement_time(asset_class, bar.time) if hold_to_settlement else None
                        ),
                    )

        marked = result.cash + (position.quantity * bar.close if position else Decimal("0"))
        result.equity_curve.append(marked)
        peak = max(peak, marked)
    return result


def position_snapshot(position: OpenPosition, mark: Decimal) -> RiskPosition:
    """Convert an open position for the risk engine (used by live loops)."""

    return RiskPosition(position.plan.symbol, position.quantity, mark, position.stop)
