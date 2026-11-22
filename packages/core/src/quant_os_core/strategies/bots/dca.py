"""Spot DCA bot: periodic buys, optional scaled safety buys on dips, take-profit on cost.

Halal shape: spot only, long-only, unlevered, capital capped. Every buy is paid from cash on
hand (``SpotWallet`` refuses anything else) and the open position's cost may never exceed
``max_invested``. The only sale is the take-profit, which sells the coins the bot owns.

Bar-by-bar rules (conservative):

* A periodic buy of ``order_amount`` happens at the close every ``interval`` bars. It is
  time-scheduled, so deciding and filling at the same close is not look-ahead.
* Safety buy ``k`` (1-based) is a limit at ``reference x (1 - step x (1 + s + ... + s^(k-1)))``
  where ``reference`` is the first buy price of the cycle and ``s`` is ``safety_step_scale``.
  It rests from the previous close and fills only if the bar trades through it
  (``low < price``). Its size is ``order_amount x safety_volume_scale^k``.
* The take-profit limit sits at ``average cost x (1 + take_profit)`` (average cost includes
  buy fees). It fills only if ``high > price`` and only in a bar with no buy, so a lot is
  never bought and sold in the same bar. Selling everything ends the cycle.

Simulation only; real orders still go through the Shariah gate, the risk engine, the
edge-over-cost filter and protections.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from quant_os_core.strategies.bots.common import (
    ONE,
    ZERO,
    Bar,
    BotResult,
    SpotWallet,
    require_fraction,
    require_positive,
)


@dataclass(frozen=True, slots=True)
class DcaConfig:
    capital: Decimal
    order_amount: Decimal
    interval: int
    max_invested: Decimal | None = None
    safety_orders: int = 0
    safety_step: Decimal = Decimal("0.05")
    safety_step_scale: Decimal = Decimal(1)
    safety_volume_scale: Decimal = Decimal("1.5")
    take_profit: Decimal | None = None
    fee_rate: Decimal = Decimal("0.001")

    def __post_init__(self) -> None:
        require_positive(self.capital, "capital")
        require_positive(self.order_amount, "order_amount")
        if self.interval < 1:
            raise ValueError("interval must be >= 1 bar")
        if self.max_invested is not None:
            require_positive(self.max_invested, "max_invested")
            if self.max_invested > self.capital:
                raise ValueError("max_invested cannot exceed capital (no leverage)")
        if self.safety_orders < 0:
            raise ValueError("safety_orders cannot be negative")
        require_fraction(self.safety_step, "safety_step", allow_zero=False)
        require_positive(self.safety_step_scale, "safety_step_scale")
        require_positive(self.safety_volume_scale, "safety_volume_scale")
        if self.take_profit is not None:
            require_positive(self.take_profit, "take_profit")
        require_fraction(self.fee_rate, "fee_rate")
        if self.safety_orders and self.safety_trigger_offset(self.safety_orders) >= ONE:
            raise ValueError("deepest safety order would sit at or below zero")

    @property
    def cap(self) -> Decimal:
        return self.max_invested if self.max_invested is not None else self.capital

    def safety_trigger_offset(self, k: int) -> Decimal:
        """Fractional drop below the cycle reference that triggers safety order ``k``."""

        return self.safety_step * sum((self.safety_step_scale**j for j in range(k)), ZERO)

    def safety_amount(self, k: int) -> Decimal:
        return self.order_amount * self.safety_volume_scale**k


class DcaBot:
    """Stateful DCA engine; ``step`` is called once per bar in order."""

    def __init__(self, config: DcaConfig, wallet: SpotWallet | None = None) -> None:
        self.config = config
        self.wallet = wallet if wallet is not None else SpotWallet(config.capital)
        self.quantity = ZERO
        self.cost_basis = ZERO
        self.reference: Decimal | None = None
        self.safety_filled = 0
        self.cycle_start = -1
        self.cycles_closed = 0

    @property
    def average_cost(self) -> Decimal | None:
        return self.cost_basis / self.quantity if self.quantity > ZERO else None

    def _buy(self, index: int, time: str, price: Decimal, amount: Decimal, reason: str) -> bool:
        budget = min(amount, self.config.cap - self.cost_basis)
        qty = self.wallet.max_buy_quantity(price, self.config.fee_rate, budget)
        if qty <= ZERO:
            return False
        fill = self.wallet.buy(time, price, qty, self.config.fee_rate, reason)
        if self.quantity == ZERO:
            self.reference = price
            self.cycle_start = index
            self.safety_filled = 0
        self.quantity += qty
        self.cost_basis += fill.notional + fill.fee
        return True

    def step(self, index: int, bar: Bar) -> None:
        cfg = self.config
        bought = False
        tp_price = None
        avg = self.average_cost
        if cfg.take_profit is not None and avg is not None:
            tp_price = avg * (ONE + cfg.take_profit)
        # safety buys rest from the previous close of an open cycle
        if self.reference is not None and self.quantity > ZERO and self.cycle_start < index:
            while self.safety_filled < cfg.safety_orders:
                k = self.safety_filled + 1
                price = self.reference * (ONE - cfg.safety_trigger_offset(k))
                if not bar.low < price:
                    break
                if not self._buy(index, bar.time, price, cfg.safety_amount(k), f"safety_{k}"):
                    break
                self.safety_filled = k
                bought = True
        if tp_price is not None and not bought and bar.high > tp_price:
            self.wallet.sell(bar.time, tp_price, self.quantity, cfg.fee_rate, "take_profit")
            self.quantity = ZERO
            self.cost_basis = ZERO
            self.reference = None
            self.safety_filled = 0
            self.cycles_closed += 1
        if index % cfg.interval == 0:
            self._buy(index, bar.time, bar.close, cfg.order_amount, "periodic")


def run_dca(bars: Sequence[Bar], config: DcaConfig, *, name: str = "dca") -> BotResult:
    if not bars:
        raise ValueError("bars cannot be empty")
    bot = DcaBot(config)
    equity: list[Decimal] = []
    for index, bar in enumerate(bars):
        bot.step(index, bar)
        equity.append(bot.wallet.equity(bar.close))
    return BotResult(
        name,
        config.capital,
        tuple(b.time for b in bars),
        tuple(equity),
        tuple(bot.wallet.fills),
        bot.wallet.cash,
        bot.wallet.quantity,
        (f"cycles_closed={bot.cycles_closed}", f"open_cost={bot.cost_basis}"),
    )
