"""Profit levers that raise expected return without leverage, shorting or derivatives.

1. **Edge-over-cost filter** - refuse trades whose expected value does not beat round-trip
   costs by a margin. This removes the trades that made day trading lose in simulation.
2. **Cheaper execution** - maker (resting limit) orders where urgency is low: Alpaca crypto
   tier 1 is 0.15% maker vs 0.25% taker (https://docs.alpaca.markets/docs/crypto-fees).
3. **Risk dial** - conservative / balanced / growth presets for plan sizing. Growth raises
   risk per trade and position size; it never adds leverage or shorting.
4. **Sleeve allocator** - combine uncorrelated halal sleeves (US stocks, PSX, crypto spot,
   gold) by risk parity, then scale towards a volatility target, capped at fully invested.
5. **Deposit planner** - monthly deposits are the largest lever on a small account.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

import numpy as np

from quant_os_core.agents.trade_plan import PlanPolicy, TradePlan
from quant_os_core.instruments import AssetClass
from quant_os_core.portfolio.optimization import risk_parity_weights


@dataclass(frozen=True, slots=True)
class VenueCost:
    """Cost of one side of a trade, as a fraction of notional."""

    maker_fee: Decimal
    taker_fee: Decimal
    half_spread: Decimal
    slippage: Decimal

    def per_side(self, *, maker: bool) -> Decimal:
        fee = self.maker_fee if maker else self.taker_fee
        # A resting limit order earns the half-spread back instead of paying it, but may not
        # fill; this model counts zero spread and zero slippage for makers (fill risk aside).
        crossing = Decimal("0") if maker else self.half_spread + self.slippage
        return fee + crossing


# Defaults: Alpaca stocks are commission-free (regulatory fees ignored as negligible);
# liquid US large caps quote about 1-2 bp half-spread. Crypto tier-1 fees from Alpaca docs.
DEFAULT_COSTS: Mapping[AssetClass, VenueCost] = {
    AssetClass.EQUITY: VenueCost(Decimal("0"), Decimal("0"), Decimal("0.0002"), Decimal("0.0003")),
    AssetClass.ETF: VenueCost(Decimal("0"), Decimal("0"), Decimal("0.0001"), Decimal("0.0002")),
    AssetClass.CRYPTO: VenueCost(
        Decimal("0.0015"), Decimal("0.0025"), Decimal("0.0003"), Decimal("0.0005")
    ),
}


@dataclass(frozen=True, slots=True)
class CostModel:
    costs: Mapping[AssetClass, VenueCost] = field(default_factory=lambda: dict(DEFAULT_COSTS))
    prefer_maker: bool = True  # resting limit entries; exits on stops stay taker
    min_edge_to_cost: Decimal = Decimal("1.5")  # expected edge must be 1.5x round-trip cost

    def __post_init__(self) -> None:
        if self.min_edge_to_cost < 1:
            raise ValueError("min_edge_to_cost below 1 accepts trades that lose to costs")

    def round_trip(self, asset_class: AssetClass) -> Decimal:
        venue = self.costs.get(asset_class)
        if venue is None:
            raise ValueError(f"no cost model for {asset_class}")
        return venue.per_side(maker=self.prefer_maker) + venue.per_side(maker=False)


@dataclass(frozen=True, slots=True)
class EdgeVerdict:
    accepted: bool
    expected_edge: Decimal  # fraction of entry price, before costs
    round_trip_cost: Decimal
    reason: str


def expected_edge(plan: TradePlan) -> Decimal:
    """Expected move per unit as a fraction of entry: p*reward - (1-p)*risk.

    ``confidence`` is read as the probability of reaching the target before the stop, so it
    must be calibrated (the meta-labeller's probability), not a vibe score.
    """

    p = plan.confidence
    return (p * plan.reward_per_unit - (Decimal("1") - p) * plan.risk_per_unit) / plan.entry_price


def check_edge(plan: TradePlan, asset_class: AssetClass, model: CostModel) -> EdgeVerdict:
    edge = expected_edge(plan)
    cost = model.round_trip(asset_class)
    if edge <= 0:
        return EdgeVerdict(False, edge, cost, "negative expected value before costs")
    if edge < cost * model.min_edge_to_cost:
        return EdgeVerdict(
            False,
            edge,
            cost,
            f"edge {edge:.4%} is under {model.min_edge_to_cost}x round-trip cost {cost:.4%}",
        )
    return EdgeVerdict(True, edge, cost, f"edge {edge:.4%} vs cost {cost:.4%}")


class RiskDial(StrEnum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    GROWTH = "growth"


_DIAL: Mapping[RiskDial, tuple[str, str, str]] = {
    # risk per trade, max position fraction, portfolio volatility target
    RiskDial.CONSERVATIVE: ("0.005", "0.15", "0.08"),
    RiskDial.BALANCED: ("0.01", "0.20", "0.12"),
    RiskDial.GROWTH: ("0.02", "0.30", "0.20"),
}


def dial_policy(dial: RiskDial, base: PlanPolicy | None = None) -> PlanPolicy:
    """Plan limits for a dial. Growth doubles risk per trade; drawdowns roughly double too."""

    risk, position, _ = _DIAL[dial]
    b = base or PlanPolicy()
    return PlanPolicy(
        min_confidence=b.min_confidence,
        min_reward_to_risk=b.min_reward_to_risk,
        min_stop_distance=b.min_stop_distance,
        max_stop_distance=b.max_stop_distance,
        risk_per_trade=Decimal(risk),
        max_position_fraction=Decimal(position),
        max_trades_per_day=b.max_trades_per_day,
        quantity_step=b.quantity_step,
    )


def dial_vol_target(dial: RiskDial) -> float:
    return float(_DIAL[dial][2])


@dataclass(frozen=True, slots=True)
class Allocation:
    weights: dict[str, float]  # sleeve -> fraction of equity
    cash: float
    expected_vol: float


def allocate_sleeves(
    names: Sequence[str],
    covariance: np.ndarray,
    *,
    vol_target: float,
    max_sleeve: float = 0.6,
) -> Allocation:
    """Risk-parity weights scaled to ``vol_target``; total exposure is capped at 1.0.

    If the risk-parity mix is less volatile than the target, it is fully invested (no
    leverage, so the target may not be reached); otherwise the remainder sits in cash.
    """

    cov = np.asarray(covariance, dtype=np.float64)
    if len(names) != cov.shape[0] or len(set(names)) != len(names):
        raise ValueError("names must be unique and match the covariance")
    if not 0 < vol_target < 2 or not 0 < max_sleeve <= 1:
        raise ValueError("vol_target in (0, 2) and max_sleeve in (0, 1]")
    w = np.asarray(risk_parity_weights(cov), dtype=np.float64)
    # Cap any single sleeve, redistributing the excess pro rata to the others.
    for _ in range(len(w)):
        over = w > max_sleeve
        if not over.any():
            break
        excess = float((w[over] - max_sleeve).sum())
        w[over] = max_sleeve
        room = ~over
        if not room.any():
            break
        w[room] += excess * w[room] / w[room].sum()
    vol = float(np.sqrt(w @ cov @ w))
    scale = min(1.0, vol_target / vol) if vol > 0 else 0.0
    weights = {n: float(x * scale) for n, x in zip(names, w, strict=True)}
    return Allocation(weights, 1.0 - sum(weights.values()), vol * scale)


@dataclass(frozen=True, slots=True)
class DepositPoint:
    month: int
    deposited: Decimal
    balance: Decimal


def deposit_plan(
    start: Decimal, monthly: Decimal, annual_return: Decimal, months: int
) -> list[DepositPoint]:
    """Deterministic path: grow monthly at the compounded rate, deposit at month end."""

    if start < 0 or monthly < 0 or months < 1 or annual_return <= -1:
        raise ValueError("invalid deposit plan inputs")
    monthly_rate = Decimal(str((1 + float(annual_return)) ** (1 / 12) - 1))
    balance, deposited = start, start
    path = []
    for month in range(1, months + 1):
        balance = balance * (1 + monthly_rate) + monthly
        deposited += monthly
        path.append(DepositPoint(month, deposited, balance.quantize(Decimal("0.01"))))
    return path
