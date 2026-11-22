"""Arbitrage *detectors* (SIMULATION ONLY): cross-exchange spot gaps and triangular cycles.

These functions return :class:`ArbOpportunity` records, never orders. Nothing here can
trade. A gap that looks positive on screen usually disappears once you pay the taker fee on
both legs, the withdrawal/transfer fee to move coins between venues, and the slippage you
suffer while the transfer or the second leg is in flight. Every opportunity reports both the
apparent gap and the net edge after those costs, so the reader can see how many survive.

Halal note: both patterns are spot-only. Cross-exchange means buying a coin you then own on
venue A and selling that owned coin on venue B (no borrowing, no shorting on B). Triangular
means converting owned balances through three spot pairs. Spot currency exchange must settle
hand to hand; the latency haircut is where real-world delay is priced in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise, permutations

import numpy as np

from quant_os_core.strategies.bots.common import ONE, ZERO, require_fraction, require_positive

SIMULATION_ONLY = True
BPS = Decimal(10_000)


@dataclass(frozen=True, slots=True)
class Quote:
    """Top-of-book for ``BASE/QUOTE`` on one venue."""

    venue: str
    pair: str
    bid: Decimal
    ask: Decimal

    def __post_init__(self) -> None:
        require_positive(self.bid, "bid")
        if self.ask < self.bid:
            raise ValueError("ask cannot be below bid (crossed book)")
        if self.pair.count("/") != 1:
            raise ValueError("pair must look like BASE/QUOTE")

    @property
    def base(self) -> str:
        return self.pair.split("/")[0]

    @property
    def quote(self) -> str:
        return self.pair.split("/")[1]


@dataclass(frozen=True, slots=True)
class ArbCosts:
    """Per-leg taker fee, latency slippage and the fixed transfer fee (in base units)."""

    taker_fee: Decimal = Decimal("0.001")
    slippage_bps: Decimal = Decimal(5)
    withdrawal_fee: Decimal = ZERO

    def __post_init__(self) -> None:
        require_fraction(self.taker_fee, "taker_fee")
        if self.slippage_bps < ZERO or self.slippage_bps >= BPS:
            raise ValueError("slippage_bps must be in [0, 10000)")
        if self.withdrawal_fee < ZERO:
            raise ValueError("withdrawal_fee cannot be negative")

    @property
    def leg_haircut(self) -> Decimal:
        return (ONE - self.taker_fee) * (ONE - self.slippage_bps / BPS)


@dataclass(frozen=True, slots=True)
class ArbOpportunity:
    """A detected gap. ``net_edge`` is in the start currency after every modelled cost."""

    kind: str
    route: tuple[str, ...]
    start_amount: Decimal
    end_amount: Decimal
    apparent_edge_bps: Decimal
    simulation_only: bool = True

    @property
    def net_edge(self) -> Decimal:
        return self.end_amount - self.start_amount

    @property
    def net_edge_bps(self) -> Decimal:
        return self.net_edge / self.start_amount * BPS

    @property
    def profitable(self) -> bool:
        return self.net_edge > ZERO


def detect_cross_exchange(
    quotes: Sequence[Quote], size: Decimal, costs: ArbCosts
) -> list[ArbOpportunity]:
    """Every buy-on-A / sell-on-B pair for one asset with an apparent gap (``bid_B > ask_A``).

    ``size`` is the base quantity bought on A. The withdrawal fee is paid in base units when
    moving the coins to B, and both legs pay the taker fee and the latency slippage.
    """

    require_positive(size, "size")
    if len({q.pair for q in quotes}) > 1:
        raise ValueError("cross-exchange quotes must all be for the same pair")
    found: list[ArbOpportunity] = []
    slip = costs.slippage_bps / BPS
    for buy in quotes:
        for sell in quotes:
            if buy.venue == sell.venue or sell.bid <= buy.ask:
                continue
            cost = size * buy.ask * (ONE + slip) * (ONE + costs.taker_fee)
            delivered = size - costs.withdrawal_fee
            proceeds = (
                delivered * sell.bid * (ONE - slip) * (ONE - costs.taker_fee)
                if delivered > ZERO
                else ZERO
            )
            found.append(
                ArbOpportunity(
                    "cross_exchange",
                    (f"buy@{buy.venue}", f"sell@{sell.venue}"),
                    cost,
                    proceeds,
                    (sell.bid - buy.ask) / buy.ask * BPS,
                )
            )
    return found


def _convert(
    amount: Decimal, source: str, target: str, book: Mapping[str, Quote], haircut: Decimal
) -> tuple[Decimal, Decimal] | None:
    """Spend ``amount`` of ``source`` for ``target``; returns (net, frictionless) amounts."""

    direct = book.get(f"{target}/{source}")  # buy target with source at the ask
    if direct is not None:
        gross = amount / direct.ask
        return gross * haircut, gross
    inverse = book.get(f"{source}/{target}")  # sell source for target at the bid
    if inverse is not None:
        gross = amount * inverse.bid
        return gross * haircut, gross
    return None


def detect_triangular(
    quotes: Sequence[Quote], start_asset: str, start_amount: Decimal, costs: ArbCosts
) -> list[ArbOpportunity]:
    """Both directions of every 3-asset cycle from ``start_asset`` through one venue's book.

    Apparent edge is the frictionless cycle at quoted bid/ask; net applies the taker fee and
    the slippage on each of the three legs (no transfer: one venue).
    """

    require_positive(start_amount, "start_amount")
    book = {q.pair: q for q in quotes}
    assets = sorted({a for q in quotes for a in (q.base, q.quote)} - {start_asset})
    found: list[ArbOpportunity] = []
    for mid_a, mid_b in permutations(assets, 2):
        path = (start_asset, mid_a, mid_b, start_asset)
        net, gross = start_amount, start_amount
        ok = True
        for src, dst in pairwise(path):
            step_net = _convert(net, src, dst, book, costs.leg_haircut)
            step_gross = _convert(gross, src, dst, book, ONE)
            if step_net is None or step_gross is None:
                ok = False
                break
            net, gross = step_net[0], step_gross[1]
        if not ok:
            continue
        found.append(
            ArbOpportunity(
                "triangular",
                path,
                start_amount,
                net,
                (gross - start_amount) / start_amount * BPS,
            )
        )
    return found


@dataclass(frozen=True, slots=True)
class GapSurvival:
    apparent: int
    survived: int

    @property
    def vanished_share(self) -> float:
        return 1.0 if self.apparent == 0 else 1.0 - self.survived / self.apparent


def survival(opportunities: Sequence[ArbOpportunity]) -> GapSurvival:
    """How many apparent gaps (> 0 bps before costs) still pay after costs."""

    apparent = [o for o in opportunities if o.apparent_edge_bps > ZERO]
    return GapSurvival(len(apparent), sum(1 for o in apparent if o.profitable))


def synthetic_cross_venue_quotes(
    snapshots: int = 500,
    *,
    venues: Sequence[str] = ("venue_a", "venue_b", "venue_c"),
    mid: float = 60_000.0,
    venue_noise_bps: float = 6.0,
    half_spread_bps: float = 1.5,
    seed: int = 7,
) -> list[list[Quote]]:
    """Deterministic SAMPLE data: BTC/USDT books on several venues with small price noise.

    Not market data. Venue mids differ by a few bps (``venue_noise_bps``), which is roughly
    what liquid venues show, so apparent gaps appear often but rarely pay after costs.
    """

    rng = np.random.default_rng(seed)
    out: list[list[Quote]] = []
    price = mid
    for _ in range(snapshots):
        price *= float(np.exp(rng.normal(0.0, 0.001)))
        books = []
        for venue in venues:
            venue_mid = price * (1.0 + rng.normal(0.0, venue_noise_bps) / 1e4)
            half = venue_mid * half_spread_bps / 1e4
            books.append(
                Quote(
                    venue,
                    "BTC/USDT",
                    Decimal(f"{venue_mid - half:.2f}"),
                    Decimal(f"{venue_mid + half:.2f}"),
                )
            )
        out.append(books)
    return out


def synthetic_triangle_quotes(
    snapshots: int = 500, *, noise_bps: float = 4.0, half_spread_bps: float = 2.0, seed: int = 11
) -> list[list[Quote]]:
    """Deterministic SAMPLE books for BTC/USDT, ETH/BTC and ETH/USDT on one venue."""

    rng = np.random.default_rng(seed)
    out: list[list[Quote]] = []
    btc, eth = 60_000.0, 3_000.0
    for _ in range(snapshots):
        btc *= float(np.exp(rng.normal(0.0, 0.001)))
        eth *= float(np.exp(rng.normal(0.0, 0.0012)))
        mids = {
            "BTC/USDT": btc,
            "ETH/BTC": eth / btc * (1.0 + rng.normal(0.0, noise_bps) / 1e4),
            "ETH/USDT": eth,
        }
        books = []
        for pair, m in mids.items():
            half = m * half_spread_bps / 1e4
            books.append(
                Quote("venue_a", pair, Decimal(f"{m - half:.8f}"), Decimal(f"{m + half:.8f}"))
            )
        out.append(books)
    return out
