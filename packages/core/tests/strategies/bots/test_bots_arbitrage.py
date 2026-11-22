from decimal import Decimal

import pytest

from quant_os_core.strategies.bots import (
    ArbCosts,
    Quote,
    detect_cross_exchange,
    detect_triangular,
    survival,
)
from quant_os_core.strategies.bots.arbitrage import (
    SIMULATION_ONLY,
    GapSurvival,
    synthetic_cross_venue_quotes,
    synthetic_triangle_quotes,
)

D = Decimal
FREE = ArbCosts(D(0), D(0), D(0))


def test_quote_validation() -> None:
    q = Quote("a", "ETH/BTC", D("0.05"), D("0.0501"))
    assert q.base == "ETH" and q.quote == "BTC"
    with pytest.raises(ValueError, match="crossed"):
        Quote("a", "BTC/USDT", D(101), D(100))
    with pytest.raises(ValueError, match="BASE/QUOTE"):
        Quote("a", "BTCUSDT", D(100), D(101))


def test_costs_validation() -> None:
    assert ArbCosts().leg_haircut == D("0.999") * (1 - D(5) / D(10_000))
    with pytest.raises(ValueError, match="slippage"):
        ArbCosts(slippage_bps=D(-1))
    with pytest.raises(ValueError, match="withdrawal"):
        ArbCosts(withdrawal_fee=D(-1))


def test_cross_exchange_gap_without_costs_equals_the_gap() -> None:
    quotes = [
        Quote("a", "BTC/USDT", D(99), D(100)),
        Quote("b", "BTC/USDT", D(101), D(102)),
    ]
    found = detect_cross_exchange(quotes, D(2), FREE)
    assert SIMULATION_ONLY and len(found) == 1
    opp = found[0]
    assert opp.route == ("buy@a", "sell@b") and opp.simulation_only
    assert opp.net_edge == D(2) and opp.profitable
    assert opp.apparent_edge_bps == D(100)
    assert opp.net_edge_bps == D(100)


def test_cross_exchange_costs_eat_a_small_gap() -> None:
    quotes = [
        Quote("a", "BTC/USDT", D("59990"), D("60000")),
        Quote("b", "BTC/USDT", D("60030"), D("60040")),  # 5 bps apparent gap
    ]
    costs = ArbCosts(D("0.001"), D(5), D("0.0002"))
    (opp,) = detect_cross_exchange(quotes, D("0.5"), costs)
    assert opp.apparent_edge_bps == pytest.approx(D(5))
    assert not opp.profitable and opp.net_edge_bps < D(-20)


def test_cross_exchange_edge_cases() -> None:
    same = [Quote("a", "BTC/USDT", D(99), D(100)), Quote("a", "BTC/USDT", D(101), D(102))]
    assert detect_cross_exchange(same, D(1), FREE) == []
    no_gap = [Quote("a", "BTC/USDT", D(99), D(100)), Quote("b", "BTC/USDT", D(99), D(100))]
    assert detect_cross_exchange(no_gap, D(1), FREE) == []
    eaten = [Quote("a", "BTC/USDT", D(99), D(100)), Quote("b", "BTC/USDT", D(101), D(102))]
    (opp,) = detect_cross_exchange(eaten, D(1), ArbCosts(D(0), D(0), D(2)))
    assert opp.end_amount == 0 and not opp.profitable
    with pytest.raises(ValueError, match="same pair"):
        detect_cross_exchange(
            [Quote("a", "BTC/USDT", D(1), D(2)), Quote("b", "ETH/USDT", D(1), D(2))],
            D(1),
            FREE,
        )


def _triangle(eth_btc_bid: str, eth_btc_ask: str) -> list[Quote]:
    return [
        Quote("x", "BTC/USDT", D(60_000), D(60_000)),
        Quote("x", "ETH/BTC", D(eth_btc_bid), D(eth_btc_ask)),
        Quote("x", "ETH/USDT", D(3_000), D(3_000)),
    ]


def test_triangular_detects_mispriced_cycle_both_directions() -> None:
    found = detect_triangular(_triangle("0.049", "0.049"), "USDT", D(10_000), FREE)
    routes = {o.route: o for o in found}
    assert set(routes) == {("USDT", "BTC", "ETH", "USDT"), ("USDT", "ETH", "BTC", "USDT")}
    cheap_eth = routes[("USDT", "BTC", "ETH", "USDT")]
    # 10000 USDT -> 1/6 BTC -> (1/6)/0.049 ETH -> x 3000 USDT
    expected = D(10_000) / D(60_000) / D("0.049") * D(3_000)
    assert cheap_eth.end_amount == pytest.approx(expected)
    assert cheap_eth.profitable
    assert not routes[("USDT", "ETH", "BTC", "USDT")].profitable


def test_triangular_fees_erase_a_small_mispricing() -> None:
    found = detect_triangular(_triangle("0.04999", "0.04999"), "USDT", D(10_000), ArbCosts())
    best = max(found, key=lambda o: o.apparent_edge_bps)
    assert best.apparent_edge_bps > 0 and not best.profitable


def test_triangular_skips_missing_pairs() -> None:
    quotes = [
        Quote("x", "BTC/USDT", D(1), D(1)),
        Quote("x", "ETH/USDT", D(1), D(1)),
        Quote("x", "SOL/USDT", D(1), D(1)),
    ]
    assert detect_triangular(quotes, "USDT", D(1), FREE) == []


def test_most_sample_gaps_vanish_after_costs() -> None:
    cross_costs = ArbCosts(D("0.001"), D(5), D("0.0002"))
    cross = [
        o
        for books in synthetic_cross_venue_quotes(200)
        for o in detect_cross_exchange(books, D("0.5"), cross_costs)
    ]
    s = survival(cross)
    assert s.apparent > 100
    assert s.vanished_share >= 0.9
    free = survival(
        [
            o
            for books in synthetic_cross_venue_quotes(200)
            for o in detect_cross_exchange(books, D("0.5"), FREE)
        ]
    )
    assert free.survived == free.apparent  # without costs every apparent gap "pays"
    tri = [
        o
        for books in synthetic_triangle_quotes(200)
        for o in detect_triangular(books, "USDT", D(10_000), ArbCosts())
    ]
    assert survival(tri).vanished_share >= 0.9


def test_gap_survival_empty() -> None:
    assert GapSurvival(0, 0).vanished_share == 1.0
    assert survival([]) == GapSurvival(0, 0)
