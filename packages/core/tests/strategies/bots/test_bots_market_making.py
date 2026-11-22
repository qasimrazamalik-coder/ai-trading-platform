from decimal import Decimal

import numpy as np
import pytest

from quant_os_core.strategies.bots import (
    Bar,
    MarketMakerConfig,
    Side,
    bars_from_arrays,
    run_market_maker,
)
from quant_os_core.strategies.bots.market_making import SIMULATION_ONLY, make_quotes

D = Decimal


def cfg(**overrides: object) -> MarketMakerConfig:
    base: dict[str, object] = {
        "capital": D(1000),
        "order_size": D(1),
        "max_inventory": D(4),
        "half_spread": D("0.01"),
        "skew": D("0.02"),
        "target_ratio": D("0.5"),
        "fee_rate": D(0),
    }
    base.update(overrides)
    return MarketMakerConfig(**base)  # type: ignore[arg-type]


def test_quotes_skew_with_inventory() -> None:
    flat = make_quotes(cfg(), D(100), D(2), D(1000))
    assert flat.reservation == D(100)
    assert flat.bid == D(99) and flat.ask == D(101)
    heavy = make_quotes(cfg(), D(100), D(3), D(1000))
    assert heavy.reservation < D(100)
    light = make_quotes(cfg(), D(100), D(1), D(1000))
    assert light.reservation > D(100)


def test_constraints_remove_quote_sides() -> None:
    empty = make_quotes(cfg(), D(100), D(0), D(1000))
    assert empty.ask is None and empty.bid is not None  # nothing owned, nothing to sell
    full = make_quotes(cfg(), D(100), D(4), D(1000))
    assert full.bid is None and full.ask is not None
    broke = make_quotes(cfg(), D(100), D(1), D(50))
    assert broke.bid is None


def test_round_trip_fills_need_trade_through() -> None:
    bars = [
        Bar("t0", D(100), D(100), D(100)),
        Bar("t1", D(100), D(98), D(99)),  # bid ~99.01 traded through -> buy 1
        Bar("t2", D(102), D(99), D(101)),  # ask from prev close 99 with 1 owned
        Bar("t3", D("101.5"), D("101.1"), D("101.3")),  # inside the quotes: nothing
    ]
    res = run_market_maker(bars, cfg())
    assert SIMULATION_ONLY and "SIMULATION ONLY" in res.notes
    assert [f.side for f in res.fills] == [Side.BUY, Side.SELL]
    assert res.final_quantity == 0 and res.final_cash > D(1000)


def test_inventory_stays_in_bounds_on_random_walk() -> None:
    rng = np.random.default_rng(9)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    wig = np.abs(rng.normal(0, 0.01, 500)) * close
    bars = bars_from_arrays(
        [f"d{i:04d}" for i in range(500)], list(close + wig), list(close - wig), list(close)
    )
    res = run_market_maker(bars, cfg(fee_rate=D("0.001")))
    held = D(0)
    for fill in res.fills:
        held += fill.quantity if fill.side is Side.BUY else -fill.quantity
        assert D(0) <= held <= D(4)
    assert res.final_cash >= 0 and len(res.fills) > 10
    with pytest.raises(ValueError, match="empty"):
        run_market_maker([], cfg())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"order_size": D(5)}, "max_inventory"),
        ({"half_spread": D(0)}, "half_spread"),
        ({"target_ratio": D(2)}, "target_ratio"),
        ({"capital": D(0)}, "capital"),
    ],
)
def test_config_validation(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        cfg(**kwargs)
