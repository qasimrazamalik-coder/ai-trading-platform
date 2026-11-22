from decimal import Decimal

import numpy as np
import pytest

from quant_os_core.strategies.bots import (
    Bar,
    GridBot,
    GridConfig,
    GridStatus,
    RollingGrid,
    Side,
    Spacing,
    SpotWallet,
    StopAction,
    bars_from_arrays,
    grid_levels,
    run_grid,
    run_rolling_grid,
)

D = Decimal


def bar(i: int, high: float, low: float, close: float) -> Bar:
    return Bar(f"t{i:03d}", D(str(high)), D(str(low)), D(str(close)))


def config(**overrides: object) -> GridConfig:
    base: dict[str, object] = {
        "lower": D(90),
        "upper": D(110),
        "levels": 3,
        "capital": D(1000),
        "spacing": Spacing.ARITHMETIC,
        "fee_rate": D(0),
        "stop_margin": D("0.05"),
        "stop_action": StopAction.SELL,
        "slippage": D(0),
    }
    base.update(overrides)
    return GridConfig(**base)  # type: ignore[arg-type]


def test_grid_levels_arithmetic_and_geometric() -> None:
    assert grid_levels(D(100), D(120), 3, Spacing.ARITHMETIC) == (D(100), D(110), D(120))
    geo = grid_levels(D(100), D(200), 3, Spacing.GEOMETRIC)
    assert geo[0] == D(100) and geo[-1] == D(200)
    assert float(geo[1]) == pytest.approx(100 * 2**0.5)
    with pytest.raises(ValueError, match="above lower"):
        grid_levels(D(100), D(100), 3, Spacing.ARITHMETIC)
    with pytest.raises(ValueError, match="at least 2"):
        grid_levels(D(100), D(110), 1, Spacing.ARITHMETIC)


def test_config_validation_and_exit_bounds() -> None:
    cfg = config()
    assert cfg.exit_low == D("85.5") and cfg.exit_high == D("115.5")
    with pytest.raises(ValueError, match="fee_rate"):
        config(fee_rate=D("1.5"))
    with pytest.raises(ValueError, match="capital"):
        config(capital=D(0))


def test_buy_below_then_sell_owned_lot_one_level_up() -> None:
    bot = GridBot(config())
    bot.step(0, bar(0, 105, 105, 105))  # sets reference, no orders yet
    assert bot.wallet.fills == []
    bot.step(1, bar(1, 104, 99, 101))  # trades through 100 -> buy cell 1
    assert bot.lots[1] == D(5) and bot.wallet.cash == D(500)
    bot.step(2, bar(2, 111, 100.5, 108))  # trades through 110 -> sell that lot
    sides = [(f.side, f.price, f.quantity) for f in bot.wallet.fills]
    assert sides == [(Side.BUY, D(100), D(5)), (Side.SELL, D(110), D(5))]
    assert bot.wallet.cash == D(1050) and bot.wallet.quantity == 0
    assert bot.round_trips == 1


def test_touch_is_not_a_fill() -> None:
    bot = GridBot(config())
    bot.step(0, bar(0, 105, 105, 105))
    bot.step(1, bar(1, 104, 100, 101))  # low == level: no fill
    assert bot.wallet.fills == []


def test_no_buy_above_market_and_no_same_bar_round_trip() -> None:
    bot = GridBot(config())
    bot.step(0, bar(0, 95, 95, 95))
    bot.step(1, bar(1, 108, 89, 100.5))  # cell 1 (100) not armed: prev close 95 < 100
    assert [f.price for f in bot.wallet.fills] == [D(90)]
    bot2 = GridBot(config())
    bot2.step(0, bar(0, 105, 105, 105))
    bot2.step(1, bar(1, 111, 99, 105))  # buys 100, cannot sell 110 in the same bar
    assert [f.side for f in bot2.wallet.fills] == [Side.BUY]
    bot2.step(2, bar(2, 111, 104, 105))
    assert [f.side for f in bot2.wallet.fills] == [Side.BUY, Side.SELL]


def test_one_fill_per_level_per_bar() -> None:
    bot = GridBot(config())
    bot.step(0, bar(0, 95, 95, 95))
    bot.step(1, bar(1, 101, 89, 101))  # buys cell 0 at 90 (held through the bar)
    bot.step(2, bar(2, 102, 99, 101))  # level 100: cell-0 sell and cell-1 buy compete
    level_100 = [f for f in bot.wallet.fills if f.price == D(100)]
    assert len(level_100) == 1 and level_100[0].side is Side.SELL
    assert bot.lots == [D(0), D(0)]


def test_cell_cannot_rebuy_in_bar_it_sold() -> None:
    bot = GridBot(config(levels=5))  # 90, 95, 100, 105, 110
    bot.step(0, bar(0, 98, 98, 98))
    bot.step(1, bar(1, 98, 94, 96))  # buy cell 1 at 95
    bot.step(2, bar(2, 101, 94, 99))  # sells cell 1 at 100; cell 1 may not rebuy at 95
    sells = [f for f in bot.wallet.fills if f.side is Side.SELL]
    buys_at_95 = [f for f in bot.wallet.fills if f.side is Side.BUY and f.price == D(95)]
    assert len(sells) == 1 and len(buys_at_95) == 1


def test_stop_sells_grid_lots_at_next_close() -> None:
    bot = GridBot(config(slippage=D("0.01"), fee_rate=D("0.001")))
    bot.step(0, bar(0, 105, 105, 105))
    bot.step(1, bar(1, 104, 99, 101))
    bot.step(2, bar(2, 101, 84, 85))  # close below 85.5 exit -> exiting
    assert bot.status is GridStatus.EXITING
    bot.step(3, bar(3, 86, 70, 80))
    assert bot.status is GridStatus.STOPPED
    exit_fill = bot.wallet.fills[-1]
    assert exit_fill.reason == "grid_exit" and exit_fill.price == D("79.2")
    assert bot.wallet.quantity == 0
    count = len(bot.wallet.fills)
    bot.step(4, bar(4, 120, 60, 100))
    assert len(bot.wallet.fills) == count  # stopped grid does nothing


def test_upper_exit_with_no_lots_stops_immediately() -> None:
    bot = GridBot(config())
    bot.step(0, bar(0, 105, 105, 105))
    bot.step(1, bar(1, 120, 104, 118))
    assert bot.status is GridStatus.STOPPED and bot.wallet.fills == []


def test_pause_keeps_inventory_and_resumes() -> None:
    bot = GridBot(config(stop_action=StopAction.PAUSE))
    bot.step(0, bar(0, 105, 105, 105))
    bot.step(1, bar(1, 104, 99, 101))
    bot.step(2, bar(2, 101, 80, 82))  # also buys the 90 level on the way down
    assert bot.status is GridStatus.PAUSED
    assert bot.lots[1] == D(5) and D("5.5555") < bot.lots[0] <= D(500) / D(90)
    assert len(bot.wallet.fills) == 2
    bot.step(3, bar(3, 120, 70, 84))  # paused: no fills even though levels traded
    assert len(bot.wallet.fills) == 2
    bot.step(4, bar(4, 100, 84, 95))
    assert bot.status is GridStatus.ACTIVE
    bot.step(5, bar(5, 111, 94, 109))
    assert bot.wallet.fills[-1].side is Side.SELL and bot.wallet.quantity == 0


def test_run_grid_marks_equity_and_notes() -> None:
    bars = [bar(0, 105, 105, 105), bar(1, 104, 99, 101), bar(2, 111, 100.5, 108)]
    res = run_grid(bars, config())
    assert len(res.equity) == 3 and res.equity[-1] == D(1050)
    assert any("trades through" in n for n in res.notes)
    assert "round_trips=1" in res.notes
    with pytest.raises(ValueError, match="empty"):
        run_grid([], config())


def _random_walk(n: int, seed: int) -> list[Bar]:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    spread = np.abs(rng.normal(0, 0.01, n)) * close
    times = [f"d{i:04d}" for i in range(n)]
    return bars_from_arrays(times, list(close + spread), list(close - spread), list(close))


def test_rolling_grid_recentres_and_never_goes_negative() -> None:
    bars = _random_walk(400, 3)
    res = run_rolling_grid(
        bars, D(10_000), width=D("0.1"), levels=6, fee_rate=D("0.001"), slippage=D("0.001")
    )
    assert any(n.startswith("recentres=") and n != "recentres=0" for n in res.notes)
    held = D(0)
    for fill in res.fills:
        held += fill.quantity if fill.side is Side.BUY else -fill.quantity
        assert held >= 0
    assert res.final_cash >= 0 and res.final_quantity >= 0
    assert held == res.final_quantity
    with pytest.raises(ValueError, match="empty"):
        run_rolling_grid([], D(1), width=D("0.1"), levels=3)


def test_rolling_grid_validation() -> None:
    with pytest.raises(ValueError, match="width"):
        RollingGrid(D(100), width=D(0), levels=3)


def test_grid_skips_buys_when_wallet_is_empty() -> None:
    bot = GridBot(config(), wallet=SpotWallet(D(0)))
    bot.step(0, bar(0, 105, 105, 105))
    bot.step(1, bar(1, 104, 85.6, 101))
    assert bot.wallet.fills == [] and bot.lots == [D(0), D(0)]
