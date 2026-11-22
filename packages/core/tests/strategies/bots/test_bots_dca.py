from decimal import Decimal

import pytest

from quant_os_core.strategies.bots import Bar, DcaBot, DcaConfig, Side, run_dca

D = Decimal


def bar(i: int, high: float, low: float, close: float) -> Bar:
    return Bar(f"t{i:03d}", D(str(high)), D(str(low)), D(str(close)))


def flat(i: int, price: float) -> Bar:
    return bar(i, price, price, price)


def test_periodic_buys_respect_interval_and_cap() -> None:
    cfg = DcaConfig(D(1000), D(100), 2, max_invested=D(250), fee_rate=D(0))
    res = run_dca([flat(i, 10) for i in range(10)], cfg)
    buys = [f for f in res.fills if f.side is Side.BUY]
    assert [f.time for f in buys] == ["t000", "t002", "t004"]
    assert sum(f.notional for f in buys) == D(250)  # third buy trimmed to the cap
    assert res.final_cash == D(750)
    assert res.equity[-1] == D(1000)
    assert "cycles_closed=0" in res.notes


def test_cash_limits_buys_without_cap() -> None:
    cfg = DcaConfig(D(150), D(100), 1, fee_rate=D(0))
    res = run_dca([flat(i, 10) for i in range(4)], cfg)
    assert len(res.fills) == 2 and res.final_cash == 0


def test_safety_orders_fill_on_dips_with_scaled_size() -> None:
    cfg = DcaConfig(
        D(10_000),
        D(100),
        100,
        safety_orders=2,
        safety_step=D("0.1"),
        safety_volume_scale=D(2),
        fee_rate=D(0),
    )
    bot = DcaBot(cfg)
    bot.step(0, flat(0, 100))  # base buy at 100, reference 100
    bot.step(1, bar(1, 100, 90, 95))  # touch 90 only: no fill
    assert len(bot.wallet.fills) == 1
    bot.step(2, bar(2, 95, 75, 78))  # through 90 and 80 -> both safeties
    prices = [f.price for f in bot.wallet.fills]
    notionals = [f.notional for f in bot.wallet.fills]
    assert prices == [D(100), D(90), D(80)]
    assert notionals == [D(100), pytest.approx(D(200)), pytest.approx(D(400))]
    assert bot.safety_filled == 2
    bot.step(3, bar(3, 80, 50, 60))  # no more safeties left
    assert len(bot.wallet.fills) == 3


def test_safety_step_scale_offsets() -> None:
    cfg = DcaConfig(
        D(1000), D(10), 1, safety_orders=3, safety_step=D("0.05"), safety_step_scale=D(2)
    )
    assert cfg.safety_trigger_offset(1) == D("0.05")
    assert cfg.safety_trigger_offset(2) == D("0.15")
    assert cfg.safety_trigger_offset(3) == D("0.35")
    assert cfg.safety_amount(2) == D("22.5")


def test_take_profit_sells_all_and_restarts_cycle() -> None:
    cfg = DcaConfig(D(1000), D(100), 3, take_profit=D("0.1"), fee_rate=D(0))
    bot = DcaBot(cfg)
    bot.step(0, flat(0, 100))
    assert bot.average_cost == D(100)
    bot.step(1, bar(1, 110, 100, 105))  # touch only
    assert bot.quantity == 1
    bot.step(2, bar(2, 112, 104, 108))  # through 110 -> sell all at 110
    assert bot.quantity == 0 and bot.cycles_closed == 1 and bot.average_cost is None
    sell = bot.wallet.fills[-1]
    assert sell.side is Side.SELL and sell.price == D(110) and sell.quantity == 1
    bot.step(3, flat(3, 120))  # next periodic buy opens a new cycle at 120
    assert bot.reference == D(120)


def test_take_profit_blocked_in_a_bar_with_a_safety_buy() -> None:
    cfg = DcaConfig(
        D(1000),
        D(100),
        50,
        safety_orders=1,
        safety_step=D("0.1"),
        take_profit=D("0.05"),
        fee_rate=D(0),
    )
    bot = DcaBot(cfg)
    bot.step(0, flat(0, 100))
    bot.step(1, bar(1, 120, 85, 100))  # safety buys at 90; TP must wait
    assert [f.side for f in bot.wallet.fills] == [Side.BUY, Side.BUY]
    bot.step(2, bar(2, 120, 100, 110))
    assert bot.wallet.fills[-1].side is Side.SELL
    assert bot.quantity == 0


def test_fees_are_part_of_average_cost() -> None:
    cfg = DcaConfig(D(1000), D(101), 10, fee_rate=D("0.01"))
    bot = DcaBot(cfg)
    bot.step(0, flat(0, 100))
    assert bot.cost_basis <= D(101)
    assert bot.average_cost is not None and bot.average_cost == pytest.approx(D(101))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"interval": 0}, "interval"),
        ({"max_invested": D(5000)}, "no leverage"),
        ({"safety_orders": -1}, "negative"),
        ({"safety_orders": 5, "safety_step": D("0.3")}, "below zero"),
        ({"take_profit": D(0)}, "take_profit"),
        ({"fee_rate": D(1)}, "fee_rate"),
    ],
)
def test_config_validation(kwargs: dict[str, object], match: str) -> None:
    base: dict[str, object] = {"capital": D(1000), "order_amount": D(10), "interval": 1}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        DcaConfig(**base)  # type: ignore[arg-type]


def test_run_dca_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        run_dca([], DcaConfig(D(1), D(1), 1))


def test_safety_order_skipped_when_cash_is_gone() -> None:
    cfg = DcaConfig(D(100), D(100), 100, safety_orders=1, safety_step=D("0.1"), fee_rate=D(0))
    bot = DcaBot(cfg)
    bot.step(0, flat(0, 100))
    bot.step(1, bar(1, 100, 80, 85))
    assert len(bot.wallet.fills) == 1 and bot.safety_filled == 0
