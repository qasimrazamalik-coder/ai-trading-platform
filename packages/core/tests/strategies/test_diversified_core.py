from datetime import date
from decimal import Decimal

import numpy as np
import pytest

from quant_os_core.strategies.diversified_core import (
    CoreMode,
    RebalanceOrder,
    is_rebalance_day,
    rebalance_orders,
    target_weights,
)


def _walk(seed: int, n: int = 260, vol: float = 0.01, drift: float = 0.0) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(100 * np.exp(np.cumsum(rng.normal(drift, vol, n))))


def test_growth_weights_favour_low_volatility_and_sum_to_one() -> None:
    closes = {"CALM": _walk(1, vol=0.005), "WILD": _walk(2, vol=0.02)}
    w = target_weights(closes, CoreMode.GROWTH)
    assert w["CALM"] > 3 * w["WILD"]
    assert Decimal("0.999") <= sum(w.values()) <= 1
    assert all(v >= 0 for v in w.values())


def test_defensive_holds_cash_for_assets_below_their_average() -> None:
    up = [100.0 + i * 0.5 for i in range(260)]
    down = [230.0 - i * 0.5 for i in range(260)]
    noise = np.random.default_rng(3).normal(0, 0.2, 260)
    closes = {"UP": list(np.array(up) + noise), "DOWN": list(np.array(down) + noise)}
    w = target_weights(closes, CoreMode.DEFENSIVE)
    assert w["DOWN"] == 0
    assert 0 < w["UP"] < 1  # the down asset's share stays in cash, not levered into UP


def test_target_weights_rejects_short_or_bad_history() -> None:
    with pytest.raises(ValueError, match="at least 200"):
        target_weights({"A": _walk(1, n=100)}, CoreMode.DEFENSIVE)
    with pytest.raises(ValueError, match="positive"):
        target_weights({"A": [*_walk(1), -1.0]}, CoreMode.GROWTH)
    with pytest.raises(ValueError):
        target_weights({})


def test_rebalance_day_is_first_run_of_each_month() -> None:
    assert is_rebalance_day(date(2026, 12, 1), None)
    assert is_rebalance_day(date(2026, 12, 1), date(2026, 11, 28))
    assert not is_rebalance_day(date(2026, 12, 15), date(2026, 12, 1))
    assert is_rebalance_day(date(2027, 1, 4), date(2026, 12, 31))


def test_rebalance_from_cash_buys_targets_without_exceeding_cash() -> None:
    targets = {"SPY": Decimal("0.5"), "GLD": Decimal("0.3")}
    orders = rebalance_orders(targets, {}, Decimal("50"))
    assert orders == [
        RebalanceOrder("GLD", "buy", Decimal("15.00")),
        RebalanceOrder("SPY", "buy", Decimal("25.00")),
    ]
    assert sum(o.notional for o in orders) <= Decimal("50")


def test_rebalance_skips_small_drift_and_sells_before_buying() -> None:
    targets = {"SPY": Decimal("0.5"), "GLD": Decimal("0.5"), "XLE": Decimal("0")}
    holdings = {"SPY": Decimal("50.5"), "GLD": Decimal("30"), "XLE": Decimal("19.5")}
    orders = rebalance_orders(targets, holdings, Decimal("0"))
    assert [o.side for o in orders] == ["sell", "buy"]
    assert orders[0] == RebalanceOrder("XLE", "sell", Decimal("19.50"))
    assert orders[1].symbol == "GLD"
    assert "SPY" not in {o.symbol for o in orders}  # 0.5 point drift is inside the band


def test_rebalance_scales_buys_down_when_cash_is_short() -> None:
    targets = {"SPY": Decimal("1")}
    # equity 100, wants 100 in SPY, has 60 held and 40 cash: buy is exactly funded
    orders = rebalance_orders(targets, {"SPY": Decimal("60")}, Decimal("40"))
    assert orders == [RebalanceOrder("SPY", "buy", Decimal("40.00"))]


def test_rebalance_drops_dust_orders_and_rejects_leverage() -> None:
    assert rebalance_orders({"SPY": Decimal("1")}, {"SPY": Decimal("99.5")}, Decimal("0.5")) == []
    with pytest.raises(ValueError, match="long-only"):
        rebalance_orders({"A": Decimal("0.7"), "B": Decimal("0.7")}, {}, Decimal("10"))
    with pytest.raises(ValueError, match="no margin"):
        rebalance_orders({"A": Decimal("1")}, {}, Decimal("-1"))
    with pytest.raises(ValueError):
        RebalanceOrder("A", "short", Decimal("1"))
