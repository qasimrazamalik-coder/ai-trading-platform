from decimal import Decimal

import numpy as np
import pytest

from quant_os_core.strategies.bots import (
    Bar,
    InsufficientFunds,
    InsufficientInventory,
    Side,
    SpotWallet,
    bars_from_arrays,
    buy_and_hold,
)
from quant_os_core.strategies.bots.common import require_fraction, to_decimal

D = Decimal


def test_to_decimal_avoids_float_noise() -> None:
    assert to_decimal(0.1) == D("0.1")
    assert to_decimal("2.5") == D("2.5")
    assert to_decimal(3) == D(3)
    assert to_decimal(D("1.25")) == D("1.25")
    with pytest.raises(ValueError, match="finite"):
        to_decimal(float("inf"))


def test_bar_validation() -> None:
    with pytest.raises(ValueError, match="> 0"):
        Bar("t", D(1), D(0), D(1))
    with pytest.raises(ValueError, match="below low"):
        Bar("t", D(1), D(2), D("1.5"))


def test_bars_from_arrays_widens_range_to_close() -> None:
    bars = bars_from_arrays(["a", "b"], [10.0, 10.0], [9.0, 9.0], [10.5, 8.5])
    assert bars[0].high == D("10.5") and bars[0].low == D(9)
    assert bars[1].low == D("8.5") and bars[1].high == D(10)
    with pytest.raises(ValueError, match="equal length"):
        bars_from_arrays(["a"], [1.0, 2.0], [1.0], [1.0])


def test_wallet_refuses_leverage_and_shorting() -> None:
    wallet = SpotWallet(D(100))
    fill = wallet.buy("t", D(10), D(5), D("0.01"), "test")
    assert fill.side is Side.BUY and fill.fee == D("0.5") and fill.notional == D(50)
    assert wallet.cash == D("49.5") and wallet.quantity == D(5)
    with pytest.raises(InsufficientFunds):
        wallet.buy("t", D(10), D(5), D("0.01"), "too much")
    with pytest.raises(InsufficientInventory, match="no short selling"):
        wallet.sell("t", D(10), D(6), D(0), "short")
    wallet.sell("t", D(12), D(5), D("0.01"), "exit")
    assert wallet.quantity == 0
    assert wallet.cash == D("49.5") + D(60) - D("0.6")
    assert wallet.equity(D(999)) == wallet.cash
    with pytest.raises(ValueError, match="quantity must be > 0"):
        wallet.sell("t", D(1), D(0), D(0), "zero")
    with pytest.raises(ValueError, match="negative"):
        SpotWallet(D(-1))


def test_max_buy_quantity_fits_cash_and_budget() -> None:
    wallet = SpotWallet(D(100))
    qty = wallet.max_buy_quantity(D(3), D("0.001"), D(1000))
    assert qty * D(3) * D("1.001") <= D(100)
    wallet.buy("t", D(3), qty, D("0.001"), "all in")
    assert wallet.cash >= 0
    assert wallet.max_buy_quantity(D(3), D(0), D(0)) == 0
    assert SpotWallet(D(10)).max_buy_quantity(D(2), D(0), D(4)) == D(2)


def test_require_fraction_bounds() -> None:
    assert require_fraction(D(0), "x") == 0
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        require_fraction(D(0), "x", allow_zero=False)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        require_fraction(D(1), "x")


def test_buy_and_hold_result() -> None:
    bars = bars_from_arrays(["a", "b", "c"], [10, 12, 15], [10, 12, 15], [10, 12, 15])
    res = buy_and_hold(bars, D(1000), D("0.01"))
    assert len(res.fills) == 1 and res.final_cash < D("0.01")
    assert res.equity[-1] == pytest.approx(D(1000) / D("1.01") * D("1.5"), rel=D("1e-9"))
    assert float(res.total_return) == pytest.approx(1.5 / 1.01 - 1, rel=1e-9)
    rets = res.returns()
    assert rets.size == 3
    assert rets[0] == pytest.approx(1 / 1.01 - 1)
    np.testing.assert_allclose(rets[1:], [0.2, 0.25])
    assert res.fees_paid == res.fills[0].fee
    with pytest.raises(ValueError, match="empty"):
        buy_and_hold([], D(1), D(0))
