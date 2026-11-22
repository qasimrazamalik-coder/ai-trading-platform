from decimal import Decimal

import numpy as np
import pytest

from quant_os_core.strategies.bots import (
    ComboConfig,
    DcaConfig,
    Side,
    bars_from_arrays,
    run_combo,
)

D = Decimal


def _bars(n: int = 300, seed: int = 5) -> list:  # type: ignore[type-arg]
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    wiggle = np.abs(rng.normal(0, 0.015, n)) * close
    return bars_from_arrays(
        [f"d{i:04d}" for i in range(n)], list(close + wiggle), list(close - wiggle), list(close)
    )


def test_combo_splits_capital_and_caps_core() -> None:
    dca = DcaConfig(D(10_000), D(200), 5, max_invested=D(9000), fee_rate=D("0.001"))
    cfg = ComboConfig(dca, grid_fraction=D("0.3"), grid_width=D("0.2"), grid_levels=6)
    assert cfg.grid_capital == D(3000)
    core = cfg.core_config()
    assert core.capital == D(7000) and core.max_invested == D(7000)
    no_cap = ComboConfig(DcaConfig(D(1000), D(10), 1)).core_config()
    assert no_cap.max_invested is None


def test_combo_run_is_sum_of_two_spot_sleeves() -> None:
    dca = DcaConfig(D(10_000), D(200), 5, fee_rate=D("0.001"))
    res = run_combo(_bars(), ComboConfig(dca, grid_width=D("0.1"), grid_levels=6))
    assert res.capital == D(10_000) and len(res.equity) == 300
    assert res.final_cash >= 0 and res.final_quantity >= 0
    held = sum((f.quantity if f.side is Side.BUY else -f.quantity for f in res.fills), D(0))
    assert held == res.final_quantity
    assert any(f.reason == "periodic" for f in res.fills)
    assert any(f.reason.startswith("grid_") for f in res.fills)
    assert [f.time for f in res.fills] == sorted(f.time for f in res.fills)
    with pytest.raises(ValueError, match="empty"):
        run_combo([], ComboConfig(dca))


def test_combo_validation() -> None:
    dca = DcaConfig(D(1000), D(10), 1)
    with pytest.raises(ValueError, match=r"capped at 0.5"):
        ComboConfig(dca, grid_fraction=D("0.6"))
    with pytest.raises(ValueError, match="grid_width"):
        ComboConfig(dca, grid_width=D(0))
