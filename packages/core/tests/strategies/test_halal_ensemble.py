import importlib.util
from pathlib import Path

import numpy as np
import pytest
from quant_os_core.strategies.halal_ensemble import (
    PriceTable,
    buy_hold,
    evaluate_book,
    renormalized_mix,
    run_weights,
    strategy_book,
    trend_200,
)

SYMBOLS = ("SPY", "QQQ", "GLD", "XLK", "XLE", "XLV", "XLP", "XLY", "XLI", "XLB", "XLU")


def _dates(n: int) -> tuple[str, ...]:
    days = np.arange(
        np.datetime64("2010-01-04"), np.datetime64("2030-01-01"), dtype="datetime64[D]"
    )
    weekdays = [str(d) for d in days if np.is_busday(d)]
    return tuple(weekdays[:n])


def _table(n: int = 900, seed: int = 7) -> PriceTable:
    rng = np.random.default_rng(seed)
    drift = np.linspace(0.0002, 0.0006, len(SYMBOLS))
    rets = rng.normal(drift, 0.012, size=(n, len(SYMBOLS)))
    prices = 100 * np.cumprod(1 + rets, axis=0)
    return PriceTable(_dates(n), SYMBOLS, prices)


def test_price_table_validation() -> None:
    with pytest.raises(ValueError):
        PriceTable(("2020-01-02",), ("A",), np.array([[0.0]]))
    with pytest.raises(ValueError):
        PriceTable(("2020-01-03", "2020-01-02"), ("A",), np.array([[1.0], [1.0]]))


def test_buy_hold_matches_price_ratio_without_costs() -> None:
    table = _table()
    result = run_weights(table, buy_hold(table), name="bh", cost_per_side=0.0)
    spy = table.prices[:, table.col("SPY")]
    assert result.equity[-1] == pytest.approx(spy[-1] / spy[0], rel=1e-9)


def test_no_lookahead_weights_use_only_past_prices() -> None:
    table = _table()
    full = trend_200(table)
    truncated = PriceTable(table.dates[:600], table.symbols, table.prices[:600])
    assert np.array_equal(full[:600], trend_200(truncated))


def test_rejects_leverage_and_shorts() -> None:
    table = _table()
    with pytest.raises(ValueError):
        run_weights(table, buy_hold(table) * 2, name="levered")
    with pytest.raises(ValueError):
        run_weights(table, -buy_hold(table), name="short")


def test_renormalized_mix_uses_full_capital_when_any_sleeve_is_active() -> None:
    a = np.array([[0.5, 0.0], [0.0, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 0.0]])
    mix = renormalized_mix([a, b])
    assert mix[0].sum() == pytest.approx(1.0)
    assert mix[1].sum() == 0.0


def test_book_and_report_on_synthetic_data() -> None:
    table = _table()
    book = strategy_book(table)
    assert set(book) >= {"buy_hold", "ensemble_v3", "donchian", "dual_momentum"}
    for weights in book.values():
        assert np.all(weights >= 0) and np.all(weights.sum(axis=1) <= 1 + 1e-9)
    report = evaluate_book(table, book, extra_trial_sharpes=(0.1, 0.5))
    assert 0.0 <= report.deflated_sharpe_probability <= 1.0
    assert report.results["ensemble_v3"].max_drawdown <= 0
    assert isinstance(report.passes, bool)


def test_research_script_runs_on_a_csv(tmp_path: Path) -> None:
    table = _table()
    csv_path = tmp_path / "prices.csv"
    lines = ["date," + ",".join(SYMBOLS)]
    lines += [
        d + "," + ",".join(f"{p:.4f}" for p in row)
        for d, row in zip(table.dates, table.prices, strict=True)
    ]
    csv_path.write_text("\n".join(lines))
    script = Path(__file__).parents[4] / "research" / "backtests" / "run_halal_v3.py"
    spec = importlib.util.spec_from_file_location("run_halal_v3", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.load(csv_path).prices.shape == table.prices.shape
    assert module.main(["run", "--csv", str(csv_path)]) in (0, 1)
