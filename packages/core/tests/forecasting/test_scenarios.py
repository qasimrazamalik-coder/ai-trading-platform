import numpy as np
import pytest

from quant_os_core.forecasting.scenarios import (
    LABEL,
    PathOutcome,
    ScenarioReport,
    block_bootstrap_paths,
    diversified_core_weights,
    garch_paths,
    log_returns_from_prices,
    model_paths,
    price_paths,
    run_path,
    run_scenarios,
)
from quant_os_core.strategies.diversified_core import CoreMode


def _returns(n=400, k=2, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0004, 0.01, size=(n, 1))
    return base + rng.normal(0, 0.005, size=(n, k))


def _prices(n=400, k=2, seed=0):
    r = _returns(n, k, seed)
    return 100 * np.exp(np.cumsum(r, axis=0))


def test_log_returns_from_prices():
    r = log_returns_from_prices([100.0, 110.0, 99.0])
    assert r.shape == (2, 1) and r[0, 0] == pytest.approx(np.log(1.1))
    with pytest.raises(ValueError):
        log_returns_from_prices([100.0, -1.0])


def test_block_bootstrap_uses_real_joint_rows():
    r = _returns()
    paths = block_bootstrap_paths(r, 45, 10, block=20, seed=1)
    assert paths.shape == (10, 45, 2)
    rows = {tuple(row) for row in r}
    assert all(tuple(row) in rows for row in paths.reshape(-1, 2))
    again = block_bootstrap_paths(r, 45, 10, block=20, seed=1)
    assert np.array_equal(paths, again)
    with pytest.raises(ValueError):
        block_bootstrap_paths(r, 10, 2, block=0)
    with pytest.raises(ValueError):
        block_bootstrap_paths(r, 0, 2)
    with pytest.raises(ValueError):
        block_bootstrap_paths(np.ones((5, 2)), 10, 2)


def test_garch_paths_shape_and_scale():
    r = _returns(600)
    paths = garch_paths(r, 30, 50, seed=2)
    assert paths.shape == (50, 30, 2) and np.isfinite(paths).all()
    assert 0.3 < paths.std() / r.std() < 3.0
    single = garch_paths(r[:, 0], 5, 3)
    assert single.shape == (3, 5, 1)
    with pytest.raises(ValueError):
        garch_paths(r[:50], 5, 3)


def test_model_paths_keep_correlation():
    r = _returns(800)
    paths = model_paths(r, 60, 200, seed=3)
    assert paths.shape == (200, 60, 2)
    flat = paths.reshape(-1, 2)
    assert np.corrcoef(flat.T)[0, 1] > 0.5
    assert model_paths(r[:, 0], 3, 2).shape == (2, 3, 1)


def test_price_paths():
    prices = price_paths([100.0, 50.0], np.zeros((3, 4, 2)) + np.log(1.01))
    assert prices.shape == (3, 5, 2)
    assert prices[0, 0, 1] == 50.0 and prices[0, -1, 0] == pytest.approx(100 * 1.01**4)
    with pytest.raises(ValueError):
        price_paths([100.0], np.zeros((3, 4, 2)))


def test_run_path_buy_and_hold_matches_asset():
    history = np.full((5, 1), 100.0)
    future = np.array([[100.0], [110.0], [121.0]])
    full = run_path(history, future, lambda h: np.array([1.0]), cost_bps=0.0)
    assert full.final_growth == pytest.approx(1.21)
    cash = run_path(history, future, lambda h: np.array([0.0]), cost_bps=0.0)
    assert cash.final_growth == 1.0 and cash.max_drawdown == 0.0
    costly = run_path(history, future, lambda h: np.array([1.0]), cost_bps=100.0)
    assert costly.final_growth == pytest.approx(1.21 * 0.99)
    down = run_path(
        history, np.array([[100.0], [50.0], [60.0]]), lambda h: np.array([1.0]), cost_bps=0
    )
    assert down.max_drawdown == pytest.approx(-0.5) and down.cagr < 0


def test_run_path_never_sees_the_future():
    history = np.full((3, 1), 100.0)
    future = np.array([[100.0], [101.0], [102.0], [103.0]])
    seen = []

    def spy(h):
        seen.append(h.shape[0])
        return np.array([0.5])

    run_path(history, future, spy, rebalance_every=1, cost_bps=0.0)
    assert seen == [3, 4, 5]


def test_run_path_guards():
    history = np.full((3, 2), 100.0)
    future = np.full((3, 2), 100.0)
    with pytest.raises(ValueError):
        run_path(history, future, lambda h: np.array([0.8, 0.8]))
    with pytest.raises(ValueError):
        run_path(history, future, lambda h: np.array([-0.1, 0.5]))
    with pytest.raises(ValueError):
        run_path(history, future[:1], lambda h: np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        run_path(history[:, :1], future, lambda h: np.array([0.5, 0.5]))


def test_wipeout_cagr():
    outcome = run_path(
        np.full((2, 1), 100.0), np.array([[100.0], [1e-9]]), lambda h: np.array([1.0]), cost_bps=0.0
    )
    assert outcome.final_growth < 1e-6


def test_report_statistics_and_label():
    report = ScenarioReport(
        "test",
        4,
        252,
        (
            PathOutcome(0.10, -0.05, 1.10),
            PathOutcome(-0.10, -0.30, 0.90),
            PathOutcome(0.05, -0.10, 1.05),
            PathOutcome(0.20, -0.25, 1.20),
        ),
    )
    assert report.probability_of_loss == 0.25
    assert report.probability_drawdown_worse_than(-0.2) == 0.5
    assert report.percentiles("cagr", (50,))[50] == pytest.approx(0.075)
    text = report.summary()
    assert text.startswith(LABEL) and "P(loss" in text


def test_run_scenarios_with_diversified_core():
    prices = _prices(320, 2)
    paths = block_bootstrap_paths(log_returns_from_prices(prices), 63, 5, seed=4)
    fn = diversified_core_weights(("A", "B"), CoreMode.GROWTH)
    w = fn(prices)
    assert w.shape == (2,) and 0.99 <= w.sum() <= 1.0
    report = run_scenarios(prices, paths, fn, generator="bootstrap")
    assert report.n_paths == 5 and report.horizon_days == 63 and len(report.outcomes) == 5
    defensive = run_scenarios(
        prices, paths, diversified_core_weights(("A", "B")), generator="bootstrap"
    )
    assert all(o.final_growth > 0 for o in defensive.outcomes)
    single = run_scenarios(
        prices[:, 0], paths[:, :, :1], lambda h: np.array([1.0]), generator="one"
    )
    assert single.n_paths == 5
    with pytest.raises(ValueError):
        run_scenarios(prices, paths, fn, generator="x", rebalance_every=0)
