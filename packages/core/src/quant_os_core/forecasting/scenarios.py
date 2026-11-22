"""Scenario backtests: run a weights function across many simulated future price paths.

THESE ARE SCENARIOS, NOT PREDICTIONS. Each path is one plausible future built from the past
under stated assumptions; the spread of outcomes shows what the allocation could live
through, not what it will earn. Three generators, each with different assumptions:

* ``block_bootstrap_paths``: resample blocks of real historical days (all assets on the
  same days, so correlations and short-run clustering survive). Assumes the future looks
  like reshuffled pieces of the past.
* ``garch_paths``: filtered historical simulation. A GARCH(1,1) per asset rescales jointly
  resampled standardized residuals, so volatility clusters and starts from today's level.
* ``model_paths``: an AR(1) mean per asset plus correlated Gaussian shocks with the
  historical residual covariance. The thinnest tails of the three; a lower bound on risk.

``run_scenarios`` then applies any weights function (for example the diversified core) on
each path: weights decided at a close are applied from the next day, rebalanced every
``rebalance_every`` days, with costs on turnover. Outcomes: CAGR, worst drawdown and final
growth per path, summarised as percentiles and loss probabilities.

Research only: nothing here sends or sizes an order.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quant_os_core.forecasting.models import ARModel
from quant_os_core.forecasting.volatility import Garch11
from quant_os_core.strategies.diversified_core import CoreMode, target_weights

FloatArray = npt.NDArray[np.float64]
WeightsFn = Callable[[FloatArray], FloatArray]  # price history [t, k] -> weights [k]
TRADING_DAYS = 252
LABEL = "SCENARIOS, NOT PREDICTIONS"


def _log_returns(values: npt.ArrayLike, min_rows: int = 30) -> FloatArray:
    r = np.asarray(values, dtype=np.float64)
    if r.ndim == 1:
        r = r[:, None]
    if r.ndim != 2 or r.shape[0] < min_rows or not np.isfinite(r).all():
        raise ValueError(f"returns must be finite, [t] or [t, k], with at least {min_rows} rows")
    return r


def _check(horizon: int, n_paths: int) -> None:
    if horizon < 1 or n_paths < 1:
        raise ValueError("horizon and n_paths must be at least 1")


def log_returns_from_prices(prices: npt.ArrayLike) -> FloatArray:
    p = np.asarray(prices, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    if p.ndim != 2 or p.shape[0] < 2 or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("prices must be finite, positive, [t] or [t, k] with 2+ rows")
    return np.diff(np.log(p), axis=0)


# ---------------------------------------------------------------- generators


def block_bootstrap_paths(
    returns: npt.ArrayLike, horizon: int, n_paths: int, *, block: int = 20, seed: int = 7
) -> FloatArray:
    """Paths ``[n_paths, horizon, k]`` of log returns from joint moving blocks of history."""

    r = _log_returns(returns)
    _check(horizon, n_paths)
    if not 1 <= block <= r.shape[0]:
        raise ValueError("block must be between 1 and the number of rows")
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(horizon / block)
    starts = rng.integers(0, r.shape[0] - block + 1, size=(n_paths, n_blocks))
    offsets = np.arange(block)
    rows = (starts[:, :, None] + offsets).reshape(n_paths, -1)[:, :horizon]
    return np.asarray(r[rows], dtype=np.float64)


def garch_paths(returns: npt.ArrayLike, horizon: int, n_paths: int, *, seed: int = 7) -> FloatArray:
    """Filtered historical simulation with one GARCH(1,1) per asset (needs 100+ rows)."""

    r = _log_returns(returns, 100)
    _check(horizon, n_paths)
    k = r.shape[1]
    models = [Garch11.fit(r[:, j]) for j in range(k)]
    mus = r.mean(axis=0)
    z = np.empty_like(r)
    for j, model in enumerate(models):
        var = model.conditional_variance(r[:, j])  # var[t] is the forecast for t + 1
        prior = np.concatenate([[float(np.var(r[:, j], ddof=1))], var[:-1]])
        z[:, j] = (r[:, j] - mus[j]) / np.sqrt(prior)
    rng = np.random.default_rng(seed)
    picks = z[rng.integers(0, z.shape[0], size=(n_paths, horizon))]  # joint rows keep correlation
    out = np.empty((n_paths, horizon, k))
    omega = np.array([m.omega for m in models])
    alpha = np.array([m.alpha for m in models])
    beta = np.array([m.beta for m in models])
    var_t = np.tile([m.last_variance for m in models], (n_paths, 1))
    for h in range(horizon):
        shock = np.sqrt(var_t) * picks[:, h]
        out[:, h] = mus + shock
        var_t = omega + alpha * shock**2 + beta * var_t
    return out


def model_paths(returns: npt.ArrayLike, horizon: int, n_paths: int, *, seed: int = 7) -> FloatArray:
    """AR(1) mean per asset with correlated Gaussian shocks (historical residual covariance)."""

    r = _log_returns(returns)
    _check(horizon, n_paths)
    k = r.shape[1]
    models = [ARModel.fit(r[:, j], order=1) for j in range(k)]
    c = np.array([m.intercept for m in models])
    phi = np.array([float(m.coefficients[0]) for m in models])
    resid = r[1:] - (c + phi * r[:-1])
    cov = np.atleast_2d(np.cov(resid, rowvar=False))
    chol = np.linalg.cholesky(cov + np.eye(k) * 1e-12)
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, horizon, k)) @ chol.T
    out = np.empty((n_paths, horizon, k))
    prev = np.tile(r[-1], (n_paths, 1))
    for h in range(horizon):
        prev = c + phi * prev + shocks[:, h]
        out[:, h] = prev
    return out


def price_paths(last_prices: npt.ArrayLike, log_returns: npt.ArrayLike) -> FloatArray:
    """Turn log-return paths ``[n, h, k]`` into price paths ``[n, h + 1, k]`` from today."""

    last = np.asarray(last_prices, dtype=np.float64).reshape(-1)
    paths = np.asarray(log_returns, dtype=np.float64)
    if paths.ndim != 3 or paths.shape[2] != last.size or (last <= 0).any():
        raise ValueError("log_returns must be [n, h, k] with k positive last prices")
    growth = np.exp(np.cumsum(paths, axis=1))
    start = np.broadcast_to(last, (paths.shape[0], 1, last.size))
    return np.concatenate([start, last * growth], axis=1)


# ---------------------------------------------------------------- running a strategy


@dataclass(frozen=True, slots=True)
class PathOutcome:
    cagr: float
    max_drawdown: float  # negative fraction, e.g. -0.25
    final_growth: float  # ending equity / starting equity


def _max_drawdown(equity: FloatArray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def run_path(
    history: FloatArray,
    future: FloatArray,
    weights_fn: WeightsFn,
    *,
    rebalance_every: int = 21,
    cost_bps: float = 10.0,
) -> PathOutcome:
    """Equity curve of ``weights_fn`` over ``future`` prices ``[h + 1, k]`` (row 0 = today).

    Weights are decided at the close using all prices so far and earn the next day's return;
    they must be long-only and sum to at most 1 (the rest is cash at zero return).
    """

    if future.ndim != 2 or future.shape[0] < 2 or history.ndim != 2:
        raise ValueError("future must be [h + 1, k] and history [t, k]")
    if history.shape[1] != future.shape[1]:
        raise ValueError("history and future must have the same assets")
    prices = np.concatenate([history, future[1:]], axis=0)
    today = history.shape[0] - 1
    steps = future.shape[0] - 1
    k = future.shape[1]
    weights = np.zeros(k)
    equity = np.empty(steps + 1)
    equity[0] = 1.0
    cost = cost_bps / 10_000.0
    for h in range(steps):
        t = today + h
        if h % rebalance_every == 0:
            target = np.asarray(weights_fn(prices[: t + 1]), dtype=np.float64)
            if target.shape != (k,) or (target < -1e-12).any() or target.sum() > 1 + 1e-9:
                raise ValueError("weights must be long-only, one per asset, summing to <= 1")
            equity[h] = equity[h] * (1 - cost * float(np.abs(target - weights).sum()))
            weights = target
        growth = prices[t + 1] / prices[t]
        port = float(weights @ growth + (1 - weights.sum()))
        equity[h + 1] = equity[h] * port
        weights = weights * growth / port  # drift with prices until the next rebalance
    years = steps / TRADING_DAYS
    final = float(equity[-1])
    cagr = final ** (1 / years) - 1 if final > 0 else -1.0
    return PathOutcome(cagr, _max_drawdown(equity), final)


@dataclass(frozen=True, slots=True)
class ScenarioReport:
    generator: str
    n_paths: int
    horizon_days: int
    outcomes: tuple[PathOutcome, ...]

    def _values(self, attr: str) -> FloatArray:
        return np.array([getattr(o, attr) for o in self.outcomes], dtype=np.float64)

    def percentiles(
        self, attr: str, qs: Sequence[float] = (5, 25, 50, 75, 95)
    ) -> dict[float, float]:
        values = self._values(attr)
        return {q: float(np.percentile(values, q)) for q in qs}

    @property
    def probability_of_loss(self) -> float:
        return float(np.mean(self._values("final_growth") < 1.0))

    def probability_drawdown_worse_than(self, level: float) -> float:
        return float(np.mean(self._values("max_drawdown") < level))

    def summary(self) -> str:
        cagr = self.percentiles("cagr")
        dd = self.percentiles("max_drawdown")
        fmt = "  ".join
        return "\n".join(
            [
                f"{LABEL}: {self.generator}, {self.n_paths} paths x {self.horizon_days} days",
                "  CAGR percentiles      " + fmt(f"p{q:g} {v:+.1%}" for q, v in cagr.items()),
                "  drawdown percentiles  " + fmt(f"p{q:g} {v:+.1%}" for q, v in dd.items()),
                f"  P(loss over horizon) {self.probability_of_loss:.1%}   "
                f"P(drawdown worse than -20%) {self.probability_drawdown_worse_than(-0.20):.1%}",
            ]
        )


def run_scenarios(
    history_prices: npt.ArrayLike,
    log_return_paths: npt.ArrayLike,
    weights_fn: WeightsFn,
    *,
    generator: str,
    rebalance_every: int = 21,
    cost_bps: float = 10.0,
) -> ScenarioReport:
    """Apply ``weights_fn`` on every path; ``history_prices`` ``[t, k]`` ends today."""

    hist = np.asarray(history_prices, dtype=np.float64)
    if hist.ndim == 1:
        hist = hist[:, None]
    paths = np.asarray(log_return_paths, dtype=np.float64)
    if rebalance_every < 1 or cost_bps < 0:
        raise ValueError("rebalance_every must be >= 1 and cost_bps >= 0")
    prices = price_paths(hist[-1], paths)
    outcomes = tuple(
        run_path(hist, p, weights_fn, rebalance_every=rebalance_every, cost_bps=cost_bps)
        for p in prices
    )
    return ScenarioReport(generator, paths.shape[0], paths.shape[1], outcomes)


def diversified_core_weights(
    symbols: Sequence[str], mode: CoreMode = CoreMode.DEFENSIVE
) -> WeightsFn:
    """Adapter: the diversified core's ``target_weights`` as a weights function on arrays."""

    names = tuple(symbols)

    def weights(history: FloatArray) -> FloatArray:
        closes = {s: history[:, j].tolist() for j, s in enumerate(names)}
        target = target_weights(closes, mode)
        return np.array([float(target[s]) for s in names], dtype=np.float64)

    return weights
