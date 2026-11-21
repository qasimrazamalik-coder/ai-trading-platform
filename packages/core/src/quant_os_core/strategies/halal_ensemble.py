"""Reproducible version of the halal ensemble backtest from the research report (section 9H).

All strategies are long-only and unlevered; idle cash earns nothing (no interest). Weights
are decided on the close of day ``t`` and earn the return from ``t`` to ``t+1``, so there is
no look-ahead. Costs are charged on turnover. The module is numpy-only so it runs anywhere
the core package runs; price loading lives in ``research/backtests``.

Strategies (names match the report's table):
    buy_hold, equal_weight, trend_200, dual_momentum, sector_rotation, donchian,
    trend_filtered_sectors, momentum_trend_top4, ensemble_v3
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant_os_core.right_fit.multiple_testing import deflated_sharpe_ratio
from quant_os_core.right_fit.pbo import probability_backtest_overfitting

FloatArray = NDArray[np.float64]
TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class PriceTable:
    """Aligned daily closes: ``prices[t, j]`` is asset ``symbols[j]`` on ``dates[t]``."""

    dates: tuple[str, ...]
    symbols: tuple[str, ...]
    prices: FloatArray

    def __post_init__(self) -> None:
        if self.prices.ndim != 2 or self.prices.shape != (len(self.dates), len(self.symbols)):
            raise ValueError("prices must be shaped (len(dates), len(symbols))")
        if not np.all(np.isfinite(self.prices)) or np.any(self.prices <= 0):
            raise ValueError("prices must be finite and positive (forward-fill gaps first)")
        if list(self.dates) != sorted(self.dates) or len(set(self.dates)) != len(self.dates):
            raise ValueError("dates must be unique and increasing")

    def col(self, symbol: str) -> int:
        return self.symbols.index(symbol)

    def month_end_mask(self) -> NDArray[np.bool_]:
        months = [d[:7] for d in self.dates]
        return np.array(
            [i == len(months) - 1 or months[i] != months[i + 1] for i in range(len(months))]
        )


def _sma(series: FloatArray, window: int) -> FloatArray:
    out = np.full(series.shape, np.nan)
    if series.size >= window:
        csum = np.cumsum(np.insert(series, 0, 0.0))
        out[window - 1 :] = (csum[window:] - csum[:-window]) / window
    return out


def _empty(table: PriceTable) -> FloatArray:
    return np.zeros(table.prices.shape)


def buy_hold(table: PriceTable, symbol: str = "SPY") -> FloatArray:
    w = _empty(table)
    w[:, table.col(symbol)] = 1.0
    return w


def equal_weight(table: PriceTable, universe: Sequence[str]) -> FloatArray:
    w = _empty(table)
    month_end = table.month_end_mask()
    current = np.zeros(len(table.symbols))
    for i in range(len(table.dates)):
        if i == 0 or month_end[i - 1]:
            current = np.zeros(len(table.symbols))
            for s in universe:
                current[table.col(s)] = 1.0 / len(universe)
        w[i] = current
    return w


def trend_200(table: PriceTable, symbol: str = "SPY", band: float = 0.01) -> FloatArray:
    j = table.col(symbol)
    px = table.prices[:, j]
    ma = _sma(px, 200)
    w = _empty(table)
    on = 1.0
    for i in range(len(px)):
        if np.isfinite(ma[i]):
            if px[i] > ma[i] * (1 + band):
                on = 1.0
            elif px[i] < ma[i] * (1 - band):
                on = 0.0
        w[i, j] = on
    return w


def dual_momentum(table: PriceTable, risk: str = "SPY", hedge: str = "GLD") -> FloatArray:
    a, b = table.col(risk), table.col(hedge)
    month_end = table.month_end_mask()
    w = _empty(table)
    current = np.zeros(len(table.symbols))
    for i in range(len(table.dates)):
        if month_end[i] and i >= TRADING_DAYS:
            ra = table.prices[i, a] / table.prices[i - TRADING_DAYS, a] - 1
            rb = table.prices[i, b] / table.prices[i - TRADING_DAYS, b] - 1
            current = np.zeros(len(table.symbols))
            if max(ra, rb) > 0:
                current[a if ra > rb else b] = 1.0
        w[i] = current
    return w


def sector_rotation(table: PriceTable, sectors: Sequence[str], top: int = 3) -> FloatArray:
    month_end = table.month_end_mask()
    cols = [table.col(s) for s in sectors]
    ma = {c: _sma(table.prices[:, c], 200) for c in cols}
    w = _empty(table)
    current = np.zeros(len(table.symbols))
    for i in range(len(table.dates)):
        if month_end[i] and i >= TRADING_DAYS + 21:
            scores = sorted(
                (
                    (table.prices[i - 21, c] / table.prices[i - TRADING_DAYS, c] - 1, c)
                    for c in cols
                ),
                reverse=True,
            )[:top]
            current = np.zeros(len(table.symbols))
            for _, c in scores:
                if table.prices[i, c] > ma[c][i]:
                    current[c] = 1.0 / top
        w[i] = current
    return w


def donchian(
    table: PriceTable, universe: Sequence[str], entry: int = 55, exit_: int = 20
) -> FloatArray:
    w = _empty(table)
    state = dict.fromkeys(universe, False)
    for i in range(entry + 1, len(table.dates)):
        for s in universe:
            px = table.prices[: i + 1, table.col(s)]
            if not state[s] and px[i] > px[i - entry : i].max():
                state[s] = True
            elif state[s] and px[i] < px[i - exit_ : i].min():
                state[s] = False
            if state[s]:
                w[i, table.col(s)] = 1.0 / len(universe)
    return w


def trend_filtered(table: PriceTable, universe: Sequence[str]) -> FloatArray:
    month_end = table.month_end_mask()
    ma = {s: _sma(table.prices[:, table.col(s)], 200) for s in universe}
    w = _empty(table)
    current = np.zeros(len(table.symbols))
    for i in range(len(table.dates)):
        if month_end[i] and i >= 200:
            held = [s for s in universe if table.prices[i, table.col(s)] > ma[s][i]]
            current = np.zeros(len(table.symbols))
            for s in held:
                current[table.col(s)] = 1.0 / len(held)
        w[i] = current
    return w


def momentum_trend_top(table: PriceTable, universe: Sequence[str], top: int = 4) -> FloatArray:
    month_end = table.month_end_mask()
    ma = {s: _sma(table.prices[:, table.col(s)], 200) for s in universe}
    w = _empty(table)
    current = np.zeros(len(table.symbols))
    for i in range(len(table.dates)):
        if month_end[i] and i >= TRADING_DAYS + 21:
            scored = []
            for s in universe:
                c = table.col(s)
                if table.prices[i, c] <= ma[s][i]:
                    continue
                m12 = table.prices[i - 21, c] / table.prices[i - TRADING_DAYS, c] - 1
                m6 = table.prices[i - 21, c] / table.prices[i - 126, c] - 1
                scored.append((0.5 * m12 + 0.5 * m6, c))
            picks = sorted(scored, reverse=True)[:top]
            current = np.zeros(len(table.symbols))
            for _, c in picks:
                current[c] = 1.0 / len(picks)
        w[i] = current
    return w


def renormalized_mix(sleeves: Sequence[FloatArray]) -> FloatArray:
    """Average sleeves, then scale so capital from sleeves in cash goes to active ones."""

    mix = np.mean(np.stack(sleeves), axis=0)
    totals = mix.sum(axis=1, keepdims=True)
    return np.divide(mix, totals, out=np.zeros_like(mix), where=totals > 1e-12)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    name: str
    returns: FloatArray
    equity: FloatArray
    turnover_per_year: float

    @property
    def years(self) -> float:
        return self.returns.size / TRADING_DAYS

    @property
    def cagr(self) -> float:
        return float(self.equity[-1] ** (1 / self.years) - 1)

    @property
    def volatility(self) -> float:
        return float(np.std(self.returns, ddof=1) * np.sqrt(TRADING_DAYS))

    @property
    def sharpe(self) -> float:
        sd = np.std(self.returns, ddof=1)
        return 0.0 if sd == 0 else float(np.mean(self.returns) / sd * np.sqrt(TRADING_DAYS))

    @property
    def max_drawdown(self) -> float:
        peak = np.maximum.accumulate(np.concatenate([[1.0], self.equity]))
        return float(np.min(np.concatenate([[1.0], self.equity]) / peak - 1))

    def value_of(self, start: float = 50.0) -> float:
        return float(start * self.equity[-1])


def run_weights(
    table: PriceTable,
    weights: FloatArray,
    *,
    name: str,
    cost_per_side: float = 0.001,
    start: int = 0,
) -> BacktestResult:
    if weights.shape != table.prices.shape:
        raise ValueError("weights must match the price table shape")
    if np.any(weights < -1e-12) or np.any(weights.sum(axis=1) > 1 + 1e-9):
        raise ValueError("weights must be long-only and unlevered (no shorting, no leverage)")
    asset_returns = table.prices[1:] / table.prices[:-1] - 1
    w = weights[start:-1]
    prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    turnover = np.abs(w - prev).sum(axis=1)
    rets = (w * asset_returns[start:]).sum(axis=1) - turnover * cost_per_side
    years = rets.size / TRADING_DAYS
    return BacktestResult(name, rets, np.cumprod(1 + rets), float(turnover.sum() / years))


@dataclass(frozen=True, slots=True)
class EnsembleUniverse:
    benchmark: str = "SPY"
    growth: str = "QQQ"
    hedge: str = "GLD"
    sectors: tuple[str, ...] = ("XLK", "XLE", "XLV", "XLP", "XLY", "XLI", "XLB", "XLU")


def strategy_book(
    table: PriceTable, universe: EnsembleUniverse | None = None
) -> dict[str, FloatArray]:
    u = universe or EnsembleUniverse()
    sectors = list(u.sectors)
    book: dict[str, FloatArray] = {
        "buy_hold": buy_hold(table, u.benchmark),
        "equal_weight": equal_weight(table, sectors),
        "trend_200": trend_200(table, u.benchmark),
        "dual_momentum": dual_momentum(table, u.benchmark, u.hedge),
        "sector_rotation": sector_rotation(table, sectors),
        "donchian": donchian(table, [u.benchmark, u.growth, u.hedge]),
        "trend_filtered_sectors": trend_filtered(table, [*sectors, u.hedge]),
        "momentum_trend_top4": momentum_trend_top(table, [*sectors, u.hedge, u.growth]),
    }
    book["ensemble_v3"] = renormalized_mix(
        [
            book["dual_momentum"],
            book["sector_rotation"],
            book["donchian"],
            book["momentum_trend_top4"],
        ]
    )
    return book


@dataclass(frozen=True, slots=True)
class EnsembleReport:
    results: Mapping[str, BacktestResult]
    selected: str
    deflated_sharpe_probability: float
    pbo: float | None

    @property
    def passes(self) -> bool:
        """Report gate: DSR probability >= 0.95 and PBO < 0.5 (when computable)."""

        return self.deflated_sharpe_probability >= 0.95 and (self.pbo is None or self.pbo < 0.5)


def evaluate_book(
    table: PriceTable,
    book: Mapping[str, FloatArray],
    *,
    selected: str = "ensemble_v3",
    cost_per_side: float = 0.001,
    warmup: int = TRADING_DAYS + 21,
    extra_trial_sharpes: Sequence[float] = (),
    pbo_slices: int = 8,
    runner: Callable[..., BacktestResult] = run_weights,
) -> EnsembleReport:
    """Backtest every variant, then deflate the selected one by *all* trials run."""

    results = {
        name: runner(table, w, name=name, cost_per_side=cost_per_side, start=warmup)
        for name, w in book.items()
    }
    daily_sharpes = [r.sharpe / np.sqrt(TRADING_DAYS) for r in results.values()]
    trials = np.asarray([*daily_sharpes, *(s / np.sqrt(TRADING_DAYS) for s in extra_trial_sharpes)])
    dsr = deflated_sharpe_ratio(results[selected].returns, trial_sharpes=trials)
    matrix = np.column_stack([r.returns for r in results.values()])
    pbo: float | None
    try:
        pbo = probability_backtest_overfitting(matrix, n_slices=pbo_slices).probability
    except ValueError:
        pbo = None
    return EnsembleReport(results, selected, dsr.probability, pbo)
