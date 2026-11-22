"""Diversified core report: Shariah ETFs backtest, parameter robustness, outcome ranges.

    python research/backtests/core_report.py --dir data/ohlc

1. Backtests the core on the Shariah ETF list (short history: SPRE starts Dec 2020,
   UMMA Jan 2022) against equal weight, with next-day execution and costs.
2. Robustness: the defensive core's Sharpe and worst drawdown across a grid of volatility
   and trend windows on the 2005-2026 ETF history. This is a sensitivity check, not a
   search: the shipped settings (63, 200) are not re-picked from it.
3. Outcome ranges: a block bootstrap of the 2005-2026 daily returns gives the spread of
   what $50 could become after 1 and 5 years, including the chance of losing money.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_hunt import (
    ETF,
    WARMUP,
    equal_weight,
    inverse_vol,
    inverse_vol_trend,
    lag,
    load_closes,
)
from quant_os_core.strategies.halal_ensemble import (
    TRADING_DAYS,
    PriceTable,
    run_weights,
)

SHARIAH_5 = ("SPUS", "HLAL", "SPRE", "SPSK", "AAAU")
SHARIAH_6 = (*SHARIAH_5, "UMMA")
BOOK = {"equal_weight": equal_weight, "growth": inverse_vol, "defensive": inverse_vol_trend}


def _stats(rets: np.ndarray) -> tuple[float, float, float]:
    curve = np.cumprod(1 + rets)
    cagr = float(curve[-1] ** (TRADING_DAYS / rets.size) - 1)
    sd = float(rets.std(ddof=1))
    sharpe = float(rets.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0
    return cagr, sharpe, float((curve / np.maximum.accumulate(curve) - 1).min())


def _returns(table: PriceTable, weights: np.ndarray, cost: float) -> np.ndarray:
    return run_weights(table, lag(weights), name="x", cost_per_side=cost, start=WARMUP).returns


def backtest(table: PriceTable, cost: float) -> dict[str, np.ndarray]:
    print(f"  {table.dates[WARMUP]} to {table.dates[-1]}, cost {cost:.2%}")
    print(f"  {'strategy':14s}{'CAGR':>8s}{'Sharpe':>8s}{'MaxDD':>9s}{'$50 ->':>9s}")
    out: dict[str, np.ndarray] = {}
    for name, fn in BOOK.items():
        rets = _returns(table, fn(table), cost)
        cagr, sharpe, dd = _stats(rets)
        value = 50 * float(np.prod(1 + rets))
        print(f"  {name:14s}{cagr:8.1%}{sharpe:8.2f}{dd:9.1%}{value:9.0f}")
        out[name] = rets
    return out


def robustness(table: PriceTable, cost: float) -> None:
    print(f"  defensive core, Sharpe / worst drawdown ({table.dates[WARMUP]} on)")
    windows, mas = (21, 63, 126), (100, 150, 200)
    print("  vol \\ trend " + "".join(f"{m:>16d}" for m in mas))
    for w in windows:
        cells = []
        for m in mas:
            _, sharpe, dd = _stats(_returns(table, inverse_vol_trend(table, w, m), cost))
            cells.append(f"{sharpe:7.2f} / {dd:6.1%}")
        print(f"  {w:>11d} " + "".join(f"{c:>16s}" for c in cells))


def outcomes(rets: dict[str, np.ndarray], *, seed: int = 7, paths: int = 20_000) -> None:
    """Stationary block bootstrap (mean block 21 days) of daily returns."""

    rng = np.random.default_rng(seed)
    print(
        f"  {'strategy':14s}{'horizon':>8s}{'worst 5%':>10s}{'median':>9s}"
        f"{'best 5%':>9s}{'P(loss)':>9s}{'P($1M)':>8s}"
    )
    for name, series in rets.items():
        for years in (1, 5):
            n = years * TRADING_DAYS
            idx = np.empty((paths, n), dtype=np.int64)
            idx[:, 0] = rng.integers(0, series.size, paths)
            jump = rng.random((paths, n)) < 1 / 21
            fresh = rng.integers(0, series.size, (paths, n))
            for t in range(1, n):
                idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % series.size)
            final = 50 * np.prod(1 + series[idx], axis=1)
            p5, p50, p95 = np.percentile(final, [5, 50, 95])
            print(
                f"  {name:14s}{years:>7d}y{p5:10.0f}{p50:9.0f}{p95:9.0f}"
                f"{(final < 50).mean():9.1%}{(final >= 1e6).mean():8.1%}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path("data/ohlc"))
    parser.add_argument("--cost", type=float, default=0.001)
    args = parser.parse_args(argv)
    print("== 1. Shariah ETFs (5 funds, without UMMA)")
    backtest(load_closes(args.dir, SHARIAH_5), args.cost)
    print("== 1b. Shariah ETFs (6 funds, with UMMA; shorter)")
    backtest(load_closes(args.dir, SHARIAH_6), args.cost)
    long_table = load_closes(args.dir, ETF)
    print("== 2. Robustness on the 11 original ETFs")
    robustness(long_table, args.cost)
    print("== 3. Outcome ranges for $50 (bootstrap of 2005-2026, 11 original ETFs)")
    long_rets = {name: _returns(long_table, fn(long_table), args.cost) for name, fn in BOOK.items()}
    outcomes(long_rets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
