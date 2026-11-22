"""Hunt for a real, halal edge: long-only, unlevered, spot only, honest statistics.

    python research/backtests/edge_hunt.py --dir data/ohlc

Reads per-symbol CSVs (``time,high,low,close``) and runs two books:

* ``etf``: 11 ETFs from 2004 (the same universe as ``run_halal_v3.py``).
* ``crypto``: SPY, GLD and BTC-USD from 2014 (spot crypto under the owner profile).

Discipline, so a pass means something:

1. Every signal is computed at the close of day t and traded at the close of day t+1.
2. Costs are charged on turnover (default 0.10% per side for ETFs, 0.25% for crypto).
3. The winner is picked on data up to ``--split`` only; the years after are a holdout
   that played no part in choosing it.
4. The full-period deflated Sharpe counts every variant in the book plus every variant
   tried before (``PRIOR_TRIAL_SHARPES``), and PBO is computed across the whole book.

A book passes only if DSR >= 0.95, PBO < 0.5, and the pick beats equal weight on
the holdout Sharpe.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from quant_os_core.right_fit.multiple_testing import deflated_sharpe_ratio
from quant_os_core.right_fit.pbo import probability_backtest_overfitting
from quant_os_core.strategies.halal_ensemble import (
    TRADING_DAYS,
    BacktestResult,
    PriceTable,
    run_weights,
)

FloatArray = NDArray[np.float64]
Strategy = Callable[[PriceTable], FloatArray]

ETF = ("SPY", "QQQ", "GLD", "XLK", "XLE", "XLV", "XLP", "XLY", "XLI", "XLB", "XLU")
CRYPTO = ("SPY", "GLD", "BTC-USD")
# annual Sharpes of every variant already tried in run_halal_v3 (its 9 + 5 extra)
PRIOR_TRIAL_SHARPES: tuple[float, ...] = (0.61, 0.67, 0.67, 0.58, 0.60, 0.71, 0.64, 0.52, 0.64)
PRIOR_TRIAL_SHARPES += (0.14, 0.70, 0.72, 0.38, 0.54)
WARMUP = TRADING_DAYS + 21


# ---------------------------------------------------------------- data


def load_closes(directory: Path, symbols: Sequence[str]) -> PriceTable:
    """Inner-join closes on dates where every symbol has a bar."""

    series: list[dict[str, float]] = []
    for symbol in symbols:
        with (directory / f"{symbol}.csv").open(newline="") as handle:
            rows = csv.DictReader(handle)
            series.append({r["time"][:10]: float(r["close"]) for r in rows})
    dates = sorted(set.intersection(*(set(s) for s in series)))
    prices = np.array([[s[d] for s in series] for d in dates], dtype=np.float64)
    return PriceTable(tuple(dates), tuple(symbols), prices)


# ---------------------------------------------------------------- helpers


def _sma(x: FloatArray, n: int) -> FloatArray:
    out = np.full(x.shape, np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0, axis=0), axis=0)
    out[n - 1 :] = (c[n:] - c[:-n]) / n
    return out


def _vol(table: PriceTable, n: int) -> FloatArray:
    rets = np.vstack([np.zeros((1, table.prices.shape[1])), np.diff(np.log(table.prices), axis=0)])
    mean = _sma(rets, n)
    sq = _sma(rets**2, n)
    return np.sqrt(np.maximum(sq - mean**2, 1e-12) * TRADING_DAYS)


def _monthly(table: PriceTable, target: Callable[[int], FloatArray]) -> FloatArray:
    """Rebalance to ``target(i)`` at each month end; hold in between."""

    w = np.zeros(table.prices.shape)
    current = np.zeros(table.prices.shape[1])
    month_end = table.month_end_mask()
    for i in range(len(table.dates)):
        if month_end[i] and i >= WARMUP:
            current = target(i)
        w[i] = current
    return w


def lag(weights: FloatArray, days: int = 1) -> FloatArray:
    """Trade at the next close: weights decided on day t apply from day t+days."""

    out = np.zeros_like(weights)
    out[days:] = weights[:-days]
    return out


def _cap(w: FloatArray) -> FloatArray:
    total = w.sum()
    return w / total if total > 1 else w


# ---------------------------------------------------------------- strategies


def equal_weight(t: PriceTable) -> FloatArray:
    n = t.prices.shape[1]
    return _monthly(t, lambda i: np.full(n, 1.0 / n))


def inverse_vol(t: PriceTable, window: int = 63) -> FloatArray:
    vol = _vol(t, window)
    return _monthly(t, lambda i: (1 / vol[i]) / (1 / vol[i]).sum())


def inverse_vol_trend(t: PriceTable, window: int = 63, ma: int = 200) -> FloatArray:
    """Risk parity, but an asset below its moving average sits in cash."""

    vol, sma = _vol(t, window), _sma(t.prices, ma)

    def target(i: int) -> FloatArray:
        inv = 1 / vol[i]
        parity = inv / inv.sum()
        return np.where(t.prices[i] > sma[i], parity, 0.0)

    return _monthly(t, target)


def momentum_top(t: PriceTable, top: int = 3, lookback: int = 252, skip: int = 21) -> FloatArray:
    """12-1 cross-sectional momentum, top N, each only if its own momentum is positive."""

    def target(i: int) -> FloatArray:
        mom = t.prices[i - skip] / t.prices[i - lookback] - 1
        w = np.zeros(mom.size)
        for c in np.argsort(mom)[::-1][:top]:
            if mom[c] > 0:
                w[c] = 1.0 / top
        return w

    return _monthly(t, target)


def vol_target_trend(t: PriceTable, symbol: str, target_vol: float = 0.15) -> FloatArray:
    """Daily: hold ``symbol`` above its 100-day average, sized to a volatility target."""

    c = t.col(symbol)
    sma, vol = _sma(t.prices[:, c], 100), _vol(t, 21)[:, c]
    w = np.zeros(t.prices.shape)
    on = t.prices[:, c] > sma
    w[:, c] = np.where(on, np.minimum(1.0, target_vol / vol), 0.0)
    # only trade when the size moves by more than 10 points, to keep costs down
    for i in range(1, len(w)):
        if abs(w[i, c] - w[i - 1, c]) < 0.10 and on[i] == on[i - 1]:
            w[i, c] = w[i - 1, c]
    w[:WARMUP] = 0
    return w


def trend_each_vol_target(t: PriceTable, target_vol: float = 0.12) -> FloatArray:
    """Each asset: in when above its 100-day average; book sized to a vol target, unlevered."""

    sma, vol = _sma(t.prices, 100), _vol(t, 63)

    def target(i: int) -> FloatArray:
        on = t.prices[i] > sma[i]
        if not on.any():
            return np.zeros(on.size)
        inv = np.where(on, 1 / vol[i], 0.0)
        parity = inv / inv.sum()
        book_vol = float(np.sqrt((parity**2 * vol[i] ** 2).sum()))  # zero-correlation proxy
        return _cap(parity * min(1.0, target_vol / book_vol))

    return _monthly(t, target)


def low_vol_top(t: PriceTable, top: int = 5) -> FloatArray:
    vol = _vol(t, 252)

    def target(i: int) -> FloatArray:
        w = np.zeros(t.prices.shape[1])
        w[np.argsort(vol[i])[:top]] = 1.0 / top
        return w

    return _monthly(t, target)


def btc_trend(t: PriceTable, ma: int = 50) -> FloatArray:
    """BTC above its moving average, else SPY/GLD half-half if they trend, else cash."""

    b, s, g = t.col("BTC-USD"), t.col("SPY"), t.col("GLD")
    sma = _sma(t.prices, ma)
    sma200 = _sma(t.prices, 200)
    w = np.zeros(t.prices.shape)
    for i in range(WARMUP, len(w)):
        if t.prices[i, b] > sma[i, b]:
            w[i, b] = 1.0
        else:
            for c in (s, g):
                if t.prices[i, c] > sma200[i, c]:
                    w[i, c] = 0.5
    return w


def btc_sleeve(t: PriceTable, share: float = 0.2) -> FloatArray:
    """A fixed crypto sleeve: ``share`` in BTC when it trends, the rest in inverse-vol SPY/GLD."""

    base = inverse_vol_trend(t)
    b = t.col("BTC-USD")
    sma = _sma(t.prices[:, b], 50)
    w = base * (1 - share)
    w[:, b] = np.where(t.prices[:, b] > sma, share, 0.0)
    w[:WARMUP] = 0
    return w


ETF_BOOK: Mapping[str, Strategy] = {
    "equal_weight": equal_weight,
    "inverse_vol": inverse_vol,
    "inverse_vol_trend": inverse_vol_trend,
    "momentum_top3": momentum_top,
    "spy_vol_target_trend": lambda t: vol_target_trend(t, "SPY"),
    "trend_each_vol_target": trend_each_vol_target,
    "low_vol_top5": low_vol_top,
}
CRYPTO_BOOK: Mapping[str, Strategy] = {
    "equal_weight": equal_weight,
    "inverse_vol_trend": inverse_vol_trend,
    "btc_trend_50": btc_trend,
    "btc_trend_100": lambda t: btc_trend(t, 100),
    "btc_vol_target_trend": lambda t: vol_target_trend(t, "BTC-USD", 0.40),
    "btc_sleeve_20": btc_sleeve,
}


# ---------------------------------------------------------------- evaluation


def _sharpe(rets: FloatArray) -> float:
    sd = float(rets.std(ddof=1))
    return float(rets.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def _stats(rets: FloatArray) -> tuple[float, float, float]:
    curve = np.cumprod(1 + rets)
    years = rets.size / TRADING_DAYS
    cagr = float(curve[-1] ** (1 / years) - 1)
    dd = float((curve / np.maximum.accumulate(curve) - 1).min())
    return cagr, _sharpe(rets), dd


def evaluate(table: PriceTable, book: Mapping[str, Strategy], *, split: str, cost: float) -> bool:
    results: dict[str, BacktestResult] = {
        name: run_weights(table, lag(fn(table)), name=name, cost_per_side=cost, start=WARMUP)
        for name, fn in book.items()
    }
    # returns index k is the move from dates[WARMUP + k] to dates[WARMUP + k + 1]
    cut = next(i for i, d in enumerate(table.dates[WARMUP + 1 :]) if d > split)
    print(f"{table.dates[WARMUP]} to {table.dates[-1]}, holdout after {split}, cost {cost:.2%}")
    print(
        f"{'strategy':24s}{'IS Sharpe':>10s}{'OOS CAGR':>10s}{'OOS Sharpe':>11s}"
        f"{'OOS MaxDD':>11s}{'full CAGR':>10s}{'$50 ->':>9s}"
    )
    for name, r in results.items():
        cagr, _, _ = _stats(r.returns)
        o_cagr, o_sh, o_dd = _stats(r.returns[cut:])
        print(
            f"{name:24s}{_sharpe(r.returns[:cut]):10.2f}{o_cagr:10.1%}{o_sh:11.2f}"
            f"{o_dd:11.1%}{cagr:10.1%}{r.value_of():9.0f}"
        )

    pick = max(results, key=lambda n: _sharpe(results[n].returns[:cut]))
    trials = np.array([r.sharpe for r in results.values()] + list(PRIOR_TRIAL_SHARPES)) / np.sqrt(
        TRADING_DAYS
    )
    dsr = deflated_sharpe_ratio(results[pick].returns, trial_sharpes=trials).probability
    matrix = np.column_stack([r.returns for r in results.values()])
    pbo = probability_backtest_overfitting(matrix, n_slices=8).probability
    beats = _sharpe(results[pick].returns[cut:]) > _sharpe(results["equal_weight"].returns[cut:])
    passed = dsr >= 0.95 and pbo < 0.5 and beats
    print(f"\npicked on in-sample only: {pick}")
    print(
        f"DSR {dsr:.3f} (>= 0.95)  PBO {pbo:.2f} (< 0.5)  beats equal weight out of sample: {beats}"
    )
    print("GATE:", "PASS" if passed else "FAIL", "\n")
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path("data/ohlc"))
    parser.add_argument("--split", default="2018-12-31")
    parser.add_argument("--cost", type=float, default=0.001)
    parser.add_argument("--crypto-cost", type=float, default=0.0025)
    args = parser.parse_args(argv)
    print("== ETF book")
    etf = evaluate(load_closes(args.dir, ETF), ETF_BOOK, split=args.split, cost=args.cost)
    print("== crypto book")
    crypto = evaluate(
        load_closes(args.dir, CRYPTO), CRYPTO_BOOK, split=args.split, cost=args.crypto_cost
    )
    return 0 if (etf or crypto) else 1


if __name__ == "__main__":
    sys.exit(main())
