"""Reference strategies from well-known open-source bots, tested halal-style and honestly.

    python research/backtests/reference_strategies.py --dir data/ohlc

Long-only spot versions (no leverage, no shorting) of strategies popular in open-source
bots, written from their public descriptions, not copied code:

* golden_cross      50/200 SMA crossover (classic; Jesse and Backtrader examples)
* turtle_20_10      Donchian 20-day breakout in, 10-day low out (Turtle rules)
* supertrend        ATR(10) x3 trailing band (common Freqtrade/TradingView strategy)
* macd_cross        MACD(12,26,9) line above signal (Freqtrade sample strategies)
* bb_rsi_revert     buy close < lower Bollinger(20,2) and RSI(14) < 30, exit at the mid band
                    (Freqtrade BinHV45-style mean reversion)
* sma_offset        buy close < 0.95 x SMA(20), exit close > 1.01 x EMA(20)
                    (Freqtrade SMAOffset family)
* rsi2_connors      buy RSI(2) < 10 above SMA(200), exit close > SMA(5) (Connors)

Every signal trades at the next close with costs (0.25% crypto, 0.10% SPY). The best
variant per asset is picked on data up to ``--split`` only; the rest is a holdout. The
deflated Sharpe counts every strategy run on that asset, and PBO is computed across them.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from quant_os_core.right_fit.multiple_testing import deflated_sharpe_ratio
from quant_os_core.right_fit.pbo import probability_backtest_overfitting

Arr = np.ndarray
WARMUP = 210


def load(path: Path) -> tuple[list[str], Arr, Arr, Arr]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    return [r["time"][:10] for r in rows], col("high"), col("low"), col("close")


# ---------------------------------------------------------------- indicators


def sma(x: Arr, n: int) -> Arr:
    out = np.full(x.size, np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1 :] = (c[n:] - c[:-n]) / n
    return out


def ema(x: Arr, n: int) -> Arr:
    out = np.empty(x.size)
    a = 2 / (n + 1)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rsi(x: Arr, n: int) -> Arr:
    d = np.diff(x, prepend=x[0])
    up, dn = np.maximum(d, 0), np.maximum(-d, 0)
    au, ad = ema(up, 2 * n - 1), ema(dn, 2 * n - 1)  # Wilder smoothing
    return 100 - 100 / (1 + au / np.maximum(ad, 1e-12))


def atr(h: Arr, lo: Arr, c: Arr, n: int) -> Arr:
    prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - lo, np.maximum(abs(h - prev), abs(lo - prev)))
    return ema(tr, 2 * n - 1)


def _state(entry: Arr, exit_: Arr) -> Arr:
    """1 from an entry signal until the next exit signal."""

    w = np.zeros(entry.size)
    on = False
    for i in range(entry.size):
        if on and exit_[i]:
            on = False
        elif not on and entry[i]:
            on = True
        w[i] = float(on)
    return w


# ---------------------------------------------------------------- strategies


def hold(h: Arr, lo: Arr, c: Arr) -> Arr:
    return np.ones(c.size)


def golden_cross(h: Arr, lo: Arr, c: Arr) -> Arr:
    return (sma(c, 50) > sma(c, 200)).astype(float)


def turtle_20_10(h: Arr, lo: Arr, c: Arr) -> Arr:
    hi20 = np.array([h[max(0, i - 20) : i].max() if i else np.inf for i in range(c.size)])
    lo10 = np.array([lo[max(0, i - 10) : i].min() if i else -np.inf for i in range(c.size)])
    return _state(c > hi20, c < lo10)


def supertrend(h: Arr, lo: Arr, c: Arr, n: int = 10, mult: float = 3.0) -> Arr:
    band = atr(h, lo, c, n) * mult
    mid = (h + lo) / 2
    upper, lower = mid + band, mid - band
    w = np.zeros(c.size)
    trend_up, final_up, final_lo = True, upper[0], lower[0]
    for i in range(1, c.size):
        final_up = min(upper[i], final_up) if c[i - 1] <= final_up else upper[i]
        final_lo = max(lower[i], final_lo) if c[i - 1] >= final_lo else lower[i]
        if trend_up and c[i] < final_lo:
            trend_up = False
        elif not trend_up and c[i] > final_up:
            trend_up = True
        w[i] = float(trend_up)
    return w


def macd_cross(h: Arr, lo: Arr, c: Arr) -> Arr:
    line = ema(c, 12) - ema(c, 26)
    return (line > ema(line, 9)).astype(float)


def bb_rsi_revert(h: Arr, lo: Arr, c: Arr) -> Arr:
    mid = sma(c, 20)
    sd = np.array([c[max(0, i - 19) : i + 1].std() for i in range(c.size)])
    return _state((c < mid - 2 * sd) & (rsi(c, 14) < 30), c > mid)


def sma_offset(h: Arr, lo: Arr, c: Arr) -> Arr:
    return _state(c < 0.95 * sma(c, 20), c > 1.01 * ema(c, 20))


def rsi2_connors(h: Arr, lo: Arr, c: Arr) -> Arr:
    return _state((rsi(c, 2) < 10) & (c > sma(c, 200)), c > sma(c, 5))


BOOK: dict[str, Callable[[Arr, Arr, Arr], Arr]] = {
    "hold": hold,
    "golden_cross": golden_cross,
    "turtle_20_10": turtle_20_10,
    "supertrend": supertrend,
    "macd_cross": macd_cross,
    "bb_rsi_revert": bb_rsi_revert,
    "sma_offset": sma_offset,
    "rsi2_connors": rsi2_connors,
}


# ---------------------------------------------------------------- evaluation


def returns(c: Arr, w: Arr, cost: float) -> Arr:
    w = np.nan_to_num(w)
    held = np.concatenate([[0.0], w[:-1]])[:-1]  # decided at close t, held over t+1
    turnover = np.abs(np.diff(np.concatenate([[0.0], held])))
    return (held * (c[1:] / c[:-1] - 1) - turnover * cost)[WARMUP:]


def stats(r: Arr, periods: int) -> tuple[float, float, float]:
    curve = np.cumprod(1 + r)
    cagr = float(curve[-1] ** (periods / r.size) - 1)
    sd = float(r.std(ddof=1))
    sharpe = float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0
    return cagr, sharpe, float((curve / np.maximum.accumulate(curve) - 1).min())


def evaluate(path: Path, cost: float, periods: int, split: str) -> bool:
    dates, h, lo, c = load(path)
    results = {name: returns(c, fn(h, lo, c), cost) for name, fn in BOOK.items()}
    d = dates[WARMUP + 1 :]
    cut = next(i for i, x in enumerate(d) if x > split)
    print(f"== {path.stem}  {d[0]} to {d[-1]}, holdout after {split}, cost {cost:.2%}")
    print(
        f"{'strategy':15s}{'IS Sharpe':>10s}{'OOS CAGR':>10s}{'OOS Sharpe':>11s}"
        f"{'OOS MaxDD':>11s}{'in mkt':>8s}"
    )
    for name, r in results.items():
        _, is_sh, _ = stats(r[:cut], periods)
        o_cagr, o_sh, o_dd = stats(r[cut:], periods)
        exposure = float(np.nan_to_num(BOOK[name](h, lo, c))[WARMUP:].mean())
        print(f"{name:15s}{is_sh:10.2f}{o_cagr:10.1%}{o_sh:11.2f}{o_dd:11.1%}{exposure:8.0%}")
    pick = max(
        (n for n in results if n != "hold"), key=lambda n: stats(results[n][:cut], periods)[1]
    )
    trials = np.array([stats(r, periods)[1] for r in results.values()]) / np.sqrt(periods)
    dsr = deflated_sharpe_ratio(results[pick], trial_sharpes=trials).probability
    pbo = probability_backtest_overfitting(np.column_stack(list(results.values()))).probability
    beats = stats(results[pick][cut:], periods)[1] > stats(results["hold"][cut:], periods)[1]
    ok = dsr >= 0.95 and pbo < 0.5 and beats
    print(
        f"picked in-sample: {pick}; DSR {dsr:.3f}  PBO {pbo:.2f}  "
        f"beats holding out of sample: {beats}  GATE: {'PASS' if ok else 'FAIL'}\n"
    )
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path("data/ohlc"))
    parser.add_argument("--split", default="2021-12-31")
    args = parser.parse_args(argv)
    for sym, cost, periods in (
        ("BTC-USD", 0.0025, 365),
        ("ETH-USD", 0.0025, 365),
        ("SPY", 0.001, 252),
        ("SPUS", 0.001, 252),
    ):
        split = "2022-12-31" if sym == "SPUS" else args.split
        evaluate(args.dir / f"{sym}.csv", cost, periods, split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
