"""What high leverage does on real BTC data, and the halal (spot) version of buying dips.

    python research/backtests/leverage_reality.py --csv data/ohlc/BTC-USD.csv

Part 1 measures, for a long opened at a daily close, how often the next day's low alone
would liquidate it at each leverage (ignoring fees and funding, which make it worse).
Leveraged futures stay blocked in this bot (GUARDRAILS.md: IIFA Resolution 63, AAOIFI 1/21);
this script only shows why they also fail on risk.

Part 2 tests spot "buy the dip" on BTC with no leverage against simply holding BTC, with
next-day execution and 0.25% costs. Variants are counted for an honest deflated Sharpe.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from quant_os_core.right_fit.multiple_testing import deflated_sharpe_ratio

DAYS = 365  # crypto trades every day


def load(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        [r["time"][:10] for r in rows],
        np.array([float(r["low"]) for r in rows]),
        np.array([float(r["close"]) for r in rows]),
    )


def liquidation_table(low: np.ndarray, close: np.ndarray) -> None:
    worst = low[1:] / close[:-1] - 1  # next day's worst move against a long
    print(
        f"{'leverage':>9s}{'liquidated if drop >':>22s}{'days that do it':>17s}"
        f"{'P(survive 30 days)':>20s}"
    )
    for lev in (2, 5, 10, 20, 50, 100, 200):
        threshold = -1 / lev
        daily = float((worst <= threshold).mean())
        survive_30 = (1 - daily) ** 30
        print(f"{lev:>8d}x{threshold:>22.1%}{daily:>17.1%}{survive_30:>20.1%}")


def _stats(rets: np.ndarray) -> tuple[float, float, float]:
    curve = np.cumprod(1 + rets)
    cagr = float(curve[-1] ** (DAYS / rets.size) - 1)
    sd = float(rets.std(ddof=1))
    sharpe = float(rets.mean() / sd * np.sqrt(DAYS)) if sd > 0 else 0.0
    return cagr, sharpe, float((curve / np.maximum.accumulate(curve) - 1).min())


def dip_weights(close: np.ndarray, dip: float, lookback: int, hold: int) -> np.ndarray:
    """Buy when price is ``dip`` below its ``lookback``-day high; hold ``hold`` days."""

    w = np.zeros(close.size)
    left = 0
    for t in range(lookback, close.size):
        high = close[t - lookback : t + 1].max()
        if left == 0 and close[t] <= high * (1 - dip):
            left = hold
        if left > 0:
            w[t] = 1.0
            left -= 1
    return w


def run(close: np.ndarray, weights: np.ndarray, cost: float) -> np.ndarray:
    rets = close[1:] / close[:-1] - 1
    w = np.concatenate([[0.0], weights[:-1]])[:-1]  # decided at close t, held from t+1
    turnover = np.abs(np.diff(np.concatenate([[0.0], w])))
    return w * rets - turnover * cost


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=Path("data/ohlc/BTC-USD.csv"))
    parser.add_argument("--cost", type=float, default=0.0025)
    args = parser.parse_args(argv)
    dates, low, close = load(args.csv)
    print(f"BTC {dates[0]} to {dates[-1]}\n\n== 1. Leverage (blocked here; shown for risk)")
    liquidation_table(low, close)

    print("\n== 2. Spot buy-the-dip, no leverage")
    start = 365
    book = {"hold_btc": np.ones(close.size)}
    for dip in (0.2, 0.3, 0.5):
        for hold in (90, 180, 365):
            book[f"dip{int(dip * 100)}_hold{hold}"] = dip_weights(close, dip, 90, hold)
    results = {n: run(close, w, args.cost)[start:] for n, w in book.items()}
    print(f"{'strategy':18s}{'CAGR':>8s}{'Sharpe':>8s}{'MaxDD':>9s}{'in market':>11s}")
    for name, rets in results.items():
        cagr, sharpe, dd = _stats(rets)
        exposure = float(book[name][start:-1].mean())
        print(f"{name:18s}{cagr:8.1%}{sharpe:8.2f}{dd:9.1%}{exposure:11.0%}")
    best = max((n for n in results if n != "hold_btc"), key=lambda n: _stats(results[n])[1])
    trials = np.array([_stats(r)[1] for r in results.values()]) / np.sqrt(DAYS)
    dsr = deflated_sharpe_ratio(results[best], trial_sharpes=trials).probability
    beats = _stats(results[best])[1] > _stats(results["hold_btc"])[1]
    print(
        f"\nbest dip variant: {best}; deflated Sharpe {dsr:.3f} (>= 0.95); "
        f"beats holding on Sharpe: {beats}"
    )
    print("GATE:", "PASS" if dsr >= 0.95 and beats else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
