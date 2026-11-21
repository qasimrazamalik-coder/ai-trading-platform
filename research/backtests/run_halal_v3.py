"""Reproduce the halal ensemble backtest (research report section 9H) from the repo.

Step 1, on a machine with internet (your laptop):

    pip install yfinance
    python research/backtests/run_halal_v3.py fetch --out data/halal_prices.csv

Step 2, anywhere:

    python research/backtests/run_halal_v3.py run --csv data/halal_prices.csv

``run`` prints CAGR, Sharpe, worst drawdown and $50 value for every variant, then the
Deflated Sharpe probability and PBO for the selected ensemble. The result passes only if
DSR >= 0.95 and PBO < 0.5.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from quant_os_core.strategies.halal_ensemble import (
    EnsembleUniverse,
    PriceTable,
    evaluate_book,
    strategy_book,
)

SYMBOLS = ("SPY", "QQQ", "GLD", "XLK", "XLE", "XLV", "XLP", "XLY", "XLI", "XLB", "XLU")
START = "2004-01-01"
# variants tried in the original browser run that are not in the book (RSI(2), the
# unrenormalized ensemble and two vol-targeted variants); counted so the DSR stays honest
EXTRA_TRIAL_SHARPES = (0.14, 0.70, 0.72, 0.38, 0.54)


def fetch(out: Path) -> None:
    import yfinance as yf  # optional, only needed for downloading

    frame = yf.download(list(SYMBOLS), start=START, auto_adjust=True, progress=False)["Close"]
    frame = frame.dropna(how="any")  # common calendar where every asset has a price
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index_label="date", date_format="%Y-%m-%d")
    print(f"wrote {len(frame)} rows x {len(frame.columns)} assets to {out}")


def load(path: Path) -> PriceTable:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [row for row in reader if row and all(cell.strip() for cell in row)]
    symbols = tuple(header[1:])
    dates = tuple(row[0][:10] for row in rows)
    prices = np.array([[float(cell) for cell in row[1:]] for row in rows], dtype=np.float64)
    return PriceTable(dates, symbols, prices)


def run(path: Path, cost: float) -> int:
    table = load(path)
    report = evaluate_book(
        table,
        strategy_book(table, EnsembleUniverse()),
        cost_per_side=cost,
        extra_trial_sharpes=EXTRA_TRIAL_SHARPES,
    )
    print(f"{table.dates[0]} to {table.dates[-1]}, cost {cost:.2%} per side\n")
    print(f"{'strategy':26s}{'CAGR':>8s}{'Sharpe':>8s}{'MaxDD':>9s}{'$50 ->':>10s}")
    for name, r in report.results.items():
        print(f"{name:26s}{r.cagr:8.1%}{r.sharpe:8.2f}{r.max_drawdown:9.1%}{r.value_of():10.0f}")
    pbo = "n/a" if report.pbo is None else f"{report.pbo:.2f}"
    print(f"\nselected: {report.selected}")
    print(f"deflated Sharpe probability: {report.deflated_sharpe_probability:.3f}  PBO: {pbo}")
    print("GATE:", "PASS" if report.passes else "FAIL - do not trade this live")
    return 0 if report.passes else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", type=Path, default=Path("data/halal_prices.csv"))
    r = sub.add_parser("run")
    r.add_argument("--csv", type=Path, default=Path("data/halal_prices.csv"))
    r.add_argument("--cost", type=float, default=0.001)
    args = parser.parse_args(argv)
    if args.cmd == "fetch":
        fetch(args.out)
        return 0
    return run(args.csv, args.cost)


if __name__ == "__main__":
    sys.exit(main())
