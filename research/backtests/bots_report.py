"""Spot grid, DCA and combo bots against buy-and-hold, tested honestly.

    python research/backtests/bots_report.py --dir data/ohlc

Every bot is long-only, unlevered and spot only (``quant_os_core.strategies.bots``). Grid and
DCA limit orders are filled bar by bar from ``high``/``low`` only when the bar trades through
the level, at most once per level per bar, never bought and sold in the same bar; range exits
sell at the next close with slippage. Fees are paid on every fill (0.25% crypto, 0.10% ETFs).

Discipline, same as the other reports:

1. A small parameter grid per bot family; every variant counts as a trial.
2. The variant is picked on its in-sample Sharpe up to ``--split`` (SPUS: 2022-12-31) only;
   the years after are a holdout that played no part in the choice.
3. The deflated Sharpe of the pick counts every variant run on that asset, and PBO is
   computed across all of them.

GATE PASS needs DSR >= 0.95, PBO < 0.5 and the pick beating buy-and-hold on holdout Sharpe.
A final section runs the arbitrage detectors on deterministic sample books to show how many
apparent gaps survive fees, transfer fees and latency slippage. Simulation only.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import numpy as np

from quant_os_core.right_fit.multiple_testing import deflated_sharpe_ratio
from quant_os_core.right_fit.pbo import probability_backtest_overfitting
from quant_os_core.strategies.bots import (
    ArbCosts,
    Bar,
    BotResult,
    ComboConfig,
    DcaConfig,
    bars_from_arrays,
    buy_and_hold,
    detect_cross_exchange,
    detect_triangular,
    run_combo,
    run_dca,
    run_rolling_grid,
    survival,
)
from quant_os_core.strategies.bots.arbitrage import (
    synthetic_cross_venue_quotes,
    synthetic_triangle_quotes,
)

CAPITAL = Decimal(10_000)
Runner = Callable[[list[Bar]], BotResult]


def load_bars(path: Path) -> list[Bar]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return bars_from_arrays(
        [r["time"][:10] for r in rows],
        [float(r["high"]) for r in rows],
        [float(r["low"]) for r in rows],
        [float(r["close"]) for r in rows],
    )


def book(fee: Decimal, week: int, month: int) -> dict[str, Runner]:
    """The trial set: buy-and-hold plus 6 grid, 4 DCA and 2 combo variants."""

    slip = Decimal("0.001")
    runners: dict[str, Runner] = {"hold": lambda b: buy_and_hold(b, CAPITAL, fee)}
    for width in ("0.10", "0.20", "0.30"):
        for levels in (6, 12):
            runners[f"grid_w{width}_n{levels}"] = lambda b, w=Decimal(width), n=levels: (
                run_rolling_grid(b, CAPITAL, width=w, levels=n, fee_rate=fee, slippage=slip)
            )

    def dca_config(interval: int, tp: Decimal | None) -> DcaConfig:
        return DcaConfig(
            CAPITAL,
            CAPITAL * Decimal("0.02"),
            interval,
            safety_orders=3,
            safety_step=Decimal("0.08"),
            take_profit=tp,
            fee_rate=fee,
        )

    for label, interval in (("wk", week), ("mo", month)):
        for tp in (None, Decimal("0.20")):
            cfg = dca_config(interval, tp)
            runners[f"dca_{label}_tp{tp or 'none'}"] = lambda b, c=cfg: run_dca(b, c)
    for width in ("0.15", "0.30"):
        combo = ComboConfig(
            dca_config(week, None),
            grid_fraction=Decimal("0.3"),
            grid_width=Decimal(width),
            grid_levels=8,
            slippage=slip,
        )
        runners[f"combo_w{width}"] = lambda b, c=combo: run_combo(b, c)
    return runners


def stats(r: np.ndarray, periods: int) -> tuple[float, float, float]:
    curve = np.cumprod(1 + r)
    cagr = float(curve[-1] ** (periods / r.size) - 1)
    sd = float(r.std(ddof=1))
    sharpe = float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0
    return cagr, sharpe, float((curve / np.maximum.accumulate(curve) - 1).min())


def evaluate(path: Path, fee: Decimal, periods: int, split: str) -> bool:
    bars = load_bars(path)
    week, month = (7, 30) if periods == 365 else (5, 21)
    results = {name: run(bars) for name, run in book(fee, week, month).items()}
    rets = {name: res.returns() for name, res in results.items()}
    dates = [b.time for b in bars]
    cut = next(i for i, d in enumerate(dates) if d > split)
    print(f"== {path.stem}  {dates[0]} to {dates[-1]}, holdout after {split}, fee {fee:.2%}")
    print(
        f"{'variant':18s}{'IS Sharpe':>10s}{'OOS CAGR':>10s}{'OOS Sharpe':>11s}"
        f"{'OOS MaxDD':>11s}{'fills':>7s}{'fees%':>7s}"
    )
    for name, r in rets.items():
        _, is_sh, _ = stats(r[:cut], periods)
        o_cagr, o_sh, o_dd = stats(r[cut:], periods)
        res = results[name]
        fee_share = float(res.fees_paid / CAPITAL)
        print(
            f"{name:18s}{is_sh:10.2f}{o_cagr:10.1%}{o_sh:11.2f}{o_dd:11.1%}"
            f"{len(res.fills):7d}{fee_share:7.1%}"
        )
    pick = max((n for n in rets if n != "hold"), key=lambda n: stats(rets[n][:cut], periods)[1])
    trials = np.array([stats(r, periods)[1] for r in rets.values()]) / np.sqrt(periods)
    dsr = deflated_sharpe_ratio(rets[pick], trial_sharpes=trials).probability
    pbo = probability_backtest_overfitting(np.column_stack(list(rets.values()))).probability
    beats = stats(rets[pick][cut:], periods)[1] > stats(rets["hold"][cut:], periods)[1]
    ok = dsr >= 0.95 and pbo < 0.5 and beats
    print(
        f"picked in-sample: {pick}; trials {len(rets)}; DSR {dsr:.3f}  PBO {pbo:.2f}  "
        f"beats holding out of sample: {beats}  GATE: {'PASS' if ok else 'FAIL'}\n"
    )
    return ok


def arbitrage_section() -> None:
    print("== Arbitrage detectors on SAMPLE books (synthetic, not market data; no orders)")
    cross_costs = ArbCosts(Decimal("0.001"), Decimal(5), Decimal("0.0002"))
    cross = [
        opp
        for books in synthetic_cross_venue_quotes(500)
        for opp in detect_cross_exchange(books, Decimal("0.5"), cross_costs)
    ]
    s = survival(cross)
    print(
        f"cross-exchange BTC/USDT, 3 venues, 500 snapshots: {s.apparent} apparent gaps, "
        f"{s.survived} pay after 0.10% taker x2 + 0.0002 BTC transfer + 5 bps slippage x2 "
        f"({s.vanished_share:.0%} vanish)"
    )
    tri = [
        opp
        for books in synthetic_triangle_quotes(500)
        for opp in detect_triangular(books, "USDT", Decimal(10_000), ArbCosts())
    ]
    t = survival(tri)
    print(
        f"triangular USDT-BTC-ETH, 500 snapshots: {t.apparent} apparent cycles, "
        f"{t.survived} pay after 0.10% taker x3 + 5 bps slippage x3 ({t.vanished_share:.0%} vanish)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path("data/ohlc"))
    parser.add_argument("--split", default="2021-12-31")
    args = parser.parse_args(argv)
    for sym, fee, periods in (
        ("BTC-USD", Decimal("0.0025"), 365),
        ("ETH-USD", Decimal("0.0025"), 365),
        ("SPY", Decimal("0.001"), 252),
        ("SPUS", Decimal("0.001"), 252),
    ):
        path = args.dir / f"{sym}.csv"
        if not path.exists():
            print(f"== {sym}: data file {path} missing, skipped\n")
            continue
        split = "2022-12-31" if sym == "SPUS" else args.split
        evaluate(path, fee, periods, split)
    arbitrage_section()
    return 0


if __name__ == "__main__":
    sys.exit(main())
