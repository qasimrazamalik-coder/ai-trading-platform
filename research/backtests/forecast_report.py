"""Deep forecasters vs baselines, fit verdicts, and scenario outcomes for the diversified core.

    python research/backtests/forecast_report.py --dir data/ohlc

1. Walk-forward, out-of-sample comparison of LSTM, TCN and N-BEATS-style models against
   naive forecasts and the classical baselines (AR(1); EWMA and GARCH(1,1)) on SPY, SPUS and
   BTC-USD daily log returns, for next-day return and next-day squared return (variance).
2. A fit verdict (underfit / right fit / overfit) per deep model from fold-averaged training
   and out-of-sample errors against the naive baseline on the same rows.
3. A complexity sweep on SPY variance: three sizes per family, the simplest within one
   standard error of the best wins (1-SE rule).
4. Scenario distributions (NOT predictions) for the diversified core run on SPUS-like halal
   data: SPUS (US equities), SPSK (sukuk) and GLD (gold), monthly rebalancing, 0.10% costs.

Research only. Nothing here sends or sizes an order.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from quant_os_core.forecasting.deep_forecast import (
    Target,
    TrainConfig,
    compare_forecasters,
    complexity_from_runs,
    lstm_spec,
    nbeats_spec,
    plan_folds,
    tcn_spec,
    walk_forward_deep,
)
from quant_os_core.forecasting.scenarios import (
    LABEL,
    block_bootstrap_paths,
    diversified_core_weights,
    garch_paths,
    log_returns_from_prices,
    model_paths,
    run_scenarios,
)
from quant_os_core.strategies.diversified_core import CoreMode

FloatArray = NDArray[np.float64]
ASSETS = ("SPY", "SPUS", "BTC-USD")
SCENARIO_UNIVERSE = ("SPUS", "SPSK", "GLD")


def load_closes(directory: Path, symbols: Sequence[str]) -> tuple[list[str], FloatArray]:
    """Inner-join closes on dates where every symbol has a bar."""

    series = []
    for symbol in symbols:
        with (directory / f"{symbol}.csv").open(newline="") as handle:
            series.append({r["time"][:10]: float(r["close"]) for r in csv.DictReader(handle)})
    dates = sorted(set.intersection(*(set(s) for s in series)))
    return dates, np.array([[s[d] for s in series] for d in dates], dtype=np.float64)


def forecast_section(directory: Path, cfg: TrainConfig, folds: int) -> None:
    for asset in ASSETS:
        dates, closes = load_closes(directory, [asset])
        r = log_returns_from_prices(closes)[:, 0]
        min_train = min(1500, r.size // 2)
        print(
            f"\n=== {asset}: {r.size} daily returns {dates[1]} .. {dates[-1]}, "
            f"first forecast after {min_train} days, {folds} walk-forward folds ==="
        )
        for target in (Target.RETURN, Target.VARIANCE):
            started = time.perf_counter()
            comp = compare_forecasters(
                r, target, asset=asset, config=cfg, folds=folds, min_train=min_train
            )
            print()
            print(comp.table())
            for run in comp.runs:
                epochs = [f.epochs_run for f in run.folds]
                print(
                    f"  fit {run.name:<10} ({run.parameters} params, epochs {epochs}): "
                    f"{run.fit_report().summary()}"
                )
            print(f"  [{time.perf_counter() - started:.0f}s]")


def sweep_section(directory: Path, cfg: TrainConfig, folds: int) -> None:
    _, closes = load_closes(directory, ["SPY"])
    r = log_returns_from_prices(closes)[:, 0]
    plan = plan_folds(r.size, min_train=1500, folds=folds)
    print("\n=== complexity sweep, SPY next-day variance (fold out-of-sample MSE, 1-SE rule) ===")
    families = {
        "LSTM": (lstm_spec(4), lstm_spec(16), lstm_spec(64)),
        "TCN": (tcn_spec(4), tcn_spec(16), tcn_spec(64)),
        "NBEATS": (nbeats_spec(8), nbeats_spec(32), nbeats_spec(128)),
    }
    for family, specs in families.items():
        runs = [walk_forward_deep(r, s, Target.VARIANCE, plan, cfg) for s in specs]
        sweep = complexity_from_runs(runs)
        sizes = ", ".join(
            f"{c.name} ({int(c.complexity)}p) {c.mean_error * 1e8:.3f}+/-"
            f"{c.standard_error * 1e8:.3f}"
            for c in sweep.candidates
        )
        print(
            f"  {family}: {sizes} (x1e-8) -> best {sweep.best.name}, "
            f"1-SE choice {sweep.chosen.name}"
        )


def scenario_section(directory: Path, n_paths: int, horizon: int) -> None:
    dates, prices = load_closes(directory, SCENARIO_UNIVERSE)
    rets = log_returns_from_prices(prices)
    print(f"\n=== {LABEL} ===")
    print(
        f"diversified core on {'/'.join(SCENARIO_UNIVERSE)} history {dates[0]} .. {dates[-1]} "
        f"({len(dates)} days); paths start from the last close; monthly rebalance; 10 bps"
    )
    generators = {
        "block bootstrap (20-day blocks)": block_bootstrap_paths(rets, horizon, n_paths, seed=11),
        "GARCH(1,1) filtered historical simulation": garch_paths(rets, horizon, n_paths, seed=12),
        "AR(1) + correlated Gaussian shocks": model_paths(rets, horizon, n_paths, seed=13),
    }
    for mode in (CoreMode.DEFENSIVE, CoreMode.GROWTH):
        fn = diversified_core_weights(SCENARIO_UNIVERSE, mode)
        print(
            f"\n-- mode {mode.value}: today's weights "
            + ", ".join(f"{s} {w:.0%}" for s, w in zip(SCENARIO_UNIVERSE, fn(prices), strict=True))
        )
        for name, paths in generators.items():
            report = run_scenarios(prices, paths, fn, generator=name)
            print(report.summary())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("data/ohlc"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--paths", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=252)
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()
    cfg = TrainConfig(lookback=20, epochs=args.epochs, patience=5, max_train=2000, seed=7)
    started = time.perf_counter()
    forecast_section(args.dir, cfg, args.folds)
    if not args.skip_sweep:
        sweep_section(args.dir, cfg, args.folds)
    scenario_section(args.dir, args.paths, args.horizon)
    print(f"\ntotal runtime {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
