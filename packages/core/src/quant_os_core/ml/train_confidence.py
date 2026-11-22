"""Train, gate and save the neural confidence model; write a data-science report.

    python -m quant_os_core.ml.train_confidence \\
        --csv data/AAPL.csv data/MSFT.csv data/BTCUSD.csv \\
        --signal donchian --out models/confidence.npz --state state/trading.db \\
        --report reports/confidence.md

CSV columns: time (ISO 8601 with timezone), high, low, close. Daily bars from
``research/backtests/run_halal_v3.py fetch`` can be converted, or exported from a broker.
The model is saved only if it passes every gate AND beats the current champion.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from quant_os_core.ml.champion import GatePolicy, run_gate
from quant_os_core.ml.data_science import (
    data_quality,
    feature_diagnostics,
    render_report,
    return_profile,
)
from quant_os_core.ml.features import FEATURE_NAMES
from quant_os_core.ml.neural_signal import (
    EventDataset,
    TrainingReport,
    bars_to_arrays,
    build_event_dataset,
    train_confidence_model,
)
from quant_os_core.ops.run_paper import load_csv_bars
from quant_os_core.ops.signals import DonchianBreakout, ForecastBreakout
from quant_os_core.ops.state import StateStore

SIGNALS = {"donchian": DonchianBreakout, "forecast": ForecastBreakout}


def summarise(report: TrainingReport) -> str:
    lines = [
        "## Neural confidence model (walk-forward, out of sample)",
        "",
        f"- Events {report.events}, tested {report.tested}, base win rate {report.base_rate:.1%}",
        f"- AUC: ensemble {report.auc_model:.3f}, network {report.auc_network:.3f}, "
        f"logistic {report.auc_baseline:.3f} (0.5 = no skill)",
        f"- Brier: model {report.brier_model:.4f} vs base rate {report.brier_base_rate:.4f}; "
        f"calibration error {report.ece_calibrated:.3f}",
        "- Win rate by probability floor: "
        + ", ".join(f">= {k}: {v:.1%}" for k, v in report.precision_at.items() if v == v),
        "- Fold skill (log-loss gain vs base rate): "
        + ", ".join(f"{f:+.3f}" for f in report.fold_scores),
        f"- Avg trade: all signals {report.mean_return_all:+.3%}, filtered at "
        f"{report.chosen_threshold:.2f} {report.mean_return_filtered:+.3%}",
        f"- Deflated Sharpe {report.deflated_sharpe:.2f}; PBO (diagnostic) {report.pbo:.2f}",
        f"- Verdict: {'PASSED' if report.passed else 'FAILED - ' + '; '.join(report.reasons)}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the neural confidence layer")
    parser.add_argument("--csv", type=Path, nargs="+", required=True)
    parser.add_argument("--signal", choices=sorted(SIGNALS), default="donchian")
    parser.add_argument("--max-holding", type=int, default=20)
    parser.add_argument("--cost", type=float, default=0.002, help="round-trip cost fraction")
    parser.add_argument("--out", type=Path, default=Path("models/confidence.npz"))
    parser.add_argument("--state", type=Path, default=Path("state/trading.db"))
    parser.add_argument("--report", type=Path, default=Path("reports/confidence.md"))
    args = parser.parse_args(argv)

    primary = SIGNALS[args.signal]()
    sections: list[str] = []
    datasets: list[EventDataset] = []
    for path in args.csv:
        bars = load_csv_bars(path)
        high, low, close = bars_to_arrays(bars)
        quality = data_quality([b.time for b in bars], high, low, close)
        data = build_event_dataset(
            path.stem, bars, primary, max_holding=args.max_holding, round_trip_cost=args.cost
        )
        datasets.append(data)
        sections.append(
            render_report(f"{path.stem}: {len(data)} events", quality, return_profile(close))
        )
    pooled = EventDataset.pool(datasets)
    diagnostics = feature_diagnostics(pooled.features, pooled.labels, FEATURE_NAMES)
    model, report = train_confidence_model(pooled)
    summary = summarise(report)

    store = StateStore(args.state)
    champion = store.champion("confidence")
    champion_scores = (
        [float(v) for k, v in sorted(champion["metrics"].items()) if k.startswith("fold_")]
        if champion
        else None
    )
    if champion_scores and len(champion_scores) != len(report.fold_scores):
        # Different history -> different folds: compare against the champion's average skill.
        mean = sum(champion_scores) / len(champion_scores)
        champion_scores = [mean] * len(report.fold_scores)
    verdict = "not saved (failed gates)"
    if model is not None:
        version = datetime.now(UTC).strftime("%Y%m%d%H%M")
        # PBO is diagnostic for this model type (threshold variants are near-copies).
        gate = run_gate(
            store,
            "confidence",
            version,
            champion_scores if champion_scores else None,
            report.fold_scores,
            deflated_sharpe=report.deflated_sharpe,
            pbo=0.0,
            policy=GatePolicy(min_folds=len(report.fold_scores)),
        )
        if gate.promote:
            model.metadata["version"] = version
            model.save(args.out)
            verdict = f"saved as champion {version} -> {args.out}"
        else:
            verdict = f"kept the current champion ({gate.reason})"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    text = summary + f"\n- Model: {verdict}\n\n" + "\n".join(sections)
    feature_lines = [
        "",
        "## Feature value across all symbols",
        "",
        "| Feature | IC | 1st half | 2nd half | Stable |",
        "|---|---|---|---|---|",
    ]
    for d in diagnostics.diagnostics:
        feature_lines.append(
            f"| {d.name} | {d.ic:+.3f} | {d.ic_first_half:+.3f} | "
            f"{d.ic_second_half:+.3f} | {'yes' if d.stable else 'no'} |"
        )
    args.report.write_text(text + "\n".join(feature_lines) + "\n", encoding="utf-8")
    print(summary + f"- Model: {verdict}\n- Report: {args.report}", file=sys.stderr)
    return 0 if model is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
