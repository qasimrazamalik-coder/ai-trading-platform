"""Run the paper-trading loop against Alpaca's paper account.

    export APCA_API_KEY_ID=... APCA_API_SECRET_KEY=...          # paper keys
    python -m quant_os_core.ops.run_paper --config paper.json --once   # one step (cron)
    python -m quant_os_core.ops.run_paper --config paper.json          # loop forever

See ``docs/RUNBOOK.md`` for the config file, alerts, the kill switch and recovery.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from quant_os_core.agents.board import AgentBoard, AnalystRole, BoardAnalyst
from quant_os_core.agents.council import AnalystCouncil
from quant_os_core.agents.day_trading import day_trading_policy, market_for_venue
from quant_os_core.agents.exits import Bar
from quant_os_core.agents.pipeline import EntryPipeline
from quant_os_core.agents.profit_engine import CostModel, RiskDial, dial_policy
from quant_os_core.agents.simulator import ExitPolicy
from quant_os_core.agents.trade_plan import PlanGovernor
from quant_os_core.instruments import AssetClass, Instrument, InstrumentId, InstrumentType
from quant_os_core.market_data.free_sources import AlpacaBars, AlpacaCryptoBars
from quant_os_core.ops.alerts import AlertSink, MultiSink, NtfySink, StoreSink, WebhookSink
from quant_os_core.ops.alpaca_paper import AlpacaPaperBroker
from quant_os_core.ops.paper_loop import PaperTrader, UniverseItem
from quant_os_core.ops.resilience import CircuitBreaker, retry
from quant_os_core.ops.signals import DonchianBreakout
from quant_os_core.ops.state import StateStore
from quant_os_core.risk_engine import RiskEngine, default_protections
from quant_os_core.shariah import (
    CompanyFinancials,
    CryptoAsset,
    ShariahGate,
    ShariahProfile,
    halal_risk_limits,
)


def _universe(config: dict[str, Any]) -> dict[str, UniverseItem]:
    items: dict[str, UniverseItem] = {}
    for row in config["universe"]:
        symbol = str(row["symbol"]).upper()
        asset = AssetClass(row["asset_class"])
        base, _, quote = symbol.partition("/")
        venue = str(row.get("venue", "ALPACA"))
        financials = row.get("financials")
        items[symbol] = UniverseItem(
            instrument=Instrument(
                instrument_id=InstrumentId(venue=venue, symbol=symbol),
                asset_class=asset,
                instrument_type=InstrumentType.SPOT,
                base_currency=(base or symbol)[:12].ljust(3, "X"),
                quote_currency=(quote or row.get("currency", "USD"))[:12],
                price_precision=8,
                quantity_precision=8,
            ),
            financials=(
                CompanyFinancials(
                    sector=str(financials["sector"]),
                    **{k: Decimal(str(v)) for k, v in financials.items() if k != "sector"},
                )
                if financials
                else None
            ),
            crypto_asset=(
                CryptoAsset(symbol, str(row["token_category"]))
                if row.get("token_category")
                else None
            ),
            market=market_for_venue(venue),
        )
    return items


def _alerts(config: dict[str, Any], store: StateStore) -> AlertSink:
    sinks: list[AlertSink] = [StoreSink(store)]
    alert_cfg = config.get("alerts") or {}
    topic = os.environ.get("QUANT_OS_NTFY_TOPIC") or alert_cfg.get("ntfy_topic")
    if topic:
        sinks.append(NtfySink(str(topic)))
    webhook = os.environ.get("QUANT_OS_WEBHOOK_URL") or alert_cfg.get("webhook_url")
    if webhook:
        sinks.append(WebhookSink(str(webhook)))
    return MultiSink(sinks)


def build_trader(config: dict[str, Any]) -> tuple[PaperTrader, dict[str, Any]]:
    store = StateStore(Path(config.get("state", "state/trading.db")))
    board = AgentBoard(Path(config.get("board", "state/board.db")))
    dial = RiskDial(config.get("dial", "balanced"))
    day = bool(config.get("day_trading", False))
    policy = dial_policy(dial, day_trading_policy() if day else None)
    if day:  # intraday trades risk half of the dial's per-trade budget
        policy = replace(policy, risk_per_trade=policy.risk_per_trade / 2)
    pipeline = EntryPipeline(
        shariah=ShariahGate(ShariahProfile.spot_crypto_and_day_trading()),
        protections=default_protections(),
        governor=PlanGovernor(policy),
        council=AnalystCouncil([BoardAnalyst(board, role) for role in AnalystRole]),
        risk=RiskEngine(halal_risk_limits()),
        strategy_id="paper-loop",
        cost_model=CostModel(),
    )
    trader = PaperTrader(
        broker=AlpacaPaperBroker(
            os.environ.get("APCA_API_KEY_ID", ""), os.environ.get("APCA_API_SECRET_KEY", "")
        ),
        store=store,
        pipeline=pipeline,
        universe=_universe(config),
        signal=DonchianBreakout(),
        alerts=_alerts(config, store),
        exit_policy=ExitPolicy(
            trail_fraction=None if day else Decimal("0.08"),
            max_holding=timedelta(hours=8) if day else timedelta(days=60),
            flatten_before_close=day,
        ),
        board=board,
    )
    return trader, config


def load_csv_bars(path: Path) -> list[Bar]:
    """CSV with a header row: time (ISO 8601 with timezone), high, low, close."""

    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return [
            Bar(datetime.fromisoformat(r["time"]), Decimal(r["high"]), Decimal(r["low"]),
                Decimal(r["close"]))
            for r in csv.DictReader(handle)
        ]


def fetch_bars(
    trader: PaperTrader,
    timeframe: str,
    lookback_days: int,
    breaker: CircuitBreaker,
    csv_paths: dict[str, str] | None = None,
) -> dict[str, Sequence[Bar]]:
    end = datetime.now(UTC)
    start = (end - timedelta(days=lookback_days)).date().isoformat()
    stop = end.date().isoformat()
    key, secret = os.environ.get("APCA_API_KEY_ID", ""), os.environ.get("APCA_API_SECRET_KEY", "")
    stocks, crypto = AlpacaBars(key, secret), AlpacaCryptoBars(key, secret)
    out: dict[str, Sequence[Bar]] = {}
    for symbol, item in trader.universe.items():
        if csv_paths and symbol in csv_paths:  # e.g. PSX bars exported from your broker
            try:
                out[symbol] = load_csv_bars(Path(csv_paths[symbol]))
            except (OSError, KeyError, ValueError) as exc:
                trader.store.log("warning", "data", f"{symbol} csv: {exc!r}")
            continue
        source = crypto if item.instrument.asset_class is AssetClass.CRYPTO else stocks

        def load(source: Any = source, symbol: str = symbol) -> list[Bar]:
            return retry(lambda: source.bars(symbol, start, stop, timeframe))

        try:
            out[symbol] = breaker.call(load)
        except Exception as exc:
            trader.store.log("warning", "data", f"{symbol}: {exc!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quant OS paper-trading loop (Alpaca paper)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true", help="run one step and exit")
    args = parser.parse_args(argv)
    trader, config = build_trader(json.loads(args.config.read_text("utf-8")))
    timeframe = str(config.get("timeframe", "1Day"))
    interval = int(config.get("interval_minutes", 1440))
    lookback = int(config.get("lookback_days", 150))
    data_breaker = CircuitBreaker("market-data", failure_threshold=3)
    trader.store.log("info", "loop", f"started: {len(trader.universe)} symbols, {timeframe}")
    while True:
        try:
            now = datetime.now(UTC)
            bars = fetch_bars(trader, timeframe, lookback, data_breaker, config.get("bars_csv"))
            report = trader.step(now, bars)
            print(
                f"{now:%Y-%m-%d %H:%M} entries={report.entries_submitted} fills={report.fills} "
                f"exits={report.exits} blocked={report.blocked} issues={len(report.issues)}",
                file=sys.stderr,
            )
        except Exception as exc:  # keep the loop alive; the next step fails closed if needed
            trader.store.log("critical", "loop", f"step crashed: {exc!r}")
            trader.alerts.send("critical", "loop step crashed", repr(exc))
        if args.once:
            return 0
        time.sleep(interval * 60)


if __name__ == "__main__":
    raise SystemExit(main())
