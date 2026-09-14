#!/usr/bin/env python3
"""Small deterministic release benchmark; informational, not a correctness gate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from time import perf_counter

from fastapi.testclient import TestClient

from quant_os_api.main import app
from quant_os_core.backtesting import EventDrivenBacktester
from quant_os_core.backtesting.engine import StrategyContext
from quant_os_core.backtesting.models import OrderRequest
from quant_os_core.instruments import InstrumentId
from quant_os_core.market_data.models import Candle

BTC = InstrumentId(venue="BENCH", symbol="BTCUSDT")


class NoOpStrategy:
    def on_bar(self, candle: Candle, context: StrategyContext) -> tuple[OrderRequest, ...]:
        del candle, context
        return ()


def backtest_benchmark(count: int = 10_000) -> float:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles = tuple(
        Candle(
            instrument_id=BTC,
            timestamp=start + timedelta(minutes=index),
            source="benchmark",
            timeframe="1m",
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1000"),
            close_time=start + timedelta(minutes=index + 1),
        )
        for index in range(count)
    )
    began = perf_counter()
    EventDrivenBacktester(initial_cash=Decimal("100000")).run(candles, NoOpStrategy())
    return perf_counter() - began


def api_benchmark(count: int = 1_000) -> float:
    began = perf_counter()
    with TestClient(app) as client:
        for _ in range(count):
            response = client.get("/health")
            if response.status_code != 200:
                raise RuntimeError("health benchmark request failed")
    return perf_counter() - began


def main() -> None:
    backtest_seconds = backtest_benchmark()
    api_seconds = api_benchmark()
    print(f"event_backtest_10k_bars_seconds={backtest_seconds:.6f}")
    print(f"api_health_1000_requests_seconds={api_seconds:.6f}")


if __name__ == "__main__":
    main()
