"""Offline tests for the paper-loop runner: bar fetching and the ``--once`` entry point."""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from quant_os_core.agents.exits import Bar
from quant_os_core.ops import run_paper
from quant_os_core.ops.paper_loop import StepReport
from quant_os_core.ops.resilience import CircuitBreaker
from quant_os_core.ops.state import StateStore

EXAMPLE = Path(__file__).resolve().parents[4] / "examples" / "paper.json"
BAR = Bar(datetime(2026, 11, 20, tzinfo=UTC), Decimal("101"), Decimal("99"), Decimal("100"))


def _trader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> run_paper.Trader:
    monkeypatch.setenv("APCA_API_KEY_ID", "paper-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "paper-secret")
    config = json.loads(EXAMPLE.read_text()) | {
        "state": str(tmp_path / "t.db"),
        "board": str(tmp_path / "b.db"),
    }
    trader, _ = run_paper.build_trader(config)
    return trader


class _FakeSource:
    def __init__(self, key: str, secret: str, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str, str, str]] = []

    def bars(self, symbol: str, start: str, stop: str, timeframe: str) -> list[Bar]:
        self.calls.append((symbol, start, stop, timeframe))
        if self.fail:
            raise ConnectionError(f"{symbol} feed down")
        return [BAR]


def _events(trader: Any) -> list[str]:
    return [row["message"] for row in trader.store.recent("events")]


def test_fetch_bars_routes_by_asset_class_and_reads_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trader = _trader(tmp_path, monkeypatch)
    stocks, crypto = _FakeSource("", ""), _FakeSource("", "")
    monkeypatch.setattr(run_paper, "AlpacaBars", lambda key, secret: stocks)
    monkeypatch.setattr(run_paper, "AlpacaCryptoBars", lambda key, secret: crypto)
    monkeypatch.setattr(run_paper, "retry", lambda fn: fn())

    out = run_paper.fetch_bars(trader, "1Day", 30, CircuitBreaker("data", failure_threshold=3))
    assert out == {"AAPL": [BAR], "BTC/USD": [BAR]}
    assert [call[0] for call in stocks.calls] == ["AAPL"]
    assert [call[0] for call in crypto.calls] == ["BTC/USD"]
    assert stocks.calls[0][3] == "1Day"

    csv_file = tmp_path / "AAPL.csv"
    csv_file.write_text("time,high,low,close\n2026-11-20T10:30:00+00:00,301,295,299.5\n")
    out = run_paper.fetch_bars(
        trader, "1Day", 30, CircuitBreaker("data"), {"AAPL": str(csv_file)}
    )
    assert out["AAPL"][0].close == Decimal("299.5")
    assert len(stocks.calls) == 1  # the CSV replaced the API call for AAPL


def test_fetch_bars_logs_and_skips_broken_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trader = _trader(tmp_path, monkeypatch)
    monkeypatch.setattr(run_paper, "AlpacaBars", lambda key, secret: _FakeSource(key, secret))
    monkeypatch.setattr(
        run_paper, "AlpacaCryptoBars", lambda key, secret: _FakeSource(key, secret, fail=True)
    )
    monkeypatch.setattr(run_paper, "retry", lambda fn: fn())

    out = run_paper.fetch_bars(
        trader,
        "1Hour",
        5,
        CircuitBreaker("data", failure_threshold=3),
        {"AAPL": str(tmp_path / "missing.csv")},
    )
    assert out == {}  # missing CSV and failing crypto feed both skipped, not raised
    messages = _events(trader)
    assert any(m.startswith("AAPL csv:") for m in messages)
    assert any("BTC/USD" in m and "feed down" in m for m in messages)


@dataclass
class _FakeTrader:
    store: StateStore
    universe: dict[str, Any] = field(default_factory=lambda: {"AAPL": object()})
    crash: bool = False
    alerts_sent: list[tuple[str, str]] = field(default_factory=list)
    steps: int = 0

    @property
    def alerts(self) -> "_FakeTrader":
        return self

    def send(self, level: str, title: str, body: str) -> None:
        self.alerts_sent.append((level, title))

    def step(self, now: datetime, bars: dict[str, Any]) -> StepReport:
        self.steps += 1
        if self.crash:
            raise RuntimeError("broker unreachable")
        return StepReport(time=now, entries_submitted=["AAPL"])


@pytest.mark.parametrize("crash", [False, True])
def test_main_once_runs_one_step_and_survives_a_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    crash: bool,
) -> None:
    fake = _FakeTrader(StateStore(tmp_path / "t.db"), crash=crash)
    config = {"timeframe": "1Hour", "lookback_days": 10, "bars_csv": {"AAPL": "x.csv"}}
    seen: dict[str, Any] = {}

    def fake_fetch(trader: Any, timeframe: str, lookback: int, breaker: Any, csv: Any) -> dict:
        seen.update(timeframe=timeframe, lookback=lookback, csv=csv)
        return {}

    monkeypatch.setattr(run_paper, "build_trader", lambda cfg: (fake, cfg))
    monkeypatch.setattr(run_paper, "fetch_bars", fake_fetch)
    config_file = tmp_path / "paper.json"
    config_file.write_text(json.dumps(config))

    assert run_paper.main(["--config", str(config_file), "--once"]) == 0
    assert fake.steps == 1
    assert seen == {"timeframe": "1Hour", "lookback": 10, "csv": {"AAPL": "x.csv"}}
    messages = _events(fake)
    assert any(m.startswith("started: 1 symbols") for m in messages)
    if crash:
        assert fake.alerts_sent == [("critical", "loop step crashed")]
        assert any("step crashed" in m for m in messages)
    else:
        assert fake.alerts_sent == []
        assert "entries=['AAPL']" in capsys.readouterr().err
