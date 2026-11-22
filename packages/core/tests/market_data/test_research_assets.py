import ast
import sys
import types
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from quant_os_core.market_data import research_assets
from quant_os_core.market_data.research_assets import (
    RESEARCH_ASSETS,
    ResearchSeries,
    load_all_research,
    load_research_series,
    yfinance_fetch,
)


def fake_fetch(symbol, period):
    base = datetime(2026, 1, 5)
    return [
        {"time": base + timedelta(days=1), "high": 71.0, "low": 69.0, "close": 70.5},
        {"time": base, "high": 70.0, "low": 68.0, "close": 69.5},
        {"time": "2026-01-07T00:00:00+05:00", "high": 72.0, "low": 70.0, "close": 71.0},
        {"time": base + timedelta(days=3), "high": float("nan"), "low": 1.0, "close": 1.0},
        {"time": base + timedelta(days=4), "high": 1.0, "low": 2.0, "close": 1.5},
        {"time": base + timedelta(days=5), "high": 0.0, "low": 0.0, "close": 0.0},
        {"time": base + timedelta(days=6), "low": 1.0, "close": 1.0},
    ]


def test_symbols_and_blocking_rules():
    assert set(RESEARCH_ASSETS) == {"CL=F", "BZ=F", "EURUSD=X", "GBPUSD=X", "USDPKR=X"}
    assert RESEARCH_ASSETS["CL=F"].blocked_by == ("R1", "R2")
    assert "R10" in RESEARCH_ASSETS["USDPKR=X"].blocked_by


def test_series_is_tagged_research_only_and_cleaned():
    series = load_research_series("cl=f", fetch=fake_fetch)
    assert series.research_only is True and "research only" in series.note
    assert series.symbol == "CL=F"
    assert series.closes() == [Decimal("69.5"), Decimal("70.5"), Decimal("71.0")]
    assert all(p.time.tzinfo is not None for p in series.points)
    assert series.points[-1].time == datetime(2026, 1, 6, 19, tzinfo=UTC)


def test_research_only_cannot_be_switched_off():
    asset = RESEARCH_ASSETS["EURUSD=X"]
    with pytest.raises(ValueError):
        ResearchSeries(asset, (), research_only=False)  # type: ignore[arg-type]


def test_unknown_symbol_and_bad_timestamp():
    with pytest.raises(ValueError, match="unknown research symbol"):
        load_research_series("SPY", fetch=fake_fetch)
    with pytest.raises(ValueError, match="unsupported timestamp"):
        load_research_series(
            "BZ=F", fetch=lambda s, p: [{"time": 12345, "high": 2, "low": 1, "close": 1.5}]
        )


def test_load_all_passes_period():
    seen = []

    def fetch(symbol, period):
        seen.append((symbol, period))
        return fake_fetch(symbol, period)

    data = load_all_research(period="1y", fetch=fetch)
    assert set(data) == set(RESEARCH_ASSETS)
    assert all(period == "1y" for _, period in seen)
    assert all(s.research_only for s in data.values())


def test_yfinance_fetch_with_fake_module(monkeypatch):
    class Stamp:
        def to_pydatetime(self):
            return datetime(2026, 2, 2, tzinfo=timezone(timedelta(hours=-5)))

    class Frame:
        def iterrows(self):
            yield Stamp(), {"High": 1.10, "Low": 1.08, "Close": 1.09}

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period, interval, auto_adjust):
            assert (period, interval, auto_adjust) == ("2y", "1d", False)
            return Frame()

    fake = types.ModuleType("yfinance")
    fake.Ticker = Ticker
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    rows = yfinance_fetch("EURUSD=X", "2y")
    assert rows[0]["close"] == 1.09
    series = load_research_series("EURUSD=X", period="2y")
    assert series.points[0].time == datetime(2026, 2, 2, 5, tzinfo=UTC)


def test_yfinance_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(RuntimeError, match="pip install yfinance"):
        yfinance_fetch("CL=F", "1y")


def test_module_has_no_path_to_orders():
    source = Path(research_assets.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    forbidden = ("execution", "risk_engine", "portfolio", "ops", "agents", "strategies")
    assert not any(part in name.split(".") for name in imported for part in forbidden)
    public = [n.lower() for n in dir(research_assets) if not n.startswith("_")]
    assert not any(w in n for n in public for w in ("order", "signal", "plan", "trade"))
