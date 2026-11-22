import json
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from quant_os_core.ops import analytics
from quant_os_core.ops.dashboard import make_handler, snapshot
from quant_os_core.ops.state import StateStore


def _csv(folder: Path, symbol: str, n: int = 260) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["time,high,low,close"]
    for i in range(n):
        c = 100 + i * 0.1 + (i % 7) * 0.3
        day = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=i)
        lines.append(f"{day.isoformat()},{c + 1},{c - 1},{c}")
    (folder / f"{symbol}.csv").write_text("\n".join(lines))


def test_metrics_match_hand_computed_values() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    curve = [((t0 + timedelta(days=i)).isoformat(), e) for i, e in enumerate([100, 110, 99, 120])]
    m = analytics.metrics(curve, [0.10, -0.05, 0.02])
    assert m["total_return"] == pytest.approx(0.20)
    assert m["max_drawdown"] == pytest.approx(99 / 110 - 1)
    assert m["win_rate"] == pytest.approx(2 / 3)
    assert m["profit_factor"] == pytest.approx(0.12 / 0.05)
    assert m["cagr"] is None  # under 30 days: annualising would mislead
    assert set(m) == set(analytics.FORMULAS)
    assert analytics.metrics([], [])["sharpe"] is None


def test_bars_have_indicators_and_reject_unknown_symbols(tmp_path: Path) -> None:
    _csv(tmp_path, "SPUS")
    out = analytics.bars(tmp_path, "SPUS", limit=100)["bars"]
    assert len(out) == 100
    last = out[-1]
    assert last["sma200"] is not None and last["bb_lower"] < last["close"] < last["bb_upper"]
    assert 0 <= last["rsi"] <= 100 and last["low"] <= last["open"] <= last["high"]
    with pytest.raises(ValueError):
        analytics.bars(tmp_path, "../secrets")
    with pytest.raises(ValueError):
        analytics.bars(tmp_path, "NOPE")


def test_depth_is_cumulative_with_spread() -> None:
    book = {
        "b": [{"p": 99, "s": 1}, {"p": 100, "s": 2}],
        "a": [{"p": 101, "s": 3}, {"p": 102, "s": 1}],
    }
    d = analytics.depth("BTC/USD", lambda _s: book, window=0.05)
    assert d["bids"] == [[100, 2], [99, 3]] and d["asks"] == [[101, 3], [102, 4]]
    assert d["spread"] == 1 and d["spread_bps"] == pytest.approx(1 / 100.5 * 10_000)
    with pytest.raises(ValueError):
        analytics.depth("SPY", lambda _s: book)


def test_dashboard_serves_charts_depth_and_metrics(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "t.db")
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    for i, e in enumerate(["100", "105", "102"]):
        store.record_equity(t0 + timedelta(days=i), Decimal(e), Decimal(e))
    assert snapshot(store)["metrics"]["total_return"] == pytest.approx(0.02)
    data = tmp_path / "ohlc"
    _csv(data, "SPUS")

    def fake_book(symbol: str) -> dict[str, Any]:
        if symbol == "ETH/USD":
            raise OSError("offline")
        return {"b": [{"p": 100, "s": 1}], "a": [{"p": 101, "s": 1}], "x": [{"p": 900}]}

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, "x" * 20, data, fake_book))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    get = lambda path: json.loads(urllib.request.urlopen(base + path, timeout=5).read())  # noqa: E731
    try:
        assert get("/api/symbols") == ["SPUS"]
        assert get("/api/bars?symbol=SPUS")["bars"][-1]["rsi"] is not None
        assert get("/api/depth?symbol=BTC/USD")["spread"] == 1
        for bad, code in (("/api/bars?symbol=..%2Fx", 400), ("/api/depth?symbol=ETH/USD", 502)):
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(base + bad, timeout=5)
            assert err.value.code == code
        page = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "Order book depth" in page and "Setup guide" in page
    finally:
        httpd.shutdown()


def test_depth_drops_far_off_levels() -> None:
    book = {
        "b": [{"p": 100, "s": 1}, {"p": 50, "s": 9}],
        "a": [{"p": 101, "s": 1}, {"p": 500, "s": 9}],
    }
    d = analytics.depth("BTC/USD", lambda _s: book)
    assert d["bids"] == [[100, 1]] and d["asks"] == [[101, 1]]
