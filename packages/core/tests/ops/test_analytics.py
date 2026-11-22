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


def test_dashboard_calculators_fx_i18n_and_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quant_os_core.knowledge import rag
    from quant_os_core.ops.fx import FxService

    monkeypatch.delenv("QUANT_OS_OLLAMA_MODEL", raising=False)
    monkeypatch.delattr(rag, "answer_multilingual", raising=False)
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "risk.md").write_text("# Stop loss\nA stop loss limits the loss on one trade.\n")
    store = StateStore(tmp_path / "t.db")
    payload = {"result": "success", "base_code": "USD", "rates": {"USD": 1, "PKR": 280}}
    fx = FxService(lambda _url: payload, tmp_path / "fx.json")
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0), make_handler(store, "x" * 20, fx=fx, knowledge_dir=kb)
    )
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path: str, body: Any = None, raw: bytes | None = None) -> tuple[int, Any]:
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        req = urllib.request.Request(base + path, data=data, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read() or b"{}")

    try:
        code, out = call(
            "/api/calc/position_size", {"equity": 10000, "risk_pct": 1, "entry": 50, "stop": 47.5}
        )
        assert code == 200 and out["shares"] == "40"
        assert call("/api/calc/position_size", {"equity": "x"})[0] == 400
        assert call("/api/calc/nope", {})[0] == 404
        assert call("/api/calc/zakat", raw=b"{not json")[0] == 400
        assert call("/api/calc/zakat", raw=b"{" + b" " * 20_000 + b"}")[0] == 413
        code, rates = call("/api/fx?base=USD")
        assert code == 200 and rates["rates"]["PKR"] == "280"
        assert call("/api/fx?base=ZZZ")[0] == 400
        code, conv = call("/api/fx/convert?amount=2&from=USD&to=PKR")
        assert code == 200 and Decimal(conv["converted"]) == Decimal("560")
        code, ur = call("/api/i18n?lang=ur")
        assert code == 200 and ur["dir"] == "rtl" and ur["strings"]["tab_calc"] == "کیلکولیٹر"
        code, reply = call("/api/ask", {"question": "stop loss on one trade", "lang": "en"})
        assert code == 200 and reply["sources"] == ["risk.md - Stop loss"]
        assert "limits the loss" in reply["answer"]
        assert call("/api/ask", {"question": ""})[0] == 400
        assert call("/api/ask", {"question": "x" * 1001})[0] == 400
        monkeypatch.setattr(
            rag,
            "answer_multilingual",
            lambda q, lang: {"answer": f"{lang}:{q}", "sources": ["s"]},
            raising=False,
        )
        assert call("/api/ask", {"question": "hi", "lang": "ar"})[1] == {
            "answer": "ar:hi",
            "sources": ["s"],
        }
        page = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert 'id="lang"' in page and "/api/calc/" in page and "SpeechRecognition" in page
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_dashboard_ask_returns_503_when_knowledge_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quant_os_core.knowledge import rag

    monkeypatch.delenv("QUANT_OS_OLLAMA_MODEL", raising=False)
    monkeypatch.delattr(rag, "answer_multilingual", raising=False)
    store = StateStore(tmp_path / "t.db")
    handler = make_handler(store, "x" * 20, knowledge_dir=tmp_path / "missing")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    req = urllib.request.Request(
        f"http://127.0.0.1:{httpd.server_address[1]}/api/ask",
        data=b'{"question": "zakat"}',
        method="POST",
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req, timeout=5)
        assert err.value.code == 503
    finally:
        httpd.shutdown()
        httpd.server_close()
