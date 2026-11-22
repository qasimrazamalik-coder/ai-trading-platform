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


def test_dashboard_journal_ledger_practice_rulebook_and_settings(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state" / "t.db")
    data = tmp_path / "ohlc"
    _csv(data, "SPUS")
    config = tmp_path / "paper_core.json"
    fund = {"symbol": "SPUS", "financials": {"total_revenue": "100", "noncompliant_revenue": "2"}}
    config.write_text(json.dumps({"universe": [fund]}))
    token = "t" * 24
    handler = make_handler(store, token, data, ask=lambda q, lang: {}, config_path=config)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path: str, body: Any = None, auth: str | None = token) -> tuple[int, Any]:
        payload = None if body is None else json.dumps(body).encode()
        headers = {"Authorization": f"Bearer {auth}"} if auth else {}
        req = urllib.request.Request(
            base + path, data=payload, method="POST" if payload else "GET", headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = resp.read()
                kind = resp.headers.get("Content-Type", "")
                return resp.status, raw.decode() if "csv" in kind else json.loads(raw)
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read() or b"{}")

    def fill(**fields: str) -> tuple[int, Any]:
        return call("/api/ledger/fill", {"symbol": "SPUS", **fields})

    try:
        # every write needs the token
        assert call("/api/ledger/fill", {"symbol": "SPUS"}, auth=None)[0] == 401
        assert call("/api/settings", {"language": "ur"}, auth="wrong" * 5)[0] == 401
        assert call("/api/journal", {"symbol": "SPUS"}, auth=None)[0] == 401
        code, buy = fill(
            side="buy", quantity="10", price="40", fee="1", time="2026-03-02T10:00+05:00"
        )
        assert code == 200 and buy["id"] > 0
        code, sell = fill(side="sell", quantity="10", price="45")
        assert code == 200
        assert fill(side="sell", quantity="1", price="45")[0] == 400  # would be a short sale
        assert fill(side="buy", quantity="1", price="1", time="2026-01-01T00:00")[0] == 400
        assert call("/api/ledger/fill", {"symbol": 5, "side": "buy"})[0] == 400
        assert call("/api/ledger/fill", ["not", "an", "object"])[0] == 400
        assert call("/api/ledger/cash", {"kind": "deposit", "amount": "1000"})[0] == 200
        dividend = {"symbol": "SPUS", "amount": "100", "time": "2026-06-01T00:00:00+00:00"}
        assert call("/api/ledger/dividend", dividend)[0] == 200
        assert (
            call("/api/ledger/payment", {"kind": "zakat", "year": "2026", "amount": "5"})[0] == 200
        )
        code, ledger = call("/api/ledger?year=2026")
        assert code == 200 and ledger["pnl"]["total"] == "49.00"
        assert ledger["purification"]["funds"][0]["purify"] == "2.00"
        assert ledger["deposits"] == "1000.00" and ledger["zakat_paid"] == "5.00"
        assert call("/api/ledger?year=abc")[0] == 400
        code, csv_text = call("/api/ledger/export.csv")
        assert code == 200 and csv_text.startswith("type,time")
        zakat_in = {"year": 2026, "zakatable_wealth": "1000000", "nisab": "100000"}
        code, zakat = call("/api/ledger/zakat", zakat_in, auth=None)
        assert code == 200 and zakat["due"] == "25000.00" and zakat["outstanding"] == "24995.00"
        gold_in = {"year": 2026, "zakatable_wealth": "10", "gold_price_per_gram": "100"}
        code, gold = call("/api/ledger/zakat", gold_in, auth=None)
        assert code == 200 and gold["above_nisab"] is False

        entry_in = {
            "symbol": "SPUS",
            "setup": "breakout",
            "reason": "range break",
            "emotion": "calm",
            "rule_followed": True,
            "tags": "trend, etf",
            "fill_ids": f"{buy['id']},{sell['id']}",
        }
        code, entry = call("/api/journal", entry_in)
        assert code == 200 and entry["id"] > 0
        bad_flag = {"symbol": "SPUS", "setup": "x", "emotion": "calm", "rule_followed": "maybe"}
        assert call("/api/journal", bad_flag)[0] == 400
        bad_tags = {"symbol": "SPUS", "setup": "x", "emotion": "calm", "tags": 5}
        assert call("/api/journal", bad_tags)[0] == 400
        code, journal = call("/api/journal")
        assert journal["stats"]["by_setup"]["breakout"]["win_rate"] == 1.0
        assert journal["entries"][0]["tags"] == ["etf", "trend"]

        code, view = call("/api/bt/start", {"symbol": "SPUS", "start": 40}, auth=None)
        assert code == 200 and view["revealed"] == 41
        sid = view["session"]
        assert call("/api/bt/next", {"session": sid, "steps": 2}, auth=None)[1]["revealed"] == 43
        assert call("/api/bt/view", {"session": "nope"}, auth=None)[0] == 404
        assert call("/api/bt/start", {"symbol": "../x"}, auth=None)[0] == 400

        code, rules = call("/api/rulebook?lang=ar")
        assert code == 200 and rules["editable"] is False and rules["lang"] == "ar"

        code, described = call("/api/settings")
        assert code == 200 and any(s["locked"] for s in described["settings"])
        code, out = call("/api/settings", {"max_leverage": 3, "language": "ur"})
        assert code == 400 and "locked" in out["errors"]["max_leverage"]
        code, out = call("/api/settings", {"language": "ur"})
        assert code == 200
        assert next(s for s in out["settings"] if s["key"] == "language")["value"] == "ur"
        assert (tmp_path / "paper_core.settings.json").is_file()
        assert call("/api/nope", {"x": 1})[0] == 404

        page = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        tabs = ("journal", "ledger", "practice", "rules", "settings")
        for label in ("Kill switch", "<svg", *(f'data-tab="{t}"' for t in tabs)):
            assert label in page
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_dashboard_helpers_and_missing_config(tmp_path: Path) -> None:
    from quant_os_core.ops import dashboard

    assert dashboard._flag({"x": "on"}, "x") is True and dashboard._flag({}, "x") is False
    with pytest.raises(ValueError):
        dashboard._flag({"x": 3}, "x")
    assert dashboard._items(None) == [] and dashboard._items(["a", 1]) == ["a", "1"]
    with pytest.raises(ValueError):
        dashboard._items({"a": 1})
    assert dashboard._text({"k": None}, "k", "d") == "d"
    with pytest.raises(ValueError):
        dashboard._when({"time": 5})
    assert dashboard._when({}).tzinfo is not None
    assert dashboard._year_of({}) is not None and dashboard._year_of({"year": ["1800"]}) is None
    store = StateStore(tmp_path / "t.db")
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    for config in (None, bad):
        handler = make_handler(
            store, "x" * 20, tmp_path, ask=lambda q, lang: {}, config_path=config
        )
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{httpd.server_address[1]}/api/ledger"
            body = json.loads(urllib.request.urlopen(url, timeout=5).read())
            assert body["purification"]["funds"] == []
        finally:
            httpd.shutdown()
            httpd.server_close()
