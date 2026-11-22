import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from quant_os_core.agents.board import AgentBoard
from quant_os_core.agents.exits import OpenPosition
from quant_os_core.agents.signals import Signal, SignalKind
from quant_os_core.agents.trade_plan import TradePlan
from quant_os_core.knowledge.rag import KnowledgeBase
from quant_os_core.mcp_board_tools import BoardTools
from quant_os_core.mcp_server import ADVANCED_TOOLS, QuantOsMcp, build_server, cached_fx_rates
from quant_os_core.ops.alerts import StoreSink
from quant_os_core.ops.state import StateStore

MODERN = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {},
}
NOW = datetime(2026, 11, 22, 12, tzinfo=UTC)
FIN = {
    "sector": "diversified_fund",
    "market_cap": "3126103507",
    "interest_bearing_debt": "219704290",
    "interest_bearing_cash_and_securities": "74112695",
    "total_revenue": "516039909",
    "noncompliant_revenue": "9847385",
    "total_assets": "782321707",
    "tangible_assets": "582503776",
    "shares_outstanding": "1",
}


def _plan() -> TradePlan:
    return TradePlan(
        symbol="SPUS",
        entry_price=Decimal("40"),
        profit_target=Decimal("48"),
        stop_loss=Decimal("36"),
        invalidation_condition="close below the 200-day average",
        confidence=Decimal("0.6"),
        justification="core",
    )


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.delenv("QUANT_OS_OLLAMA_MODEL", raising=False)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "zakat.md").write_text(
        "# Zakat\n## Zakat basics\nZakat is 2.5% of zakatable wealth above the nisab.\n", "utf-8"
    )
    (docs / "risk.md").write_text(
        "# Risk\n## Drawdown\nA drawdown is the fall from the equity peak.\n## Spread\n"
        "The spread is the gap between bid and ask.\n",
        "utf-8",
    )
    store = StateStore(tmp_path / "core.db")
    store.set_value("core:last_rebalance", "2026-11-02")
    store.record_equity(NOW - timedelta(days=1), Decimal("55"), Decimal("5"))
    store.record_equity(NOW, Decimal("49.5"), Decimal("5"))
    store.save_position(OpenPosition(_plan(), Decimal("1.1"), NOW), "etf")
    stages = [
        {"stage": "shariah", "passed": True, "detail": "ok"},
        {"stage": "risk", "passed": False, "detail": "daily loss limit"},
    ]
    store.record_decision(NOW, "SPUS", False, "risk", Decimal("0"), stages)
    store.record_decision(NOW, "HLAL", True, None, Decimal("2"), stages[:1])
    StoreSink(store).send("warning", "Core signals", "Core signals (information only): x")
    config = tmp_path / "core.json"
    config.write_text(
        json.dumps(
            {
                "universe": [
                    {"symbol": "SPUS", "financials": FIN, "financials_source": "N-PORT 2026-08-31"},
                    {"symbol": "HLAL", "financials": FIN, "financials_source": "N-PORT 2026-01-31"},
                    {"symbol": "AAAU"},
                ]
            }
        ),
        "utf-8",
    )
    board = AgentBoard(tmp_path / "board.db")
    server = QuantOsMcp(
        KnowledgeBase.from_directory(docs),
        BoardTools(board),
        docs=docs,
        store=store,
        screening=config,
        rates=lambda a, b: Decimal("280.5") if (a, b) == ("USD", "PKR") else Decimal("x"),
        clock=lambda: NOW,
    )
    return {"server": server, "store": store, "board": board, "tmp": tmp_path}


def _tool(server: QuantOsMcp, name: str, **arguments: object) -> dict[str, Any]:
    params = {"name": name, "arguments": arguments, "_meta": MODERN}
    reply = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params})
    assert reply is not None and "result" in reply, reply
    return reply["result"]  # type: ignore[no-any-return]


def _text(server: QuantOsMcp, name: str, **arguments: object) -> str:
    result = _tool(server, name, **arguments)
    assert result["isError"] is False, result
    return result["content"][0]["text"]  # type: ignore[no-any-return]


def _dump(path: Path) -> list[str]:
    connection = sqlite3.connect(path)
    try:
        return list(connection.iterdump())
    finally:
        connection.close()


def test_every_advanced_tool_has_a_schema_and_is_read_only() -> None:
    for tool in ADVANCED_TOOLS:
        assert tool["inputSchema"]["type"] == "object" and tool["description"]
        assert tool["annotations"]["readOnlyHint"] is True


def test_bot_status_and_explain_decision(env: dict[str, Any]) -> None:
    server = env["server"]
    status = json.loads(_text(server, "bot_status"))
    assert status["equity"] == "49.5" and status["peak_equity"] == "55"
    assert status["drawdown_pct"] == "10.00" and status["kill_switch"] == "off"
    assert status["last_rebalance"] == "2026-11-02" and status["pending_orders"] == 0
    assert status["positions"][0]["symbol"] == "SPUS"
    latest = json.loads(_text(server, "explain_decision", limit=5))
    assert [d["symbol"] for d in latest] == ["HLAL", "SPUS"]
    spus = json.loads(_text(server, "explain_decision", symbol="spus"))
    assert spus[0]["verdict"] == "blocked at risk"
    assert spus[0]["stages"][1]["detail"] == "daily loss limit"
    assert _text(server, "explain_decision", symbol="XLK") == "No decisions recorded yet."


def test_screening_results_with_verdicts_and_staleness(env: dict[str, Any]) -> None:
    rows = {r["symbol"]: r for r in json.loads(_text(env["server"], "screening_results"))}
    assert rows["SPUS"]["verdict"] == "pass" and rows["SPUS"]["stale"] is False
    assert rows["SPUS"]["as_of"] == "2026-08-31" and rows["SPUS"]["age_days"] == 83
    assert rows["SPUS"]["ratios"]["interest_debt_to_market_cap"] == "7.03%"
    assert rows["HLAL"]["stale"] is True and rows["HLAL"]["age_days"] == 295
    assert rows["AAAU"]["verdict"].startswith("blocked (R6)") and rows["AAAU"]["stale"]
    one = json.loads(_text(env["server"], "screening_results", symbol="spus"))
    assert [r["symbol"] for r in one] == ["SPUS"]
    assert _tool(env["server"], "screening_results", symbol="NOPE")["isError"]


def test_signals_from_provider_or_latest_published(env: dict[str, Any]) -> None:
    published = _text(env["server"], "signals")
    assert "latest published at" in published and "Core signals" in published
    found = [Signal(SignalKind.TREND_EXIT, "SPUS", "warning", "SPUS closed below average")]
    server = QuantOsMcp(None, signals=lambda: found)
    body = json.loads(_text(server, "signals", lang="ur"))
    assert body["signals"][0]["kind"] == "TREND_EXIT" and "SPUS" in body["summary"]
    assert _tool(QuantOsMcp(None), "signals")["isError"]
    assert _tool(server, "signals", lang="fr")["isError"]


def test_calculators(env: dict[str, Any]) -> None:
    server = env["server"]
    rr = _text(server, "reward_risk", entry=100, stop=95, target=110)
    assert rr.startswith("reward:risk 2.00R") and "break-even win rate 33.3%" in rr
    assert _tool(server, "reward_risk", entry=100, stop=105, target=110)["isError"]
    due = _text(server, "zakat", cash=1000, investments=600, debts_due=100, nisab=500)
    assert "net zakatable wealth 1500.00" in due and "= 37.50" in due and "not a fatwa" in due
    assert "zakat 0" in _text(server, "zakat", cash=100, nisab=500)
    pure = _text(server, "purification", dividend=10, noncompliant_revenue=2, total_revenue=100)
    assert pure.startswith("purify 0.20 of the 10.00 dividend (2.00%)")
    per_share = _text(
        server, "purification", noncompliant_income=50, shares_outstanding=1000, shares_held=10
    )
    assert per_share.startswith("purify 0.5000")
    assert _tool(server, "purification")["isError"]
    assert "needs a 100.00% gain" in _text(server, "drawdown_recovery", drawdown=0.5)
    assert "needs a 25.00% gain" in _text(server, "drawdown_recovery", peak=100, current=80)
    assert _tool(server, "drawdown_recovery", drawdown=1.5)["isError"]
    fx = _text(server, "currency_convert", amount=10, **{"from": "usd", "to": "PKR"})
    assert fx == "10 USD = 2805.00 PKR at 280.5 PKR/USD"
    explicit = _text(server, "currency_convert", amount=2, rate=0.5, **{"from": "EUR", "to": "GBP"})
    assert explicit.startswith("2 EUR = 1.00 GBP")
    broken = _tool(server, "currency_convert", amount=1, **{"from": "EUR", "to": "JPY"})
    assert broken["isError"] and "rates unavailable" in broken["content"][0]["text"]
    none = QuantOsMcp(None)
    assert _tool(none, "currency_convert", amount=1, **{"from": "USD", "to": "PKR"})["isError"]
    bad_number = _tool(server, "reward_risk", entry="abc", stop=1, target=2)
    assert bad_number["isError"]


def test_knowledge_ask_multilingual(env: dict[str, Any]) -> None:
    server = env["server"]
    english = _text(server, "knowledge_ask", question="what is zakat nisab")
    assert "Zakat is 2.5%" in english and "zakat.md - Zakat basics" in english
    urdu = _text(server, "knowledge_ask", question="زکوٰۃ کا نصاب کیا ہے؟")
    assert "Ollama" in urdu and "zakat.md" in urdu
    refused = _text(server, "knowledge_ask", question="کیا میں سونا خریدوں؟")
    assert "zakat.md" not in refused
    assert _tool(QuantOsMcp(None), "knowledge_ask", question="zakat")["isError"]
    assert _tool(server, "knowledge_ask", question="zakat", lang="fr")["isError"]


def test_unconfigured_state_tools_are_tool_errors_not_crashes() -> None:
    server = QuantOsMcp(None)
    for name in ("bot_status", "explain_decision", "screening_results"):
        result = _tool(server, name)
        assert result["isError"] is True and "configured" in result["content"][0]["text"]


def test_no_advanced_tool_mutates_state(env: dict[str, Any]) -> None:
    tmp: Path = env["tmp"]
    server = env["server"]
    before = (_dump(tmp / "core.db"), _dump(tmp / "board.db"), (tmp / "core.json").read_bytes())
    calls: dict[str, dict[str, object]] = {
        "bot_status": {},
        "explain_decision": {"limit": 20},
        "screening_results": {},
        "signals": {"lang": "ar"},
        "reward_risk": {"entry": 10, "stop": 9, "target": 12},
        "zakat": {"cash": 1000, "nisab": 500},
        "purification": {"dividend": 1, "noncompliant_revenue": 1, "total_revenue": 10},
        "drawdown_recovery": {"drawdown": 0.2},
        "currency_convert": {"amount": 1, "from": "USD", "to": "PKR"},
        "knowledge_ask": {"question": "zakat nisab"},
    }
    assert set(calls) == {t["name"] for t in ADVANCED_TOOLS}
    for name, arguments in calls.items():
        assert _tool(server, name, **arguments)["isError"] is False, name
    after = (_dump(tmp / "core.db"), _dump(tmp / "board.db"), (tmp / "core.json").read_bytes())
    assert after == before


def test_build_server_never_creates_a_state_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.db"
    server = build_server(tmp_path / "no-docs", None, "50", "balanced", state_db=missing)
    assert not missing.exists() and server.store is None


def test_cached_fx_rates_reads_the_saved_table_without_fetching(tmp_path: Path) -> None:
    rates = cached_fx_rates(tmp_path / "fx.json")
    if rates is None:
        pytest.skip("ops.fx not installed")
    with pytest.raises(RuntimeError):  # no saved table and no download (FxUnavailable)
        rates("USD", "PKR")
    stamp = date(2026, 11, 21).isoformat() + "T00:00:00+00:00"
    (tmp_path / "fx.json").write_text(
        json.dumps({"rates": {"USD": "1", "PKR": "280"}, "as_of": stamp, "fetched_at": stamp}),
        "utf-8",
    )
    assert rates("USD", "PKR") == Decimal("280")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fx.json"]
