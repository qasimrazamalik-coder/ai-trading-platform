import json
import threading
import urllib.error
import urllib.request
from collections.abc import Sequence
from datetime import timedelta
from decimal import Decimal
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from quant_os_core.agents.board import AgentBoard, AnalystRole, BoardAnalyst
from quant_os_core.agents.board_worker import run_local_shift
from quant_os_core.agents.council import Stance
from quant_os_core.agents.trade_plan import TradePlan
from quant_os_core.mcp_board_tools import BoardTools
from quant_os_core.mcp_http import make_handler
from quant_os_core.mcp_server import QuantOsMcp

MODERN = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {},
}
TOKEN = "t" * 40


@pytest.fixture
def board(tmp_path: Path) -> AgentBoard:
    return AgentBoard(tmp_path / "state" / "board.db")


def _plan(symbol: str = "BTC/USD") -> TradePlan:
    return TradePlan(
        symbol=symbol,
        entry_price=Decimal("100"),
        profit_target=Decimal("110"),
        stop_loss=Decimal("96"),
        invalidation_condition="daily close below the 50-day average",
        confidence=Decimal("0.6"),
        justification="test",
    )


def test_board_tasks_reports_and_council_seat(board: AgentBoard) -> None:
    task = board.post_task("btc/usd", AnalystRole.NEWS_SENTIMENT, "Any material news this week?")
    assert board.open_tasks()[0]["symbol"] == "BTC/USD"
    seat = BoardAnalyst(board, AnalystRole.NEWS_SENTIMENT)
    assert seat.review(_plan()).stance is Stance.NEUTRAL  # nothing yet: no opinion
    board.submit_report(
        symbol="BTC/USD",
        role=AnalystRole.NEWS_SENTIMENT,
        source="Claude",
        stance=Stance.BULLISH,
        conviction=Decimal("0.6"),
        rationale="ETF inflows steady, no exchange incidents reported this week.",
        task_id=task,
    )
    assert board.open_tasks() == []  # answering closes the task
    report = seat.review(_plan())
    assert report.stance is Stance.BULLISH and report.analyst == "claude:news_sentiment"
    with pytest.raises(ValueError):
        board.submit_report(
            symbol="BTC/USD",
            role=AnalystRole.BEAR_ADVOCATE,
            source="chatgpt",
            stance=Stance.BULLISH,
            conviction=Decimal("0.5"),
            rationale="the bear advocate cannot be bullish at all",
        )
    with pytest.raises(ValueError):
        board.submit_report(
            symbol="BTC/USD",
            role=AnalystRole.TECHNICAL,
            source="chatgpt",
            stance=Stance.BEARISH,
            conviction=Decimal("0.5"),
            rationale="too short",
        )
    stale = BoardAnalyst(board, AnalystRole.NEWS_SENTIMENT, max_age=timedelta(seconds=0))
    assert stale.review(_plan()).stance is Stance.NEUTRAL


def test_local_worker_answers_valid_and_skips_invalid(board: AgentBoard) -> None:
    board.post_task("AAPL", AnalystRole.TECHNICAL, "Is the trend intact above the 200-day?")
    board.post_task("AAPL", AnalystRole.BEAR_ADVOCATE, "Argue against buying AAPL now.")
    board.post_task("AAPL", AnalystRole.MACRO, "How do rates affect this name now?")
    replies = iter(
        [
            '<think>x</think>{"analyst": "q", "stance": "bullish", "conviction": 0.55, '
            '"rationale": "Price above rising 200-day average with higher lows.", '
            '"red_flag": false}',
            '{"analyst": "q", "stance": "bullish", "conviction": 0.9, '
            '"rationale": "I refuse to be bearish about this company at all.", "red_flag": false}',
            "not json at all",
        ]
    )

    def generate(_m: Sequence[dict[str, str]]) -> str:
        return next(replies)

    result = run_local_shift(board, generate)
    assert (result.answered, result.skipped) == (1, 2)
    sources = {r.source for r in board.recent_reports("AAPL", timedelta(hours=1))}
    assert sources == {"qwen-local"}


def _server(board: AgentBoard) -> QuantOsMcp:
    return QuantOsMcp(None, BoardTools(board))


def _tool(server: QuantOsMcp, name: str, **arguments: object) -> dict:  # type: ignore[type-arg]
    reply = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments, "_meta": MODERN},
        }
    )
    assert reply is not None
    return reply["result"]  # type: ignore[no-any-return]


PLAN = {
    "symbol": "BTC/USD",
    "asset_class": "crypto",
    "entry_price": 100,
    "stop_loss": 96,
    "profit_target": 110,
    "confidence": 0.6,
    "invalidation_condition": "daily close below the 50-day average",
    "token_category": "layer1",
}


def _report(server: QuantOsMcp, source: str, role: str, stance: str) -> None:
    result = _tool(
        server,
        "submit_analyst_report",
        symbol="BTC/USD",
        role=role,
        source=source,
        stance=stance,
        conviction=0.6,
        rationale=f"{source} view from the {role} seat with enough detail.",
    )
    assert not result["isError"], result


def test_bidirectional_flow_and_consensus_rule(board: AgentBoard) -> None:
    server = _server(board)
    names = {t["name"] for t in server.tools}
    assert {"get_research_tasks", "submit_trade_proposal", "evaluate_trade_plan"} <= names
    posted = _tool(
        server, "post_research_task", symbol="BTC/USD", role="technical", question="Trend check?"
    )
    assert posted["content"][0]["text"].startswith("task")
    tasks = json.loads(_tool(server, "get_research_tasks")["content"][0]["text"])
    assert tasks[0]["role"] == "technical"

    dry = json.loads(_tool(server, "evaluate_trade_plan", **PLAN)["content"][0]["text"])
    assert dry["verdict"].startswith(("APPROVED", "BLOCKED"))
    assert [s["stage"] for s in dry["stages"]][:3] == ["plan", "edge", "shariah"]

    blocked = _tool(server, "submit_trade_proposal", source="claude", **PLAN)
    assert blocked["isError"] and "2 different sources" in blocked["content"][0]["text"]

    _report(server, "claude", "technical", "bullish")
    _report(server, "chatgpt", "bear_advocate", "bearish")
    recorded = json.loads(
        _tool(server, "submit_trade_proposal", source="claude", **PLAN)["content"][0]["text"]
    )
    assert "proposal_id" in recorded
    listed = json.loads(_tool(server, "list_proposals")["content"][0]["text"])
    assert listed[0]["id"] == recorded["proposal_id"] and listed[0]["source"] == "claude"

    meme = json.loads(
        _tool(server, "evaluate_trade_plan", **{**PLAN, "token_category": "meme"})["content"][0][
            "text"
        ]
    )
    assert meme["verdict"] == "BLOCKED at shariah"
    red = _tool(
        server,
        "submit_analyst_report",
        symbol="BTC/USD",
        role="news_sentiment",
        source="chatgpt",
        stance="bearish",
        conviction=0.4,
        rationale="Exchange hack reported; custody risk unresolved.",
        red_flag=True,
    )
    assert not red["isError"]
    vetoed = json.loads(_tool(server, "evaluate_trade_plan", **PLAN)["content"][0]["text"])
    assert vetoed["verdict"] == "BLOCKED at council"
    bad = _tool(server, "evaluate_trade_plan", **{**PLAN, "asset_class": "future"})
    assert bad["isError"]


@pytest.fixture
def http_server(board: AgentBoard):  # type: ignore[no-untyped-def]
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(_server(board), TOKEN))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _post(url: str, body: object, headers: dict[str, str] | None = None) -> tuple[int, dict]:  # type: ignore[type-arg]
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", **(headers or {})}
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as err:
        raw = err.read()
        return err.code, json.loads(raw) if raw else {}


def test_http_transport_auth_and_protocol(http_server: str) -> None:
    discover = {"jsonrpc": "2.0", "id": 1, "method": "server/discover", "params": {"_meta": MODERN}}
    assert _post(f"{http_server}/mcp", discover)[0] == 401
    status, body = _post(f"{http_server}/mcp", discover, {"Authorization": f"Bearer {TOKEN}"})
    assert status == 200 and "2026-07-28" in body["result"]["supportedVersions"]
    status, body = _post(f"{http_server}/mcp/{TOKEN}", discover)  # secret-path mode
    assert status == 200
    assert _post(f"{http_server}/mcp/wrong", discover)[0] == 401
    status, body = _post(
        f"{http_server}/mcp/{TOKEN}", discover, {"MCP-Protocol-Version": "2025-11-25"}
    )
    assert status == 400 and body["error"]["code"] == -32020
    note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    assert _post(f"{http_server}/mcp/{TOKEN}", note)[0] == 202
    assert (
        _post(f"{http_server}/mcp/{TOKEN}", discover, {"Origin": "https://evil.example"})[0] == 403
    )
