import io
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest
from quant_os_core.knowledge import KnowledgeBase, KnowledgeChatbot, chunk_markdown
from quant_os_core.mcp_server import QuantOsMcp

DOCS = Path(__file__).resolve().parents[4] / "docs" / "knowledge"
MODERN = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {},
}


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase.from_directory(DOCS)


def test_chunking_keeps_headings_and_splits_long_sections() -> None:
    text = "# Title\nintro\n## Part A\n" + "\n\n".join(["word " * 100] * 5) + "\n## Part B\nend"
    chunks = chunk_markdown("x.md", text, max_chars=1200)
    headings = [c.heading for c in chunks]
    assert headings[0] == "Title" and headings[-1] == "Part B"
    assert headings.count("Part A") >= 2  # long section split on blank lines


@pytest.mark.parametrize(
    ("query", "source"),
    [
        ("how do I size a position with a stop", "02-quant-math.md"),
        ("deflated sharpe probability of backtest overfitting", "02-quant-math.md"),
        ("revenge trading after a loss", "04-psychology.md"),
        ("are futures halal IIFA", "05-halal-rules.md"),
        ("look-ahead survivorship bias", "03-strategies-and-backtesting.md"),
        ("bid ask spread slippage", "01-trading-basics.md"),
    ],
)
def test_retrieval_finds_the_right_file(kb: KnowledgeBase, query: str, source: str) -> None:
    hits = kb.search(query, k=3)
    assert hits and source in {h.chunk.source for h in hits}


def _model(reply: str):  # type: ignore[no-untyped-def]
    def generate(messages: Sequence[dict[str, str]]) -> str:
        assert "[1]" in messages[1]["content"]  # sources are numbered in the prompt
        return reply

    return generate


def test_chatbot_guardrails(kb: KnowledgeBase) -> None:
    grounded = KnowledgeChatbot(kb, _model("<think>x</think>Risk 1% of equity per trade [1]."))
    answer = grounded.ask("how big should a position be relative to the stop?")
    assert answer.grounded and answer.text.startswith("Risk") and answer.sources
    uncited = KnowledgeChatbot(kb, _model("Just buy more."))
    assert not uncited.ask("position sizing with a stop").grounded
    bad_ref = KnowledgeChatbot(kb, _model("See [9]."))
    assert not bad_ref.ask("position sizing with a stop").grounded

    def offline(_m: Sequence[dict[str, str]]) -> str:
        raise ConnectionError("ollama down")

    fallback = KnowledgeChatbot(kb, offline).ask("position sizing with a stop")
    assert fallback.text.startswith("Model unavailable") and fallback.sources
    assert KnowledgeChatbot(kb).ask("Should I buy Tesla today?").refused
    assert KnowledgeChatbot(kb).ask("zzqx unknown term").refused


def _call(server: QuantOsMcp, method: str, params: dict[str, object], msg_id: int = 1) -> dict:  # type: ignore[type-arg]
    reply = server.handle({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
    assert reply is not None
    return reply


def test_mcp_modern_discover_list_and_call(kb: KnowledgeBase) -> None:
    server = QuantOsMcp(kb)
    discover = _call(server, "server/discover", {"_meta": MODERN})["result"]
    assert discover["resultType"] == "complete" and "2026-07-28" in discover["supportedVersions"]
    names = {t["name"] for t in _call(server, "tools/list", {"_meta": MODERN})["result"]["tools"]}
    assert names == {
        "knowledge_search",
        "shariah_check",
        "position_size",
        "growth_projection",
        "deposit_projection",
    }

    def tool(name: str, **arguments: object) -> dict:  # type: ignore[type-arg]
        params = {"name": name, "arguments": arguments, "_meta": MODERN}
        return _call(server, "tools/call", params)["result"]  # type: ignore[no-any-return]

    assert (
        "05-halal-rules.md" in tool("knowledge_search", query="futures IIFA")["content"][0]["text"]
    )
    blocked = tool("shariah_check", asset_class="crypto", instrument_type="perpetual")
    assert blocked["content"][0]["text"].startswith("blocked (R2)")
    meme = tool(
        "shariah_check", asset_class="crypto", instrument_type="spot", token_category="meme"
    )
    assert meme["content"][0]["text"].startswith("blocked (R13)")
    ok = tool(
        "shariah_check", asset_class="crypto", instrument_type="spot", token_category="layer1"
    )
    assert ok["content"][0]["text"].startswith("allowed")
    size = tool("position_size", equity=50, entry=100, stop=95)["content"][0]["text"]
    assert size.startswith("quantity 0.100000")
    years = tool("growth_projection", start=50, target=1_000_000, return_per_period=0.2)
    expected = math.log(20_000) / math.log(1.2)
    assert f"{expected:.1f} periods" in years["content"][0]["text"]
    bad = tool("position_size", equity=50, entry=100, stop=120)
    assert bad["isError"] is True


def test_mcp_version_and_meta_errors_and_legacy_handshake(kb: KnowledgeBase) -> None:
    server = QuantOsMcp(kb)
    old = {"io.modelcontextprotocol/protocolVersion": "1900-01-01"}
    assert _call(server, "tools/list", {"_meta": old})["error"]["code"] == -32022
    no_caps = {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}
    assert _call(server, "tools/list", {"_meta": no_caps})["error"]["code"] == -32602
    assert _call(server, "tools/list", {})["error"]["code"] == -32602
    assert _call(server, "nope", {"_meta": MODERN})["error"]["code"] == -32601
    init = _call(server, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert "tools" in _call(server, "tools/list", {})["result"]  # legacy session now works


def test_mcp_stdio_loop_survives_bad_input(kb: KnowledgeBase) -> None:
    lines = [
        "not json",
        json.dumps(
            {"jsonrpc": "2.0", "id": 7, "method": "server/discover", "params": {"_meta": MODERN}}
        ),
        "",
    ]
    out = io.StringIO()
    QuantOsMcp(kb).serve(io.StringIO("\n".join(lines)), out)
    replies = [json.loads(r) for r in out.getvalue().splitlines()]
    assert replies[0]["error"]["code"] == -32700 and replies[1]["id"] == 7


def test_mcp_deposit_projection(kb: KnowledgeBase) -> None:
    server = QuantOsMcp(kb)
    args = {"start": 50, "monthly_deposit": 50, "annual_return": 0.1, "months": 36}
    params = {"name": "deposit_projection", "arguments": args, "_meta": MODERN}
    text = _call(server, "tools/call", params)["result"]["content"][0]["text"]
    assert text.startswith("after 36 months: balance") and "deposited 1850" in text


def test_retrieval_eval_set_top3_accuracy(kb: KnowledgeBase) -> None:
    path = Path(__file__).with_name("eval_questions.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    hits = sum(1 for r in rows if r["source"] in {h.chunk.source for h in kb.search(r["q"], k=3)})
    assert len(rows) >= 40 and hits / len(rows) >= 0.9


def test_hybrid_retrieval_fuses_embeddings_and_survives_offline_embedder() -> None:
    from quant_os_core.knowledge import Chunk

    chunks = [
        Chunk("a.md", "Kelly", "kelly fraction bet sizing growth"),
        Chunk("b.md", "Halal", "futures are not permissible postponement"),
    ]

    def embed(texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] if ("kelly" in t.lower() or "bet" in t) else [0.0, 1.0] for t in texts]

    kb = KnowledgeBase(chunks, embed=embed)
    assert kb.search("how much should I bet", k=1)[0].chunk.source == "a.md"
    state = {"up": True}

    def flaky(texts: Sequence[str]) -> list[list[float]]:
        if not state["up"]:
            raise ConnectionError("ollama down")
        return embed(texts)

    kb2 = KnowledgeBase(chunks, embed=flaky)
    state["up"] = False
    assert kb2.search("futures permissible", k=1)[0].chunk.source == "b.md"
