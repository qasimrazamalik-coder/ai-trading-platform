"""Read-only MCP server (stdio) exposing Quant OS knowledge and calculators to AI assistants.

Dual-era, per the MCP versioning page (https://modelcontextprotocol.io/specification/latest):

* modern clients (2026-07-28) send ``io.modelcontextprotocol/protocolVersion`` and
  ``clientCapabilities`` in every request's ``_meta``; ``server/discover`` is supported;
* legacy clients (2025-11-25 and earlier) open with ``initialize``.

Guardrail: no tool places, modifies or cancels orders, reads credentials or writes files.
Trading decisions stay in the rule pipeline; this server only explains and calculates.

Run:  python -m quant_os_core.mcp_server --docs docs
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, TextIO

from quant_os_core.instruments import AssetClass, Instrument, InstrumentId, InstrumentType
from quant_os_core.knowledge.rag import KnowledgeBase
from quant_os_core.shariah import CryptoAsset, ShariahGate, ShariahProfile

SERVER_INFO = {"name": "quant-os", "version": "0.3.0"}
MODERN_VERSIONS = ("2026-07-28",)
LEGACY_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
META_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CAPS = "io.modelcontextprotocol/clientCapabilities"
META_SERVER = "io.modelcontextprotocol/serverInfo"
INSTRUCTIONS = (
    "Quant OS knowledge and calculators. Read-only: no tool can trade. Use knowledge_search "
    "before answering trading, quant-math, psychology or halal-rule questions, and cite the "
    "returned sources."
)

JSON = dict[str, Any]


class ToolError(Exception):
    """A tool-level failure, reported to the model inside the result (isError)."""


def _decimal(args: Mapping[str, Any], key: str) -> Decimal:
    try:
        value = Decimal(str(args[key]))
    except (KeyError, InvalidOperation) as exc:
        raise ToolError(f"'{key}' must be a number") from exc
    if not value.is_finite():
        raise ToolError(f"'{key}' must be finite")
    return value


_PROFILES = {
    "pk_hanafi": ShariahProfile(),
    "owner": ShariahProfile.spot_crypto_and_day_trading(),
}

TOOLS: list[JSON] = [
    {
        "name": "knowledge_search",
        "description": "Search the Quant OS knowledge base (trading basics, quant formulas, "
        "strategies and backtesting, psychology, halal rules). Returns cited passages.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "k": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
            },
            "required": ["query"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "shariah_check",
        "description": "Check whether an instrument type is allowed by the Shariah gate. "
        "Profiles: pk_hanafi (crypto off) or owner (spot crypto and day trading on).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "asset_class": {"type": "string", "enum": [a.value for a in AssetClass]},
                "instrument_type": {"type": "string", "enum": [t.value for t in InstrumentType]},
                "profile": {"type": "string", "enum": list(_PROFILES), "default": "owner"},
                "token_category": {"type": "string", "description": "crypto only, e.g. layer1"},
            },
            "required": ["asset_class", "instrument_type"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "position_size",
        "description": "Fixed-fractional size: equity x risk_fraction / (entry - stop), "
        "capped at 20% of equity, no leverage.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "equity": {"type": "number", "exclusiveMinimum": 0},
                "entry": {"type": "number", "exclusiveMinimum": 0},
                "stop": {"type": "number", "exclusiveMinimum": 0},
                "risk_fraction": {"type": "number", "default": 0.01, "maximum": 0.05},
            },
            "required": ["equity", "entry", "stop"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "growth_projection",
        "description": "Periods needed to grow start to target at a constant return per "
        "period: ln(target/start) / ln(1 + r). Illustrative only; returns are never constant.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "start": {"type": "number", "exclusiveMinimum": 0},
                "target": {"type": "number", "exclusiveMinimum": 0},
                "return_per_period": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["start", "target", "return_per_period"],
        },
        "annotations": {"readOnlyHint": True},
    },
]


class QuantOsMcp:
    def __init__(self, kb: KnowledgeBase | None) -> None:
        self.kb = kb
        self.legacy_version: str | None = None
        self._tools: dict[str, Callable[[Mapping[str, Any]], str]] = {
            "knowledge_search": self._knowledge_search,
            "shariah_check": self._shariah_check,
            "position_size": self._position_size,
            "growth_projection": self._growth_projection,
        }

    # ---- tools -------------------------------------------------------------------------
    def _knowledge_search(self, args: Mapping[str, Any]) -> str:
        if self.kb is None:
            raise ToolError("knowledge base not loaded; start the server with --docs")
        query = str(args.get("query", "")).strip()
        if len(query) < 2:
            raise ToolError("'query' is required")
        k = int(args.get("k", 4))
        hits = self.kb.search(query, max(1, min(8, k)))
        if not hits:
            return "No passages found."
        return "\n\n".join(
            f"[{i}] {h.chunk.source} - {h.chunk.heading}\n{h.chunk.text}"
            for i, h in enumerate(hits, start=1)
        )

    def _shariah_check(self, args: Mapping[str, Any]) -> str:
        try:
            asset = AssetClass(str(args["asset_class"]))
            kind = InstrumentType(str(args["instrument_type"]))
            profile = _PROFILES[str(args.get("profile", "owner"))]
        except (KeyError, ValueError) as exc:
            raise ToolError("unknown asset_class, instrument_type or profile") from exc
        gate = ShariahGate(profile)
        instrument = Instrument(
            instrument_id=InstrumentId(venue="CHECK", symbol="X"),
            asset_class=asset,
            instrument_type=kind,
            base_currency="XXX",
            quote_currency="USD",
            price_precision=2,
            quantity_precision=4,
        )
        decision = gate.check_instrument(instrument)
        category = args.get("token_category")
        if decision.allowed and asset is AssetClass.CRYPTO and category:
            decision = gate.screen_crypto(CryptoAsset("X", str(category)))
        verdict = "allowed" if decision.allowed else "blocked"
        return f"{verdict} ({decision.rule_id}): {decision.reason}"

    def _position_size(self, args: Mapping[str, Any]) -> str:
        equity, entry, stop = (_decimal(args, k) for k in ("equity", "entry", "stop"))
        risk = Decimal(str(args.get("risk_fraction", "0.01")))
        if min(equity, entry, stop) <= 0 or not Decimal("0") < risk <= Decimal("0.05"):
            raise ToolError("equity, entry, stop must be positive; risk_fraction in (0, 0.05]")
        if stop >= entry:
            raise ToolError("long-only: stop must be below entry")
        risk_qty = equity * risk / (entry - stop)
        cap_qty = equity * Decimal("0.20") / entry
        qty = min(risk_qty, cap_qty)
        capped = " (capped at 20% of equity)" if cap_qty < risk_qty else ""
        return (
            f"quantity {qty:.6f}{capped}; cost {qty * entry:.2f}; "
            f"loss at stop {qty * (entry - stop):.2f}"
        )

    def _growth_projection(self, args: Mapping[str, Any]) -> str:
        start, target, rate = (_decimal(args, k) for k in ("start", "target", "return_per_period"))
        if min(start, target, rate) <= 0:
            raise ToolError("all inputs must be positive")
        if target <= start:
            return "0 periods: target already reached"
        periods = math.log(float(target / start)) / math.log1p(float(rate))
        return f"{periods:.1f} periods at a constant {float(rate):.2%} per period"

    # ---- protocol ----------------------------------------------------------------------
    def _result(self, body: JSON) -> JSON:
        return {"resultType": "complete", **body, "_meta": {META_SERVER: SERVER_INFO}}

    @staticmethod
    def _error(msg_id: Any, code: int, message: str, data: Any = None) -> JSON:
        error: JSON = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": msg_id, "error": error}

    def handle(self, message: Mapping[str, Any]) -> JSON | None:
        msg_id = message.get("id")
        method = message.get("method")
        if not isinstance(method, str):
            return self._error(msg_id, -32600, "invalid request")
        if msg_id is None:
            return None  # notification, e.g. notifications/initialized
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return self._error(msg_id, -32602, "params must be an object")

        if method == "initialize":
            requested = str(params.get("protocolVersion", ""))
            # Legacy negotiation: echo a supported version, else offer our latest legacy one.
            self.legacy_version = requested if requested in LEGACY_VERSIONS else LEGACY_VERSIONS[0]
            result = {
                "protocolVersion": self.legacy_version,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
                "instructions": INSTRUCTIONS,
            }
            return {"jsonrpc": "2.0", "id": msg_id, "result": result}

        meta = params.get("_meta") or {}
        version = meta.get(META_VERSION) if isinstance(meta, dict) else None
        if version is not None:
            if version not in MODERN_VERSIONS:
                supported = [*MODERN_VERSIONS, *LEGACY_VERSIONS[:1]]
                data = {"supported": supported, "requested": version}
                return self._error(msg_id, -32022, "Unsupported protocol version", data)
            if META_CAPS not in meta:
                return self._error(msg_id, -32602, f"missing _meta '{META_CAPS}'")
        elif self.legacy_version is None and method != "ping":
            return self._error(
                msg_id, -32602, f"missing _meta '{META_VERSION}' (or send initialize first)"
            )

        body: JSON
        if method == "server/discover":
            body = {
                "supportedVersions": [*MODERN_VERSIONS, *LEGACY_VERSIONS],
                "capabilities": {"tools": {}},
                "instructions": INSTRUCTIONS,
            }
            return {"jsonrpc": "2.0", "id": msg_id, "result": self._result(body)}
        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": self._result({})}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": self._result({"tools": TOOLS})}
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            tool = self._tools.get(str(name))
            if tool is None or not isinstance(arguments, dict):
                return self._error(msg_id, -32602, f"unknown tool or bad arguments: {name}")
            try:
                text, is_error = tool(arguments), False
            except ToolError as exc:
                text, is_error = str(exc), True
            body = {"content": [{"type": "text", "text": text}], "isError": is_error}
            return {"jsonrpc": "2.0", "id": msg_id, "result": self._result(body)}
        return self._error(msg_id, -32601, f"method not found: {method}")

    def serve(self, stdin: TextIO, stdout: TextIO) -> None:
        """Newline-delimited JSON-RPC over stdio; logs go to stderr only."""

        for line in stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                reply: JSON | None = self._error(None, -32700, "parse error")
            else:
                try:
                    reply = (
                        self.handle(message)
                        if isinstance(message, dict)
                        else self._error(None, -32600, "invalid request")
                    )
                except Exception as exc:  # never crash the server on one bad request
                    print(f"quant-os mcp: internal error: {exc!r}", file=sys.stderr)
                    reply = self._error(message.get("id"), -32603, "internal error")
            if reply is not None:
                stdout.write(json.dumps(reply) + "\n")
                stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quant OS read-only MCP server (stdio)")
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args(argv)
    kb = KnowledgeBase.from_directory(args.docs) if args.docs.is_dir() else None
    if kb is None:
        print(f"quant-os mcp: docs folder {args.docs} not found; search disabled", file=sys.stderr)
    QuantOsMcp(kb).serve(sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
