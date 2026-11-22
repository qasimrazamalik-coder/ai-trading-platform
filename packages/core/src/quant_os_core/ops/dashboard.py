"""Local dashboard: equity, drawdown, quant metrics, charts, order-book depth, kill switch,
calculators, currency conversion, English/Urdu/Arabic labels and a knowledge assistant.

    QUANT_OS_DASHBOARD_TOKEN=<random> python -m quant_os_core.ops.dashboard \
        --state state/trading.db --data data/ohlc

Serves on 127.0.0.1 only. Reading is open on localhost; the kill switch needs the token.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import threading
from collections import Counter
from collections.abc import Callable
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from quant_os_core.ops import analytics, calculators, i18n
from quant_os_core.ops.dashboard_page import PAGE
from quant_os_core.ops.fx import FxService, FxUnavailable
from quant_os_core.ops.state import StateStore

MAX_BODY = 10_000
MAX_QUESTION = 1_000
Ask = Callable[[str, str], dict[str, Any]]


def knowledge_asker(knowledge_dir: Path) -> Ask:
    """Lazy question answering over ``knowledge_dir`` for ``POST /api/ask``.

    Prefers ``knowledge.rag.answer_multilingual(question, lang)`` when that function exists.
    Otherwise builds a ``KnowledgeChatbot`` once, on the first question: extractive answers,
    or a local Ollama model when ``QUANT_OS_OLLAMA_MODEL`` is set. Errors propagate; the
    handler turns them into 503.
    """

    lock = threading.Lock()
    holder: dict[str, Any] = {}

    def ask(question: str, lang: str) -> dict[str, Any]:
        from quant_os_core.knowledge import rag

        multilingual = getattr(rag, "answer_multilingual", None)
        if callable(multilingual):
            result = multilingual(question, lang)
            if not isinstance(result, dict) or "answer" not in result:
                raise RuntimeError("answer_multilingual returned no answer")
            return {"answer": str(result["answer"]), "sources": list(result.get("sources", []))}
        with lock:
            if "bot" not in holder:
                model = os.environ.get("QUANT_OS_OLLAMA_MODEL", "").strip()
                generate = rag.ollama_generate(model) if model else None
                kb = rag.KnowledgeBase.from_directory(knowledge_dir)
                holder["bot"] = rag.KnowledgeChatbot(kb, generate)
            bot = holder["bot"]
        reply = bot.ask(question)
        return {
            "answer": reply.text,
            "sources": [f"{c.source} - {c.heading}" for c in reply.sources],
            "grounded": reply.grounded,
            "refused": reply.refused,
        }

    return ask


def snapshot(store: StateStore) -> dict[str, Any]:
    curve = store.equity_curve(2000)
    peak = Decimal("0")
    drawdown = []
    for _, equity in curve:
        peak = max(peak, equity)
        drawdown.append(float(equity / peak - 1) if peak > 0 else 0.0)
    decisions = store.recent("decisions", 500)
    blocked = Counter(d["blocked_by"] for d in decisions if not d["approved"])
    closed = store.recent("closed_trades", 200)
    wins = sum(1 for t in closed if Decimal(t["profit_ratio"]) > 0)
    ratios = [float(t["profit_ratio"]) for t in closed]
    return {
        "metrics": analytics.metrics([(t.isoformat(), float(e)) for t, e in curve], ratios),
        "formulas": analytics.FORMULAS,
        "kill_switch": store.kill_switch_on(),
        "equity": [[t.isoformat(), float(e)] for t, e in curve],
        "drawdown": drawdown,
        "max_drawdown": min(drawdown, default=0.0),
        "positions": [
            {
                "symbol": s.position.plan.symbol,
                "quantity": str(s.position.quantity),
                "entry": str(s.position.plan.entry_price),
                "stop": str(s.position.stop),
                "target": str(s.position.plan.profit_target),
                "since": s.position.entry_time.isoformat(),
            }
            for s in store.open_positions()
        ],
        "pending": [
            {
                "symbol": p["symbol"],
                "quantity": str(p["quantity"]),
                "submitted": p["submitted"].isoformat(),
            }
            for p in store.pending()
        ],
        "closed": closed[:30],
        "win_rate": wins / len(closed) if closed else None,
        "blocked_by": dict(blocked.most_common()),
        "decisions": decisions[:30],
        "events": store.recent("events", 40),
    }


def make_handler(
    store: StateStore,
    token: str,
    data_dir: Path = Path("data/ohlc"),
    depth_fetch: analytics.DepthFetcher = analytics.alpaca_crypto_book,
    *,
    fx: FxService | None = None,
    knowledge_dir: Path = Path("docs/knowledge"),
    ask: Ask | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Request handler class. ``fx`` defaults to rates cached next to the state database
    (``<state dir>/fx_rates.json``); ``ask`` defaults to ``knowledge_asker(knowledge_dir)``."""

    rates = fx or FxService(cache_path=Path(store.path).parent / "fx_rates.json")
    answer = ask or knowledge_asker(knowledge_dir)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, json.dumps(payload, default=str).encode(), "application/json")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            query = parse_qs(url.query)
            symbol = (query.get("symbol") or [""])[0]
            if url.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self._json(200, snapshot(store))
            elif url.path == "/api/symbols":
                self._json(200, analytics.available_symbols(data_dir))
            elif url.path == "/api/bars":
                try:
                    self._json(200, analytics.bars(data_dir, symbol))
                except (ValueError, OSError, KeyError) as exc:
                    self._json(400, {"error": str(exc)})
            elif url.path == "/api/depth":
                try:
                    self._json(200, analytics.depth(symbol, depth_fetch))
                except ValueError as exc:
                    self._json(400, {"error": str(exc)})
                except Exception:  # network or upstream failure: tell the page, keep serving
                    self._json(502, {"error": "order book unavailable (check internet)"})
            elif url.path == "/api/i18n":
                self._json(200, i18n.bundle((query.get("lang") or ["en"])[0]))
            elif url.path == "/api/fx":
                self._fx(lambda: rates.table((query.get("base") or ["USD"])[0][:5]))
            elif url.path == "/api/fx/convert":
                self._fx(lambda: self._convert(query))
            else:
                self._send(404, b"{}", "application/json")

        def _fx(self, build: Callable[[], dict[str, Any]]) -> None:
            try:
                self._json(200, build())
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            except FxUnavailable as exc:
                self._json(503, {"error": str(exc)})

        def _convert(self, query: dict[str, list[str]]) -> dict[str, Any]:
            first = {k: v[0] for k, v in query.items() if v}
            amount = calculators.parse_decimal(first, "amount")
            source, target = first.get("from", "USD")[:5], first.get("to", "PKR")[:5]
            table = rates.rates()
            rate = table.rate(source, target)
            return {
                "amount": str(amount),
                "from": source.upper(),
                "to": target.upper(),
                "rate": str(rate),
                "converted": str(
                    table.convert(amount, source, target).quantize(Decimal("0.00000001"))
                ),
                "as_of": table.as_of.isoformat(),
                "stale": table.stale,
            }

        def _body(self) -> Any:
            """Parsed JSON body, or ``None`` after an error response was sent."""

            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY:
                status = 413 if length > MAX_BODY else 400
                self._json(status, {"error": f"body must be JSON under {MAX_BODY} bytes"})
                return None
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._json(400, {"error": "bad json"})
                return None

        def _calc(self, name: str) -> None:
            if name not in calculators.CALCULATORS:
                self._json(404, {"error": "unknown calculator"})
                return
            payload = self._body()
            if payload is None:
                return
            try:
                self._json(200, calculators.run(name, payload))
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            except ArithmeticError:
                self._json(400, {"error": "numbers out of range"})

        def _ask(self) -> None:
            payload = self._body()
            if payload is None:
                return
            question = payload.get("question") if isinstance(payload, dict) else None
            if not isinstance(question, str) or not question.strip():
                self._json(400, {"error": "question is required"})
                return
            if len(question) > MAX_QUESTION:
                self._json(400, {"error": f"question is longer than {MAX_QUESTION} characters"})
                return
            lang = i18n.normalize(str(payload.get("lang", "en")))
            try:
                self._json(200, answer(question.strip(), lang))
            except Exception:  # knowledge files missing or model broken: say so, keep serving
                self._json(
                    503,
                    {
                        "error": "Assistant unavailable: check that docs/knowledge exists and, "
                        "if QUANT_OS_OLLAMA_MODEL is set, that Ollama is running."
                    },
                )

        def do_POST(self) -> None:
            if self.path.startswith("/api/calc/"):
                self._calc(self.path.removeprefix("/api/calc/"))
                return
            if self.path == "/api/ask":
                self._ask()
                return
            if self.path != "/api/kill":
                self._send(404, b"{}", "application/json")
                return
            header = self.headers.get("Authorization", "")
            if len(token) < 16 or not hmac.compare_digest(
                header.encode(), f"Bearer {token}".encode()
            ):
                self._send(401, b'{"error":"unauthorized"}', "application/json")
                return
            length = min(int(self.headers.get("Content-Length") or 0), 10_000)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, b'{"error":"bad json"}', "application/json")
                return
            store.set_kill_switch(bool(payload.get("on")), str(payload.get("reason", "dashboard")))
            self._send(
                200,
                json.dumps({"kill_switch": store.kill_switch_on()}).encode(),
                "application/json",
            )

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quant OS local dashboard")
    parser.add_argument("--state", type=Path, default=Path("state/trading.db"))
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--data", type=Path, default=Path("data/ohlc"), help="OHLC CSV folder")
    args = parser.parse_args(argv)
    token = os.environ.get("QUANT_OS_DASHBOARD_TOKEN", "")
    if len(token) < 16:
        print(
            "set QUANT_OS_DASHBOARD_TOKEN (16+ characters) to enable the kill switch",
            file=sys.stderr,
        )
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(StateStore(args.state), token, args.data)
    )
    print(f"dashboard on http://127.0.0.1:{args.port}", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
