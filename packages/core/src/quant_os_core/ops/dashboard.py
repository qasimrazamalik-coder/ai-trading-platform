"""Local dashboard: equity, drawdown, quant metrics, charts, order-book depth, kill switch,
calculators, currency conversion, English/Urdu/Arabic labels, a knowledge assistant, the trading
journal, the trade and fee ledger with zakat and purification, a manual practice backtester, the
read-only rulebook and the settings panel.

    QUANT_OS_DASHBOARD_TOKEN=<random> python -m quant_os_core.ops.dashboard \
        --state state/trading.db --data data/ohlc --config paper_core.json

Serves on 127.0.0.1 only. Reading is open on localhost; the kill switch and every write to the
settings, journal or ledger need the token. Practice sessions live in memory and never reach the
order path.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import threading
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from quant_os_core.ops import analytics, calculators, i18n, rulebook
from quant_os_core.ops.dashboard_page import PAGE
from quant_os_core.ops.fx import FxService, FxUnavailable
from quant_os_core.ops.journal import Journal
from quant_os_core.ops.ledger import Ledger, noncompliant_ratios
from quant_os_core.ops.manual_backtest import ManualBacktester
from quant_os_core.ops.settings_store import SettingsError, SettingsStore, settings_path_for
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
    config_path: Path | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Request handler class. ``fx`` defaults to rates cached next to the state database
    (``<state dir>/fx_rates.json``); ``ask`` defaults to ``knowledge_asker(knowledge_dir)``.
    ``config_path`` (the paper config) supplies each fund's non-compliant revenue ratio for
    purification; the settings JSON is written next to it (or next to the state database)."""

    rates = fx or FxService(cache_path=Path(store.path).parent / "fx_rates.json")
    answer = ask or knowledge_asker(knowledge_dir)
    ledger = Ledger(Path(store.path))
    journal = Journal(Path(store.path), ledger)
    settings = SettingsStore(settings_path_for(config_path, Path(store.path)))
    practice = ManualBacktester(data_dir)

    def ratios() -> dict[str, Decimal]:
        if config_path is None:
            return {}
        try:
            return noncompliant_ratios(json.loads(config_path.read_text("utf-8")))
        except (OSError, ValueError, AttributeError):
            return {}

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
            elif url.path == "/api/rulebook":
                self._json(200, rulebook.as_json((query.get("lang") or ["en"])[0]))
            elif url.path == "/api/settings":
                self._json(200, settings.describe())
            elif url.path == "/api/journal":
                self._json(
                    200,
                    {"entries": [e.as_dict() for e in journal.entries()], "stats": journal.stats()},
                )
            elif url.path == "/api/ledger":
                year = _year_of(query)
                if year is None:
                    self._json(400, {"error": "year must be a whole number"})
                else:
                    self._json(200, ledger.summary(year, ratios()))
            elif url.path == "/api/ledger/export.csv":
                body = ledger.export_csv().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="ledger.csv"')
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
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

        def _authorized(self) -> bool:
            """Bearer-token check shared by every write; sends 401 when it fails."""

            header = self.headers.get("Authorization", "")
            ok = len(token) >= 16 and hmac.compare_digest(
                header.encode(), f"Bearer {token}".encode()
            )
            if not ok:
                self._send(401, b'{"error":"unauthorized"}', "application/json")
            return ok

        def _run(self, action: Callable[[Mapping[str, Any]], Any], *, auth: bool) -> None:
            if auth and not self._authorized():
                return
            payload = self._body()
            if payload is None:
                return
            if not isinstance(payload, dict):
                self._json(400, {"error": "send a JSON object"})
                return
            try:
                self._json(200, action(payload))
            except SettingsError as exc:
                self._json(400, {"error": "invalid settings", "errors": exc.errors})
            except KeyError as exc:
                self._json(404, {"error": str(exc.args[0]) if exc.args else "not found"})
            except (ValueError, ArithmeticError, TypeError) as exc:
                self._json(400, {"error": str(exc)})

        def _routes(self) -> dict[str, tuple[Callable[[Mapping[str, Any]], Any], bool]]:
            """Path -> (action, needs token). Practice and zakat maths change no records."""

            def journal_add(p: Mapping[str, Any]) -> dict[str, Any]:
                entry_id = journal.add(
                    _when(p),
                    _text(p, "symbol"),
                    _text(p, "setup"),
                    _text(p, "reason"),
                    _text(p, "emotion"),
                    _flag(p, "rule_followed"),
                    screenshot=_text(p, "screenshot"),
                    lesson=_text(p, "lesson"),
                    tags=_items(p.get("tags")),
                    fill_ids=[
                        int(calculators.parse_decimal({"id": i}, "id"))
                        for i in _items(p.get("fill_ids"))
                    ],
                )
                return {"id": entry_id}

            def fill(p: Mapping[str, Any]) -> dict[str, Any]:
                return {
                    "id": ledger.record_fill(
                        _when(p),
                        _text(p, "symbol"),
                        _text(p, "side"),
                        calculators.parse_decimal(p, "quantity"),
                        calculators.parse_decimal(p, "price"),
                        calculators.parse_decimal(p, "fee", "0"),
                        _text(p, "currency", "USD"),
                        _text(p, "note"),
                    )
                }

            def cash(p: Mapping[str, Any]) -> dict[str, Any]:
                return {
                    "id": ledger.record_cash(
                        _when(p),
                        _text(p, "kind"),
                        calculators.parse_decimal(p, "amount"),
                        _text(p, "currency", "USD"),
                        _text(p, "note"),
                    )
                }

            def dividend(p: Mapping[str, Any]) -> dict[str, Any]:
                return {
                    "id": ledger.record_dividend(
                        _when(p),
                        _text(p, "symbol"),
                        calculators.parse_decimal(p, "amount"),
                        _text(p, "currency", "USD"),
                        _text(p, "note"),
                    )
                }

            def payment(p: Mapping[str, Any]) -> dict[str, Any]:
                return {
                    "id": ledger.record_payment(
                        _when(p),
                        _text(p, "kind"),
                        int(calculators.parse_decimal(p, "year")),
                        calculators.parse_decimal(p, "amount"),
                        _text(p, "currency", "USD"),
                        _text(p, "symbol"),
                        _text(p, "note"),
                    )
                }

            def zakat(p: Mapping[str, Any]) -> dict[str, Any]:
                nisab = p.get("nisab")
                return ledger.zakat(
                    int(calculators.parse_decimal(p, "year")),
                    calculators.parse_decimal(p, "zakatable_wealth"),
                    nisab=None if nisab in (None, "") else calculators.parse_decimal(p, "nisab"),
                    gold_price_per_gram=None
                    if p.get("gold_price_per_gram") in (None, "")
                    else calculators.parse_decimal(p, "gold_price_per_gram"),
                )

            def practise(action: str) -> Callable[[Mapping[str, Any]], Any]:
                return lambda p: practice.act(_text(p, "session"), action, p)

            return {
                "/api/settings": (lambda p: (settings.save(p), settings.describe())[1], True),
                "/api/journal": (journal_add, True),
                "/api/ledger/fill": (fill, True),
                "/api/ledger/cash": (cash, True),
                "/api/ledger/dividend": (dividend, True),
                "/api/ledger/payment": (payment, True),
                "/api/ledger/zakat": (zakat, False),
                "/api/bt/start": (practice.start, False),
                **{f"/api/bt/{a}": (practise(a), False) for a in ("next", "buy", "sell", "view")},
            }

        def do_POST(self) -> None:
            if self.path.startswith("/api/calc/"):
                self._calc(self.path.removeprefix("/api/calc/"))
                return
            if self.path == "/api/ask":
                self._ask()
                return
            route = self._routes().get(self.path)
            if route is not None:
                self._run(route[0], auth=route[1])
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


def _when(payload: Mapping[str, Any]) -> datetime:
    """``payload["time"]`` as an aware datetime (ISO 8601 with offset), or now in UTC."""

    raw = payload.get("time")
    if raw in (None, ""):
        return datetime.now(UTC)
    if not isinstance(raw, str) or len(raw) > 40:
        raise ValueError("time must be an ISO 8601 date and time")
    when = datetime.fromisoformat(raw)
    if when.tzinfo is None:
        raise ValueError("time needs a timezone, for example 2026-11-22T10:00:00+05:00")
    return when


def _text(payload: Mapping[str, Any], key: str, default: str = "") -> str:
    raw = payload.get(key, default)
    if raw is None:
        return default
    if not isinstance(raw, str):
        raise ValueError(f"{key} must be text")
    return raw


def _flag(payload: Mapping[str, Any], key: str) -> bool:
    raw = payload.get(key)
    if isinstance(raw, bool):
        return raw
    if raw in ("yes", "true", "on", "1"):
        return True
    if raw in ("no", "false", "off", "0", None, ""):
        return False
    raise ValueError(f"{key} must be yes or no")


def _items(raw: Any) -> list[str]:
    """A list, or a comma-separated string from a text box."""

    if raw in (None, ""):
        return []
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, list) and len(raw) <= 100:
        return [str(item) for item in raw]
    raise ValueError("expected a list or comma-separated text")


def _year_of(query: Mapping[str, list[str]]) -> int | None:
    raw = (query.get("year") or [""])[0]
    if not raw:
        return datetime.now(UTC).year
    return int(raw) if raw.isdigit() and 1900 <= int(raw) <= 2200 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quant OS local dashboard")
    parser.add_argument("--state", type=Path, default=Path("state/trading.db"))
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--data", type=Path, default=Path("data/ohlc"), help="OHLC CSV folder")
    parser.add_argument(
        "--config", type=Path, default=None, help="paper config (fund financials, settings file)"
    )
    args = parser.parse_args(argv)
    token = os.environ.get("QUANT_OS_DASHBOARD_TOKEN", "")
    if len(token) < 16:
        print(
            "set QUANT_OS_DASHBOARD_TOKEN (16+ characters) to enable the kill switch",
            file=sys.stderr,
        )
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        make_handler(StateStore(args.state), token, args.data, config_path=args.config),
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
