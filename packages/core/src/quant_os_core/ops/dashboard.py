"""Local dashboard: equity, drawdown, quant metrics, charts, order-book depth, kill switch.

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
from collections import Counter
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from quant_os_core.ops import analytics
from quant_os_core.ops.dashboard_page import PAGE
from quant_os_core.ops.state import StateStore


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
) -> type[BaseHTTPRequestHandler]:
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
            symbol = (parse_qs(url.query).get("symbol") or [""])[0]
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
            else:
                self._send(404, b"{}", "application/json")

        def do_POST(self) -> None:
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
