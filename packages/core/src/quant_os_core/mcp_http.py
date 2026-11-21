"""Streamable HTTP transport for the Quant OS MCP server (standard library only).

Remote assistants need HTTPS: Claude custom connectors and ChatGPT developer-mode apps connect
from their own cloud, not from your laptop. Run this on 127.0.0.1 and expose it through an
HTTPS tunnel (e.g. ``cloudflared tunnel --url http://127.0.0.1:8765``).

Authentication (the token comes from the ``QUANT_OS_MCP_TOKEN`` environment variable; the
server refuses to start without one of at least 32 characters):

* ``Authorization: Bearer <token>`` on ``/mcp`` - Claude Code, Claude Desktop config, and
  Claude custom connectors that send static request headers;
* ``/mcp/<token>`` path - for clients limited to OAuth or no-auth (ChatGPT developer mode).
  The URL itself is the secret: never share it, rotate the token if it leaks, and prefer an
  OAuth gateway (Cloudflare Access, Auth0) in front of the tunnel for anything beyond testing.

POST carries one JSON-RPC message and gets one JSON response; notifications get 202.
GET (server-initiated streams) is not offered and returns 405, which the spec allows.
"""

from __future__ import annotations

import hmac
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from quant_os_core.mcp_server import META_VERSION, QuantOsMcp

MAX_BODY = 1_000_000
ALLOWED_ORIGINS = frozenset({"https://claude.ai", "https://chatgpt.com", "https://chat.openai.com"})


def make_handler(server: QuantOsMcp, token: str) -> type[BaseHTTPRequestHandler]:
    secret_path = f"/mcp/{token}"

    class Handler(BaseHTTPRequestHandler):
        server_version = "quant-os-mcp"

        def log_message(self, fmt: str, *args: Any) -> None:  # keep tokens out of logs
            path = self.path.split("/mcp/")[0] + ("/mcp/<secret>" if "/mcp/" in self.path else "")
            print(f"quant-os mcp http: {self.command} {path} {args[1:]}", file=sys.stderr)

        def _send(self, status: int, body: dict[str, Any] | None) -> None:
            data = b"" if body is None else json.dumps(body).encode()
            self.send_response(status)
            if body is not None:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            path = self.path.split("?", 1)[0].rstrip("/")
            if hmac.compare_digest(path.encode(), secret_path.encode()):
                return True
            header = self.headers.get("Authorization", "")
            return path == "/mcp" and hmac.compare_digest(
                header.encode(), f"Bearer {token}".encode()
            )

        def do_GET(self) -> None:
            self._send(405, {"error": "use POST"})

        def do_POST(self) -> None:
            origin = self.headers.get("Origin")
            if origin and origin not in ALLOWED_ORIGINS:
                self._send(403, {"error": "origin not allowed"})
                return
            if not self._authorized():
                self._send(401, {"error": "unauthorized"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                self._send(413 if length > MAX_BODY else 400, {"error": "bad body size"})
                return
            try:
                message = json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                self._send(400, server._error(None, -32700, "parse error"))
                return
            if not isinstance(message, dict):
                self._send(400, server._error(None, -32600, "send one JSON-RPC message"))
                return
            header_version = self.headers.get("MCP-Protocol-Version")
            meta = (message.get("params") or {}).get("_meta") or {}
            body_version = meta.get(META_VERSION) if isinstance(meta, dict) else None
            if header_version and body_version and header_version != body_version:
                self._send(400, server._error(message.get("id"), -32020, "header mismatch"))
                return
            try:
                reply = server.handle(message)
            except Exception as exc:  # one bad request must not kill the server
                print(f"quant-os mcp http: internal error {exc!r}", file=sys.stderr)
                reply = server._error(message.get("id"), -32603, "internal error")
            if reply is None:
                self._send(202, None)
                return
            status = 400 if reply.get("error", {}).get("code") in {-32602, -32022, -32600} else 200
            self._send(status, reply)

    return Handler


def serve_http(server: QuantOsMcp, host: str, port: int) -> int:
    token = os.environ.get("QUANT_OS_MCP_TOKEN", "")
    if len(token) < 32:
        print("set QUANT_OS_MCP_TOKEN to a random string of 32+ characters", file=sys.stderr)
        return 2
    httpd = ThreadingHTTPServer((host, port), make_handler(server, token))
    print(f"quant-os mcp: listening on http://{host}:{port}/mcp", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
