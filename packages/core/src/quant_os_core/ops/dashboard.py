# ruff: noqa: E501  (the embedded HTML page keeps its own line lengths)
"""Local dashboard: equity, drawdown, positions, blocked-by reasons, events and a kill switch.

    QUANT_OS_DASHBOARD_TOKEN=<random> python -m quant_os_core.ops.dashboard --state state/trading.db

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
    return {
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


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quant OS - paper desk</title>
<style>
:root{--bg:#0d0f12;--panel:#14171c;--line:#232830;--ink:#e7e9ee;--mute:#8b93a1;--up:#3ccf91;
--down:#ef6a6a;--accent:#d8b45a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{display:flex;align-items:center;justify-content:space-between;padding:20px 28px;
border-bottom:1px solid var(--line)}h1{font-size:15px;letter-spacing:.08em;text-transform:uppercase;
margin:0;font-weight:600}h1 span{color:var(--accent)}main{display:grid;gap:16px;padding:20px 28px;
grid-template-columns:repeat(12,1fr)}section{background:var(--panel);border:1px solid var(--line);
border-radius:10px;padding:16px 18px}h2{font-size:12px;color:var(--mute);font-weight:500;
letter-spacing:.06em;text-transform:uppercase;margin:0 0 12px}.kpi{font-size:26px;font-weight:600;
font-variant-numeric:tabular-nums}.s3{grid-column:span 3}.s4{grid-column:span 4}.s8{grid-column:span 8}
.s12{grid-column:span 12}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
td,th{padding:6px 4px;border-bottom:1px solid var(--line);text-align:left;font-weight:400}
th{color:var(--mute);font-size:12px}.up{color:var(--up)}.down{color:var(--down)}
button{background:transparent;color:var(--ink);border:1px solid var(--line);border-radius:8px;
padding:8px 14px;font:inherit;cursor:pointer}button.on{border-color:var(--down);color:var(--down)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.bar{height:8px;background:var(--accent);border-radius:4px}.muted{color:var(--mute)}
svg{width:100%;height:220px;display:block}
@media(max-width:900px){.s3,.s4,.s8{grid-column:span 12}}
</style></head><body>
<header><h1>Quant OS <span>paper desk</span></h1>
<button id="kill" aria-live="polite">Kill switch</button></header>
<main>
<section class="s3"><h2>Equity</h2><div class="kpi" id="eq">-</div></section>
<section class="s3"><h2>Max drawdown</h2><div class="kpi" id="dd">-</div></section>
<section class="s3"><h2>Win rate</h2><div class="kpi" id="wr">-</div></section>
<section class="s3"><h2>Open positions</h2><div class="kpi" id="np">-</div></section>
<section class="s8"><h2>Equity curve</h2><svg id="chart" role="img" aria-label="Equity curve"></svg></section>
<section class="s4"><h2>Why trades were blocked</h2><div id="blocked"></div></section>
<section class="s12"><h2>Positions</h2><table id="pos"></table></section>
<section class="s8"><h2>Recent decisions</h2><table id="dec"></table></section>
<section class="s4"><h2>Events</h2><table id="ev"></table></section>
</main>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function chart(points){const svg=$("chart");if(points.length<2){svg.innerHTML="";return}
const w=svg.clientWidth||600,h=220,v=points.map(p=>p[1]),lo=Math.min(...v),hi=Math.max(...v),
r=hi-lo||1,d=v.map((y,i)=>`${i?"L":"M"}${(i/(v.length-1)*w).toFixed(1)},${(h-8-(y-lo)/r*(h-16)).toFixed(1)}`).join("");
svg.setAttribute("viewBox",`0 0 ${w} ${h}`);
svg.innerHTML=`<path d="${d}" fill="none" stroke="var(--accent)" stroke-width="1.6"/>`}
async function load(){const s=await (await fetch("/api/state")).json();
const last=s.equity.length?s.equity[s.equity.length-1][1]:null;
$("eq").textContent=last==null?"-":last.toFixed(2);
$("dd").textContent=(s.max_drawdown*100).toFixed(1)+"%";
$("wr").textContent=s.win_rate==null?"-":(s.win_rate*100).toFixed(0)+"%";
$("np").textContent=s.positions.length;chart(s.equity);
const k=$("kill");k.className=s.kill_switch?"on":"";k.textContent=s.kill_switch?"Kill switch ON - resume":"Kill switch";
const tot=Object.values(s.blocked_by).reduce((a,b)=>a+b,0)||1;
$("blocked").innerHTML=Object.entries(s.blocked_by).map(([k,n])=>`<div>${esc(k)} <span class="muted">${n}</span><div class="bar" style="width:${n/tot*100}%"></div></div>`).join("")||'<span class="muted">nothing blocked yet</span>';
$("pos").innerHTML="<tr><th>Symbol</th><th>Qty</th><th>Entry</th><th>Stop</th><th>Target</th></tr>"+s.positions.map(p=>`<tr><td>${esc(p.symbol)}</td><td>${esc(p.quantity)}</td><td>${esc(p.entry)}</td><td>${esc(p.stop)}</td><td>${esc(p.target)}</td></tr>`).join("");
$("dec").innerHTML="<tr><th>Time</th><th>Symbol</th><th>Result</th></tr>"+s.decisions.map(d=>`<tr><td class="muted">${esc(d.time.slice(0,16).replace("T"," "))}</td><td>${esc(d.symbol)}</td><td class="${d.approved?"up":"down"}">${d.approved?"approved "+esc(d.quantity):"blocked: "+esc(d.blocked_by)}</td></tr>`).join("");
$("ev").innerHTML=s.events.map(e=>`<tr><td class="${e.level==="critical"?"down":"muted"}">${esc(e.message)}</td></tr>`).join("")}
$("kill").onclick=async()=>{const t=prompt("Dashboard token");if(!t)return;
const on=!$("kill").classList.contains("on");
await fetch("/api/kill",{method:"POST",headers:{"Authorization":"Bearer "+t,"Content-Type":"application/json"},body:JSON.stringify({on,reason:"dashboard"})});load()};
load();setInterval(load,15000);
</script></body></html>"""


def make_handler(store: StateStore, token: str) -> type[BaseHTTPRequestHandler]:
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

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                body = json.dumps(snapshot(store), default=str).encode()
                self._send(200, body, "application/json")
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
    args = parser.parse_args(argv)
    token = os.environ.get("QUANT_OS_DASHBOARD_TOKEN", "")
    if len(token) < 16:
        print(
            "set QUANT_OS_DASHBOARD_TOKEN (16+ characters) to enable the kill switch",
            file=sys.stderr,
        )
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(StateStore(args.state), token)
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
