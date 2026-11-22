# ruff: noqa: E501  (the embedded HTML page keeps its own line lengths)
"""The dashboard's single page: overview, charts, depth, quant metrics and the setup guide."""

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Halal Desk</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;750&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{--bg:#eef1f3;--panel:#fff;--ink:#1b2a33;--soft:#5b6b74;--line:#d5dde1;--up:#1f7a6b;--down:#b4442e;--act:#3b4ba8;--gold:#9a7622;--r:6px}
@media (prefers-color-scheme:dark){:root{--bg:#15212a;--panel:#1c2b35;--ink:#e3e9ec;--soft:#9aabb4;--line:#2c3e49;--up:#4cb8a3;--down:#e0775f;--act:#8f9cf0;--gold:#d8b35a}}
*{box-sizing:border-box}html,body{margin:0}
body{background:var(--bg);color:var(--ink);font:15px/1.5 "Public Sans",system-ui,sans-serif;font-variant-numeric:tabular-nums}
.app{display:grid;grid-template-columns:200px 1fr;min-height:100vh}
nav{padding:20px 14px;border-right:1px solid var(--line);position:sticky;top:0;height:100vh}
nav h1{font-size:19px;font-weight:750;margin:0 0 4px}nav p{color:var(--soft);font-size:13px;margin:0 0 20px}
nav button{display:block;width:100%;text-align:left;background:none;border:0;color:var(--ink);font:inherit;padding:9px 10px;border-radius:var(--r);cursor:pointer}
nav button[aria-current=page]{background:var(--panel);font-weight:600;box-shadow:inset 3px 0 0 var(--act)}
nav button:focus-visible,main button:focus-visible,select:focus-visible{outline:2px solid var(--act);outline-offset:2px}
main{padding:24px clamp(16px,3vw,36px);max-width:1200px;width:100%}
section[hidden]{display:none}
h2{font-size:24px;font-weight:750;margin:0 0 4px}h3{font-size:16px;margin:0 0 10px}
.lede{color:var(--soft);margin:0 0 20px;max-width:70ch}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));background:var(--panel);border:1px solid var(--line);border-radius:var(--r);overflow:hidden;margin-bottom:20px}
.kpi{padding:14px 16px;box-shadow:inset -1px -1px 0 var(--line)}.kpi b{display:block;font-size:22px;font-weight:750}.kpi span{color:var(--soft);font-size:13px}
.pos{color:var(--up)}.neg{color:var(--down)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:16px;margin-bottom:20px}
.chart{height:320px}.chart.sm{height:140px}
.ladder{display:flex;gap:0;overflow-x:auto}
.rung{flex:1;min-width:110px;padding:12px;border-left:4px solid var(--line)}
.rung strong{display:block;font-size:20px}.rung small{color:var(--soft)}
.rung.hit{border-left-color:var(--gold)}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}th{color:var(--soft);font-weight:600}
code{background:var(--bg);padding:1px 5px;border-radius:4px;font-size:13px}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
select,.btn{font:inherit;padding:7px 10px;border-radius:var(--r);border:1px solid var(--line);background:var(--panel);color:var(--ink)}
.btn.danger{background:var(--down);color:#fff;border-color:var(--down);cursor:pointer}
.legend{display:flex;gap:14px;color:var(--soft);font-size:13px}.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}
ol.steps{padding-left:20px;max-width:75ch}ol.steps li{margin-bottom:14px}
.note{border-left:4px solid var(--gold);padding:8px 12px;background:var(--panel);max-width:75ch}
.empty{color:var(--soft);padding:24px 0}
@media (max-width:760px){.app{grid-template-columns:1fr}nav{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:4px;align-items:center}nav h1,nav p{width:100%}nav p{margin-bottom:8px}nav button{width:auto}nav button[aria-current=page]{box-shadow:inset 0 -3px 0 var(--act)}.chart{height:260px}}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style></head><body><div class="app">
<nav aria-label="Sections"><h1>Halal Desk</h1><p id="mode">Paper trading</p>
<button data-tab="overview" aria-current="page">Overview</button>
<button data-tab="charts">Charts</button>
<button data-tab="depth">Order book</button>
<button data-tab="quant">Quant metrics</button>
<button data-tab="guide">Setup guide</button>
</nav><main>

<section id="overview"><h2>Account</h2><p class="lede">Refreshes every 15 seconds from the bot's state file.</p>
<div class="kpis" id="kpis"></div>
<div class="panel"><h3>Where trade ideas stop</h3><p class="lede">Every idea passes these checks in order. A number is how many ideas each check stopped.</p><div class="ladder" id="ladder"></div></div>
<div class="panel"><h3>Equity</h3><div class="chart" id="eq"></div><h3 style="margin-top:14px">Drawdown from the last high</h3><div class="chart sm" id="dd"></div></div>
<div class="panel"><h3>Open positions</h3><div id="positions"></div></div>
<div class="panel"><h3>Recent events</h3><div id="events"></div></div>
<div class="panel"><h3>Kill switch</h3><p class="lede">Stops all new trades and closes open positions on the next step. Needs your dashboard token.</p>
<div class="toolbar"><input id="tok" class="btn" type="password" placeholder="Dashboard token" aria-label="Dashboard token" autocomplete="off">
<button class="btn danger" id="kill">Turn kill switch on</button><button class="btn" id="unkill">Turn kill switch off</button><span id="ks"></span></div></div>
</section>

<section id="charts" hidden><h2>Charts</h2><p class="lede">Daily bars from the <code>data/ohlc</code> folder, with 50 and 200-day averages, Bollinger bands (20 days, 2 deviations) and RSI (14).</p>
<div class="toolbar"><label for="sym">Symbol</label><select id="sym"></select>
<span class="legend"><span><i style="background:#3b4ba8"></i>SMA 50</span><span><i style="background:#9a7622"></i>SMA 200</span><span><i style="background:#8a9aa3"></i>Bollinger</span></span></div>
<div class="panel"><div class="chart" id="px" style="height:420px"></div><h3 style="margin-top:14px">RSI (14): above 70 stretched up, below 30 stretched down</h3><div class="chart sm" id="rsi"></div></div>
</section>

<section id="depth" hidden><h2>Order book depth</h2><p class="lede">Live crypto order book from Alpaca market data. Green is buyers waiting below the price, red is sellers waiting above. A wide gap in the middle means trading costs more. Shows orders within 0.5% of the price.</p>
<div class="toolbar"><label for="dsym">Pair</label><select id="dsym"><option>BTC/USD</option><option>ETH/USD</option></select><button class="btn" id="dref">Refresh</button><span id="spread"></span></div>
<div class="panel"><svg id="book" viewBox="0 0 800 320" width="100%" role="img" aria-label="Cumulative order book depth"></svg></div>
</section>

<section id="quant" hidden><h2>Quant metrics</h2><p class="lede">Computed from the bot's equity curve and closed trades. Blank means not enough history yet.</p>
<div class="panel"><table><thead><tr><th>Metric</th><th>Value</th><th>Formula</th><th>What it tells you</th></tr></thead><tbody id="qt"></tbody></table></div>
<div class="note">Backtests in this project (2005 to 2026) found no signal that beats diversified holding after costs. Read these numbers as risk control, not as a forecast of profit.</div>
</section>

<section id="guide" hidden><h2>Setup guide</h2><p class="lede">From a fresh Windows laptop to a running paper bot. Full version: <code>docs/SETUP_GUIDE.md</code>.</p>
<ol class="steps">
<li><b>Install Python 3.12</b> from python.org, ticking "Add to PATH".</li>
<li><b>Install the bot.</b> In PowerShell, in the project folder: <code>python -m venv .venv</code>, then <code>.venv\\Scripts\\pip install -e ".[dev,ml]"</code>.</li>
<li><b>Check it works:</b> <code>.venv\\Scripts\\python -m pytest -q</code> should end with "passed".</li>
<li><b>Get Alpaca paper keys.</b> Sign up at alpaca.markets, open the Paper dashboard, generate keys. Turn on "no shorting" in the paper account settings.</li>
<li><b>Choose a config.</b> Copy <code>examples\\paper_core.json</code> (Shariah ETFs) to <code>paper_core.json</code>. Add each fund's screening ratios under <code>financials</code>; without them the Shariah check blocks the fund.</li>
<li><b>Set your keys</b> for this window: <code>$env:APCA_API_KEY_ID="..."</code> and <code>$env:APCA_API_SECRET_KEY="..."</code>.</li>
<li><b>Run one step:</b> <code>.venv\\Scripts\\python -m quant_os_core.ops.run_paper --config paper_core.json --once</code>. Schedule it daily after the US close with Task Scheduler.</li>
<li><b>Open this dashboard:</b> set <code>$env:QUANT_OS_DASHBOARD_TOKEN</code> to a long random string, then run <code>.venv\\Scripts\\python -m quant_os_core.ops.dashboard --state state\\core.db --data data\\ohlc</code>.</li>
</ol>
<div class="note">Paper only. Live trading stays locked behind the two-key interlock. Futures, margin and short selling stay blocked.</div>
</section>
</main></div>
<script>
const $=id=>document.getElementById(id);const LC=window.LightweightCharts;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const pct=x=>x==null?"\\u2013":(x*100).toFixed(1)+"%";const num=x=>x==null?"\\u2013":Number(x).toFixed(2);
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const GATES=["plan","edge","sizing","cost_guard","shariah","protections","governor","council","risk"];
const GATE_NAMES={plan:"Plan check",edge:"Beats costs",sizing:"Size",cost_guard:"Cost guard",shariah:"Shariah",protections:"Protections",governor:"Trade limits",council:"Analysts",risk:"Risk engine"};
const charts={};
function mk(id,h){if(!LC)return null;const el=$(id);el.innerHTML="";const c=LC.createChart(el,{height:h||el.clientHeight,layout:{background:{color:"transparent"},textColor:css("--soft"),fontFamily:"Public Sans"},grid:{vertLines:{color:css("--line")},horzLines:{color:css("--line")}},rightPriceScale:{borderColor:css("--line")},timeScale:{borderColor:css("--line")}});new ResizeObserver(()=>c.applyOptions({width:el.clientWidth})).observe(el);return c}
function noLib(id){$(id).innerHTML='<p class="empty">Charts need internet once to load the chart library (unpkg.com).</p>'}
const day=iso=>iso.slice(0,10);
function dedupe(rows){const m=new Map();rows.forEach(r=>m.set(r.time,r));return [...m.values()]}
function state(s){
 $("mode").textContent=s.kill_switch?"Kill switch ON":"Paper trading";
 const m=s.metrics||{};const last=s.equity.length?s.equity[s.equity.length-1][1]:null;
 const k=[["Equity",last==null?"\\u2013":"$"+last.toFixed(2),""],["Total return",pct(m.total_return),m.total_return>=0?"pos":"neg"],["Worst drop",pct(s.max_drawdown),"neg"],["Sharpe",num(m.sharpe),""],["Win rate",pct(s.win_rate),""],["Open positions",String(s.positions.length),""]];
 $("kpis").innerHTML=k.map(([l,v,c])=>`<div class="kpi"><b class="${c}">${esc(v)}</b><span>${esc(l)}</span></div>`).join("");
 const b=s.blocked_by||{};$("ladder").innerHTML=GATES.filter(g=>g in b||["shariah","risk","protections"].includes(g)).map(g=>`<div class="rung ${b[g]?"hit":""}"><strong>${b[g]||0}</strong><small>${GATE_NAMES[g]}</small></div>`).join("");
 if(LC&&s.equity.length>1){charts.eq=charts.eq||mk("eq");charts.dd=charts.dd||mk("dd");
  if(!charts.eqs){charts.eqs=charts.eq.addAreaSeries({lineColor:css("--act"),topColor:"rgba(59,75,168,.25)",bottomColor:"rgba(59,75,168,0)"});charts.dds=charts.dd.addAreaSeries({lineColor:css("--down"),topColor:"rgba(180,68,46,0)",bottomColor:"rgba(180,68,46,.3)",invertFilledArea:true})}
  charts.eqs.setData(dedupe(s.equity.map(([t,v])=>({time:day(t),value:v}))));charts.dds.setData(dedupe(s.equity.map(([t],i)=>({time:day(t),value:s.drawdown[i]*100}))));
 }else if(!s.equity.length){$("eq").innerHTML='<p class="empty">No equity yet. Run one paper step to start the curve.</p>'}else if(!LC)noLib("eq");
 $("positions").innerHTML=s.positions.length?`<table><thead><tr><th>Symbol</th><th>Quantity</th><th>Entry</th><th>Stop</th><th>Since</th></tr></thead><tbody>${s.positions.map(p=>`<tr><td>${esc(p.symbol)}</td><td>${esc(p.quantity)}</td><td>${esc(p.entry)}</td><td>${esc(p.stop)}</td><td>${esc(day(p.since))}</td></tr>`).join("")}</tbody></table>`:'<p class="empty">No open positions.</p>';
 $("events").innerHTML=s.events.length?`<table><tbody>${s.events.slice(0,12).map(e=>`<tr><td>${esc(String(e.time||"").slice(0,16))}</td><td>${esc(e.level)}</td><td>${esc(e.message)}</td></tr>`).join("")}</tbody></table>`:'<p class="empty">No events yet.</p>';
 $("qt").innerHTML=Object.entries(s.formulas||{}).map(([n,[f,why]])=>{const v=m[n];const pc=["total_return","cagr","volatility","max_drawdown","win_rate","expectancy"].includes(n);return `<tr><td>${esc(n.replace("_"," "))}</td><td>${pc?pct(v):num(v)}</td><td><code>${esc(f)}</code></td><td>${esc(why)}</td></tr>`}).join("");
}
async function refresh(){try{state(await (await fetch("/api/state")).json())}catch(e){$("mode").textContent="Dashboard cannot read the state file"}}
async function loadSymbols(){const s=await (await fetch("/api/symbols")).json();$("sym").innerHTML=s.map(x=>`<option>${esc(x)}</option>`).join("")||"<option disabled>No CSV files in the data folder</option>";if(s.length){$("sym").value=s.includes("SPUS")?"SPUS":s[0];drawBars()}}
async function drawBars(){if(!LC){noLib("px");return}const r=await fetch("/api/bars?symbol="+encodeURIComponent($("sym").value));if(!r.ok)return;const d=(await r.json()).bars;
 const c=mk("px",420),cr=mk("rsi",140);const cs=c.addCandlestickSeries({upColor:css("--up"),downColor:css("--down"),borderVisible:false,wickUpColor:css("--up"),wickDownColor:css("--down")});cs.setData(d.map(b=>({time:b.time,open:b.open,high:b.high,low:b.low,close:b.close})));
 const line=(k,col,w)=>{const s=c.addLineSeries({color:col,lineWidth:w,priceLineVisible:false,lastValueVisible:false});s.setData(d.filter(b=>b[k]!=null).map(b=>({time:b.time,value:b[k]})))};
 line("sma50","#3b4ba8",2);line("sma200","#9a7622",2);line("bb_upper","#8a9aa3",1);line("bb_lower","#8a9aa3",1);
 const rs=cr.addLineSeries({color:css("--act"),lineWidth:1.5,priceLineVisible:false});rs.setData(d.filter(b=>b.rsi!=null).map(b=>({time:b.time,value:b.rsi})));
 [70,30].forEach(v=>rs.createPriceLine({price:v,color:css("--soft"),lineStyle:2,lineWidth:1,axisLabelVisible:true}));c.timeScale().fitContent();cr.timeScale().fitContent()}
async function drawBook(){const svg=$("book");$("spread").textContent="Loading\\u2026";
 const r=await fetch("/api/depth?symbol="+encodeURIComponent($("dsym").value));const d=await r.json();
 if(!r.ok){$("spread").textContent=d.error||"Order book unavailable";svg.innerHTML="";return}
 const all=[...d.bids,...d.asks];if(!all.length){$("spread").textContent="Empty book";return}
 const px=all.map(p=>p[0]),qty=all.map(p=>p[1]);const x0=Math.min(...px),x1=Math.max(...px),q1=Math.max(...qty);
 const X=p=>40+(p-x0)/(x1-x0||1)*740,Y=q=>290-q/q1*260;
 const path=(side,col)=>{if(!side.length)return"";const pts=side.map(([p,q])=>`${X(p)},${Y(q)}`);const base=`${X(side[side.length-1][0])},290 ${X(side[0][0])},290`;return `<polygon points="${pts.join(" ")} ${base}" fill="${col}" fill-opacity=".25" stroke="${col}" stroke-width="2"/>`};
 svg.innerHTML=`<line x1="40" y1="290" x2="780" y2="290" stroke="${css("--line")}"/>${path(d.bids,css("--up"))}${path(d.asks,css("--down"))}<text x="40" y="310" fill="${css("--soft")}" font-size="12">${x0.toFixed(2)}</text><text x="780" y="310" fill="${css("--soft")}" font-size="12" text-anchor="end">${x1.toFixed(2)}</text><text x="40" y="20" fill="${css("--soft")}" font-size="12">cumulative size ${q1.toFixed(3)}</text>`;
 $("spread").textContent=d.spread==null?"":`Spread ${d.spread.toFixed(2)} (${d.spread_bps.toFixed(1)} basis points)`}
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>{document.querySelectorAll("nav button").forEach(x=>x.removeAttribute("aria-current"));b.setAttribute("aria-current","page");document.querySelectorAll("main section").forEach(s=>s.hidden=s.id!==b.dataset.tab);if(b.dataset.tab==="depth")drawBook();if(b.dataset.tab==="charts"&&!$("sym").options.length)loadSymbols()});
$("sym").onchange=drawBars;$("dsym").onchange=drawBook;$("dref").onclick=drawBook;
async function kill(on){const r=await fetch("/api/kill",{method:"POST",headers:{"Authorization":"Bearer "+$("tok").value,"Content-Type":"application/json"},body:JSON.stringify({on,reason:"dashboard"})});$("ks").textContent=r.ok?(on?"Kill switch is on":"Kill switch is off"):"Wrong token: nothing changed";refresh()}
$("kill").onclick=()=>kill(true);$("unkill").onclick=()=>kill(false);
refresh();setInterval(refresh,15000);
</script></body></html>"""
