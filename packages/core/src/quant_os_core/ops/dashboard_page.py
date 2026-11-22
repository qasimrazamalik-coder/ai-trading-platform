# ruff: noqa: E501  (the embedded HTML page keeps its own line lengths)
"""The dashboard's single page: overview, charts, depth, quant metrics, calculators, assistant,
setup guide. Labels come from ``GET /api/i18n`` (English, Urdu, Arabic); the English text in the
markup is the fallback when scripts do not run."""

PAGE = """<!doctype html><html lang="en" dir="ltr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Halal Desk</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;750&family=Noto+Naskh+Arabic:wght@400;600;700&family=Noto+Nastaliq+Urdu:wght@400;600;700&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{--bg:#eef1f3;--panel:#fff;--ink:#1b2a33;--soft:#5b6b74;--line:#d5dde1;--up:#1f7a6b;--down:#b4442e;--act:#3b4ba8;--gold:#9a7622;--r:6px}
@media (prefers-color-scheme:dark){:root{--bg:#15212a;--panel:#1c2b35;--ink:#e3e9ec;--soft:#9aabb4;--line:#2c3e49;--up:#4cb8a3;--down:#e0775f;--act:#8f9cf0;--gold:#d8b35a}}
*{box-sizing:border-box}html,body{margin:0}
body{background:var(--bg);color:var(--ink);font:15px/1.5 "Public Sans",system-ui,sans-serif;font-variant-numeric:tabular-nums}
html[lang=ar] body{font-family:"Noto Naskh Arabic","Public Sans",system-ui,sans-serif;line-height:1.75}
html[lang=ur] body{font-family:"Noto Nastaliq Urdu","Noto Naskh Arabic","Public Sans",serif;line-height:2.1}
html[lang=ur] h2,html[lang=ur] h3,html[lang=ur] nav h1{line-height:2}
.app{display:grid;grid-template-columns:210px minmax(0,1fr);min-height:100vh}
nav{padding:20px 14px;border-inline-end:1px solid var(--line);position:sticky;top:0;height:100vh;overflow-y:auto}
nav h1{font-size:19px;font-weight:750;margin:0 0 4px}nav p{color:var(--soft);font-size:13px;margin:0 0 20px}
nav button{display:block;width:100%;text-align:start;background:none;border:0;color:var(--ink);font:inherit;padding:9px 10px;border-radius:var(--r);cursor:pointer}
nav button[aria-current=page]{background:var(--panel);font-weight:600;box-shadow:inset 3px 0 0 var(--act)}
[dir=rtl] nav button[aria-current=page]{box-shadow:inset -3px 0 0 var(--act)}
nav button:focus-visible,main button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--act);outline-offset:2px}
.prefs{margin-top:20px;display:grid;gap:8px;font-size:13px;color:var(--soft)}.prefs label{display:grid;gap:3px}.prefs select{width:100%}
#rate{font-size:12px;color:var(--soft);min-height:1em}
main{padding:24px clamp(16px,3vw,36px);max-width:1200px;width:100%;min-width:0}
section[hidden],[hidden]{display:none!important}
h2{font-size:24px;font-weight:750;margin:0 0 4px}h3{font-size:16px;margin:0 0 10px}
.lede{color:var(--soft);margin:0 0 20px;max-width:70ch}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,170px),1fr));background:var(--panel);border:1px solid var(--line);border-radius:var(--r);overflow:hidden;margin-bottom:20px}
.kpi{padding:14px 16px;box-shadow:inset -1px -1px 0 var(--line)}.kpi{min-width:0}.kpi b{display:block;font-size:clamp(17px,1.5vw,22px);font-weight:750;overflow-wrap:anywhere}.kpi span{color:var(--soft);font-size:13px}
.pos{color:var(--up)}.neg{color:var(--down)}
bdi.n,code{direction:ltr;unicode-bidi:isolate}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:16px;margin-bottom:20px;min-width:0}
.scroll{overflow-x:auto}
.chart{height:320px}.chart.sm{height:140px}
.ladder{display:flex;gap:0;overflow-x:auto}
.rung{flex:1;min-width:110px;padding:12px;border-inline-start:4px solid var(--line)}
.rung strong{display:block;font-size:20px}.rung small{color:var(--soft)}
.rung.hit{border-inline-start-color:var(--gold)}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:start;padding:7px 8px;border-bottom:1px solid var(--line)}th{color:var(--soft);font-weight:600}
code{background:var(--bg);padding:1px 5px;border-radius:4px;font-size:13px;overflow-wrap:anywhere}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
select,.btn,input{font:inherit;padding:7px 10px;border-radius:var(--r);border:1px solid var(--line);background:var(--panel);color:var(--ink);max-width:100%}
.btn{cursor:pointer}.btn.primary{background:var(--act);border-color:var(--act);color:#fff}
.btn.danger{background:var(--down);color:#fff;border-color:var(--down);cursor:pointer}
.grow{flex:1 1 220px;min-width:0}
.legend{display:flex;gap:14px;color:var(--soft);font-size:13px;flex-wrap:wrap}.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-inline-end:5px}
ol.steps{padding-inline-start:20px;max-width:75ch}ol.steps li{margin-bottom:14px}
.note{border-inline-start:4px solid var(--gold);padding:8px 12px;background:var(--panel);max-width:75ch}
.empty{color:var(--soft);padding:24px 0}
.calcs{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,330px),1fr));gap:16px;align-items:start}
.calcs .panel{margin:0}.calcs p{color:var(--soft);font-size:13px;margin:0 0 12px}
.fields{display:grid;gap:10px;margin-bottom:12px}.fields label{display:grid;gap:3px;font-size:13px;color:var(--soft)}.fields input,.fields select{width:100%}
.out table{margin-top:10px}.out td:last-child{text-align:end;font-weight:600}.err{color:var(--down)}
.mic{display:inline-flex;align-items:center;gap:6px}.mic[aria-pressed=true]{background:var(--down);border-color:var(--down);color:#fff}
#answer{white-space:pre-wrap;margin-top:12px;overflow-wrap:anywhere}#answer ul{white-space:normal;color:var(--soft);font-size:13px}
footer{color:var(--soft);font-size:13px;border-top:1px solid var(--line);padding-top:12px;margin-top:8px;max-width:75ch}
@media (max-width:760px){.app{grid-template-columns:minmax(0,1fr)}nav{position:static;height:auto;border-inline-end:0;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:4px;align-items:center}nav h1,nav p{width:100%}nav p{margin-bottom:8px}nav button{width:auto}nav button[aria-current=page],[dir=rtl] nav button[aria-current=page]{box-shadow:inset 0 -3px 0 var(--act)}.prefs{width:100%;grid-template-columns:1fr 1fr;margin-top:8px}#rate{grid-column:1/-1}.chart{height:260px}}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style></head><body><div class="app">
<nav aria-label="Sections" data-ia="nav_label"><h1 data-i="app_title">Halal Desk</h1><p id="mode">Paper trading</p>
<button data-tab="overview" aria-current="page" data-i="tab_overview">Overview</button>
<button data-tab="charts" data-i="tab_charts">Charts</button>
<button data-tab="depth" data-i="tab_depth">Order book</button>
<button data-tab="quant" data-i="tab_quant">Quant metrics</button>
<button data-tab="calc" data-i="tab_calc">Calculators</button>
<button data-tab="ask" data-i="tab_ask">Assistant</button>
<button data-tab="guide" data-i="tab_guide">Setup guide</button>
<div class="prefs"><label><span data-i="lang_label">Language</span><select id="lang"><option value="en" lang="en">English</option><option value="ur" lang="ur">اردو</option><option value="ar" lang="ar">العربية</option></select></label>
<label><span data-i="currency_label">Currency</span><select id="cur"><option>USD</option></select></label><div id="rate" aria-live="polite"></div></div>
</nav><main>

<section id="overview"><h2 data-i="h_account">Account</h2><p class="lede" data-i="lede_account">Refreshes every 15 seconds from the bot's state file.</p>
<div class="kpis" id="kpis"></div>
<div class="panel"><h3 data-i="h_ladder">Where trade ideas stop</h3><p class="lede" data-i="lede_ladder">Every idea passes these checks in order. A number is how many ideas each check stopped.</p><div class="ladder" id="ladder"></div></div>
<div class="panel"><h3 data-i="h_equity">Equity</h3><div class="chart" id="eq"></div><h3 style="margin-top:14px" data-i="h_drawdown">Drawdown from the last high</h3><div class="chart sm" id="dd"></div></div>
<div class="panel"><h3 data-i="h_positions">Open positions</h3><div id="positions" class="scroll"></div></div>
<div class="panel"><h3 data-i="h_events">Recent events</h3><div id="events" class="scroll"></div></div>
<div class="panel"><h3 data-i="h_kill">Kill switch</h3><p class="lede" data-i="lede_kill">Stops all new trades and closes open positions on the next step. Needs your dashboard token.</p>
<div class="toolbar"><input id="tok" class="btn" type="password" placeholder="Dashboard token" aria-label="Dashboard token" data-ip="token_ph" autocomplete="off">
<button class="btn danger" id="kill" data-i="btn_kill_on">Turn kill switch on</button><button class="btn" id="unkill" data-i="btn_kill_off">Turn kill switch off</button><span id="ks"></span></div></div>
</section>

<section id="charts" hidden><h2 data-i="h_charts">Charts</h2><p class="lede" data-ih="lede_charts_html">Daily bars from the <code>data/ohlc</code> folder, with 50 and 200-day averages, Bollinger bands (20 days, 2 deviations) and RSI (14).</p>
<div class="toolbar"><label for="sym" data-i="symbol_label">Symbol</label><select id="sym"></select>
<span class="legend"><span><i style="background:#3b4ba8"></i><span data-i="legend_sma50">SMA 50</span></span><span><i style="background:#9a7622"></i><span data-i="legend_sma200">SMA 200</span></span><span><i style="background:#8a9aa3"></i><span data-i="legend_bb">Bollinger</span></span></span></div>
<div class="panel"><div class="chart" id="px" style="height:420px"></div><h3 style="margin-top:14px" data-i="h_rsi">RSI (14): above 70 stretched up, below 30 stretched down</h3><div class="chart sm" id="rsi"></div></div>
</section>

<section id="depth" hidden><h2 data-i="h_depth">Order book depth</h2><p class="lede" data-i="lede_depth">Live crypto order book from Alpaca market data. Green is buyers waiting below the price, red is sellers waiting above. A wide gap in the middle means trading costs more. Shows orders within 0.5% of the price.</p>
<div class="toolbar"><label for="dsym" data-i="pair_label">Pair</label><select id="dsym"><option>BTC/USD</option><option>ETH/USD</option></select><button class="btn" id="dref" data-i="btn_refresh">Refresh</button><span id="spread"></span></div>
<div class="panel"><svg id="book" viewBox="0 0 800 320" width="100%" role="img" aria-label="Cumulative order book depth" direction="ltr"></svg></div>
</section>

<section id="quant" hidden><h2 data-i="h_quant">Quant metrics</h2><p class="lede" data-i="lede_quant">Computed from the bot's equity curve and closed trades. Blank means not enough history yet.</p>
<div class="panel scroll"><table><thead><tr><th data-i="th_metric">Metric</th><th data-i="th_val">Value</th><th data-i="th_formula">Formula</th><th data-i="th_meaning">What it tells you</th></tr></thead><tbody id="qt"></tbody></table></div>
<div class="note" data-i="note_quant">Backtests in this project (2005 to 2026) found no signal that beats diversified holding after costs. Read these numbers as risk control, not as a forecast of profit.</div>
</section>

<section id="calc" hidden><h2 data-i="h_calc">Calculators</h2><p class="lede" data-i="lede_calc">Worked out on the server with exact decimal arithmetic. Illustrations, not forecasts.</p>
<div class="calcs" id="calcs"></div>
</section>

<section id="ask" hidden><h2 data-i="h_ask">Assistant</h2><p class="lede" data-i="lede_ask">Ask about trading, risk or halal rules. Answers come only from the project's knowledge files and cite them. It never gives buy or sell calls.</p>
<div class="panel"><form id="askf" class="toolbar">
<button type="button" class="btn mic" id="mic" aria-pressed="false"><svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg><span id="micl" data-i="btn_mic">Ask by voice</span></button>
<input id="q" class="grow" maxlength="1000" placeholder="Type your question" aria-label="Type your question" data-ip="ask_ph" dir="auto">
<button class="btn primary" type="submit" data-i="btn_ask">Ask</button><button type="button" class="btn" id="hush" hidden data-i="btn_stop_voice">Stop speaking</button></form>
<p id="nospeech" class="note" hidden data-i="no_speech">Your browser has no speech recognition, so the microphone is hidden. Type your question instead (Chrome and Edge support voice).</p>
<div id="answer" aria-live="polite" dir="auto"></div></div>
</section>

<section id="guide" hidden><h2 data-i="h_guide">Setup guide</h2><p class="lede" data-ih="lede_guide_html">From a fresh Windows laptop to a running paper bot. Full version: <code>docs/SETUP_GUIDE.md</code>.</p>
<ol class="steps">
<li data-ih="g1_html"><b>Install Python 3.12</b> from python.org, ticking "Add to PATH".</li>
<li data-ih="g2_html"><b>Install the bot.</b> In PowerShell, in the project folder: <code>python -m venv .venv</code>, then <code>.venv\\Scripts\\pip install -e ".[dev,ml]"</code>.</li>
<li data-ih="g3_html"><b>Check it works:</b> <code>.venv\\Scripts\\python -m pytest -q</code> should end with "passed".</li>
<li data-ih="g4_html"><b>Get Alpaca paper keys.</b> Sign up at alpaca.markets, open the Paper dashboard, generate keys. Turn on "no shorting" in the paper account settings.</li>
<li data-ih="g5_html"><b>Choose a config.</b> Copy <code>examples\\paper_core.json</code> (Shariah ETFs) to <code>paper_core.json</code>. Add each fund's screening ratios under <code>financials</code>; without them the Shariah check blocks the fund.</li>
<li data-ih="g6_html"><b>Set your keys</b> for this window: <code>$env:APCA_API_KEY_ID="..."</code> and <code>$env:APCA_API_SECRET_KEY="..."</code>.</li>
<li data-ih="g7_html"><b>Run one step:</b> <code>.venv\\Scripts\\python -m quant_os_core.ops.run_paper --config paper_core.json --once</code>. Schedule it daily after the US close with Task Scheduler.</li>
<li data-ih="g8_html"><b>Open this dashboard:</b> set <code>$env:QUANT_OS_DASHBOARD_TOKEN</code> to a long random string, then run <code>.venv\\Scripts\\python -m quant_os_core.ops.dashboard --state state\\core.db --data data\\ohlc</code>.</li>
</ol>
<div class="note" data-i="note_guide">Paper only. Live trading stays locked behind the two-key interlock. Futures, margin and short selling stay blocked.</div>
</section>
<footer data-i="disclaimer">Educational paper-trading tool, not financial or religious advice. Ask a qualified scholar about zakat and purification for your own case.</footer>
</main></div>
<script>
const $=id=>document.getElementById(id);const LC=window.LightweightCharts;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const N=v=>`<bdi class="n">${esc(v)}</bdi>`;
const pct=x=>x==null?"\\u2013":(x*100).toFixed(1)+"%";const num=x=>x==null?"\\u2013":Number(x).toFixed(2);
const store={get:k=>{try{return localStorage.getItem(k)}catch(e){return null}},set:(k,v)=>{try{localStorage.setItem(k,v)}catch(e){}}};
let I={lang:"en",dir:"ltr",speech:"en-US",strings:{}};const t=(k,v)=>{let s=I.strings[k]??k;for(const n in v||{})s=s.split("{"+n+"}").join(v[n]);return s};
const GATES=["plan","edge","sizing","cost_guard","shariah","protections","governor","council","risk"];
const charts={};let last=null,FX=null,cur=store.get("qos.cur")||"PKR";
function locale(){return {en:"en-US",ur:"ur-PK",ar:"ar-SA"}[I.lang]+"-u-nu-latn"}
function money(usd){if(usd==null)return"\\u2013";let code="USD",v=usd;if(FX&&cur!=="USD"&&FX.rates[cur]){code=cur;v=usd*Number(FX.rates[cur])}
 try{return new Intl.NumberFormat("en-US",{style:"currency",currency:code,minimumFractionDigits:0,maximumFractionDigits:code==="BTC"?8:Math.abs(v)>=100000?0:2}).format(v)}catch(e){return code+" "+v.toFixed(2)}}
function group(s){if(s==null)return"\\u2013";if(s===true)return t("yes");if(s===false)return t("no");const[a,b]=String(s).split(".");return a.replace(/\\B(?=(\\d{3})+(?!\\d))/g,",")+(b!=null?"."+b:"")}
function mk(id,h){if(!LC)return null;const el=$(id);el.innerHTML="";const c=LC.createChart(el,{height:h||el.clientHeight,layout:{background:{color:"transparent"},textColor:css("--soft"),fontFamily:"Public Sans"},grid:{vertLines:{color:css("--line")},horzLines:{color:css("--line")}},rightPriceScale:{borderColor:css("--line")},timeScale:{borderColor:css("--line")}});new ResizeObserver(()=>c.applyOptions({width:el.clientWidth})).observe(el);return c}
function noLib(id){$(id).innerHTML=`<p class="empty">${esc(t("no_lib"))}</p>`}
const day=iso=>iso.slice(0,10);
function dedupe(rows){const m=new Map();rows.forEach(r=>m.set(r.time,r));return [...m.values()]}
function state(s){last=s;
 $("mode").textContent=s.kill_switch?t("mode_kill"):t("mode_paper");
 const m=s.metrics||{};const eq=s.equity.length?s.equity[s.equity.length-1][1]:null;
 const k=[["kpi_equity",money(eq),""],["kpi_total_return",pct(m.total_return),m.total_return>=0?"pos":"neg"],["kpi_worst_drop",pct(s.max_drawdown),"neg"],["kpi_sharpe",num(m.sharpe),""],["kpi_win_rate",pct(s.win_rate),""],["kpi_open_positions",String(s.positions.length),""]];
 $("kpis").innerHTML=k.map(([l,v,c])=>`<div class="kpi"><b class="${c}">${N(v)}</b><span>${esc(t(l))}</span></div>`).join("");
 const b=s.blocked_by||{};$("ladder").innerHTML=GATES.filter(g=>g in b||["shariah","risk","protections"].includes(g)).map(g=>`<div class="rung ${b[g]?"hit":""}"><strong>${N(b[g]||0)}</strong><small>${esc(t("gate_"+g))}</small></div>`).join("");
 if(LC&&s.equity.length>1){charts.eq=charts.eq||mk("eq");charts.dd=charts.dd||mk("dd");
  if(!charts.eqs){charts.eqs=charts.eq.addAreaSeries({lineColor:css("--act"),topColor:"rgba(59,75,168,.25)",bottomColor:"rgba(59,75,168,0)"});charts.dds=charts.dd.addAreaSeries({lineColor:css("--down"),topColor:"rgba(180,68,46,0)",bottomColor:"rgba(180,68,46,.3)",invertFilledArea:true})}
  charts.eqs.setData(dedupe(s.equity.map(([t,v])=>({time:day(t),value:v}))));charts.dds.setData(dedupe(s.equity.map(([t],i)=>({time:day(t),value:s.drawdown[i]*100}))));
 }else if(!s.equity.length){$("eq").innerHTML=`<p class="empty">${esc(t("empty_equity"))}</p>`}else if(!LC)noLib("eq");
 const th=ks=>ks.map(x=>`<th>${esc(t(x))}</th>`).join("");
 $("positions").innerHTML=s.positions.length?`<table><thead><tr>${th(["th_symbol","th_quantity","th_entry","th_stop","th_value","th_since"])}</tr></thead><tbody>${s.positions.map(p=>`<tr><td>${N(p.symbol)}</td><td>${N(p.quantity)}</td><td>${N(p.entry)}</td><td>${N(p.stop)}</td><td>${N(money(Number(p.quantity)*Number(p.entry)))}</td><td>${N(day(p.since))}</td></tr>`).join("")}</tbody></table>`:`<p class="empty">${esc(t("empty_positions"))}</p>`;
 $("events").innerHTML=s.events.length?`<table><tbody>${s.events.slice(0,12).map(e=>`<tr><td>${N(String(e.time||"").slice(0,16))}</td><td>${esc(e.level)}</td><td dir="auto">${esc(e.message)}</td></tr>`).join("")}</tbody></table>`:`<p class="empty">${esc(t("empty_events"))}</p>`;
 $("qt").innerHTML=Object.entries(s.formulas||{}).map(([n,[f,why]])=>{const v=m[n];const pc=["total_return","cagr","volatility","max_drawdown","win_rate","expectancy"].includes(n);return `<tr><td>${esc(I.strings["m_"+n]||n.replace("_"," "))}</td><td>${N(pc?pct(v):num(v))}</td><td><code>${esc(f)}</code></td><td>${esc(I.strings["w_"+n]||why)}</td></tr>`}).join("");
}
async function refresh(){try{state(await (await fetch("/api/state")).json())}catch(e){$("mode").textContent=t("mode_error")}}
async function loadSymbols(){const s=await (await fetch("/api/symbols")).json();$("sym").innerHTML=s.map(x=>`<option>${esc(x)}</option>`).join("")||`<option disabled>${esc(t("no_csv"))}</option>`;if(s.length){$("sym").value=s.includes("SPUS")?"SPUS":s[0];drawBars()}}
async function drawBars(){if(!LC){noLib("px");return}const r=await fetch("/api/bars?symbol="+encodeURIComponent($("sym").value));if(!r.ok)return;const d=(await r.json()).bars;
 const c=mk("px",420),cr=mk("rsi",140);const cs=c.addCandlestickSeries({upColor:css("--up"),downColor:css("--down"),borderVisible:false,wickUpColor:css("--up"),wickDownColor:css("--down")});cs.setData(d.map(b=>({time:b.time,open:b.open,high:b.high,low:b.low,close:b.close})));
 const line=(k,col,w)=>{const s=c.addLineSeries({color:col,lineWidth:w,priceLineVisible:false,lastValueVisible:false});s.setData(d.filter(b=>b[k]!=null).map(b=>({time:b.time,value:b[k]})))};
 line("sma50","#3b4ba8",2);line("sma200","#9a7622",2);line("bb_upper","#8a9aa3",1);line("bb_lower","#8a9aa3",1);
 const rs=cr.addLineSeries({color:css("--act"),lineWidth:1.5,priceLineVisible:false});rs.setData(d.filter(b=>b.rsi!=null).map(b=>({time:b.time,value:b.rsi})));
 [70,30].forEach(v=>rs.createPriceLine({price:v,color:css("--soft"),lineStyle:2,lineWidth:1,axisLabelVisible:true}));c.timeScale().fitContent();cr.timeScale().fitContent()}
async function drawBook(){const svg=$("book");$("spread").textContent=t("loading");
 const r=await fetch("/api/depth?symbol="+encodeURIComponent($("dsym").value));const d=await r.json();
 if(!r.ok){$("spread").textContent=d.error||t("book_unavailable");svg.innerHTML="";return}
 const all=[...d.bids,...d.asks];if(!all.length){$("spread").textContent=t("book_empty");return}
 const px=all.map(p=>p[0]),qty=all.map(p=>p[1]);const x0=Math.min(...px),x1=Math.max(...px),q1=Math.max(...qty);
 const X=p=>40+(p-x0)/(x1-x0||1)*740,Y=q=>290-q/q1*260;
 const path=(side,col)=>{if(!side.length)return"";const pts=side.map(([p,q])=>`${X(p)},${Y(q)}`);const base=`${X(side[side.length-1][0])},290 ${X(side[0][0])},290`;return `<polygon points="${pts.join(" ")} ${base}" fill="${col}" fill-opacity=".25" stroke="${col}" stroke-width="2"/>`};
 svg.innerHTML=`<line x1="40" y1="290" x2="780" y2="290" stroke="${css("--line")}"/>${path(d.bids,css("--up"))}${path(d.asks,css("--down"))}<text x="40" y="310" fill="${css("--soft")}" font-size="12">${x0.toFixed(2)}</text><text x="780" y="310" fill="${css("--soft")}" font-size="12" text-anchor="end">${x1.toFixed(2)}</text><text x="40" y="20" fill="${css("--soft")}" font-size="12">${esc(t("cum_size"))} ${q1.toFixed(3)}</text>`;
 $("spread").textContent=d.spread==null?"":t("spread_fmt",{spread:d.spread.toFixed(2),bps:d.spread_bps.toFixed(1)})}
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>{document.querySelectorAll("nav button").forEach(x=>x.removeAttribute("aria-current"));b.setAttribute("aria-current","page");document.querySelectorAll("main section").forEach(s=>s.hidden=s.id!==b.dataset.tab);if(b.dataset.tab==="depth")drawBook();if(b.dataset.tab==="charts"&&!$("sym").options.length)loadSymbols()});
$("sym").onchange=drawBars;$("dsym").onchange=drawBook;$("dref").onclick=drawBook;
async function kill(on){const r=await fetch("/api/kill",{method:"POST",headers:{"Authorization":"Bearer "+$("tok").value,"Content-Type":"application/json"},body:JSON.stringify({on,reason:"dashboard"})});$("ks").textContent=r.ok?(on?t("ks_on"):t("ks_off")):t("ks_wrong");refresh()}
$("kill").onclick=()=>kill(true);$("unkill").onclick=()=>kill(false);

/* ---- currency ---- */
function rateLine(){const el=$("rate");if(!FX){el.textContent=t("rate_unavailable");return}
 if(cur==="USD"||!FX.rates[cur]){el.textContent="";return}
 const when=new Date(FX.as_of).toLocaleString(locale(),{dateStyle:"medium",timeStyle:"short"});
 el.innerHTML=esc(t("rate_time",{rate:"\\u2068"+group(Number(FX.rates[cur]).toFixed(cur==="BTC"?8:2))+" "+cur+"\\u2069",time:when}))+(FX.stale?" ("+esc(t("rate_stale"))+")":"")}
async function loadFx(){try{const r=await fetch("/api/fx?base=USD");if(!r.ok)throw 0;FX=await r.json();
 const codes=Object.keys(FX.rates);$("cur").innerHTML=codes.map(c=>`<option>${esc(c)}</option>`).join("");if(!codes.includes(cur))cur="USD";$("cur").value=cur;
 document.querySelectorAll("select.cur").forEach(s=>{const v=s.value;s.innerHTML=$("cur").innerHTML;s.value=codes.includes(v)?v:s.dataset.def})}
 catch(e){FX=null;cur="USD";$("cur").value="USD"}rateLine();if(last)state(last)}
$("cur").onchange=()=>{cur=$("cur").value;store.set("qos.cur",cur);rateLine();if(last)state(last)};

/* ---- calculators ---- */
const CALCS=[
 {id:"position_size",t:"c_position",f:[["equity","f_equity","10000"],["risk_pct","f_risk_pct","1"],["entry","f_entry","50"],["stop","f_stop","47.5"]]},
 {id:"reward_risk",t:"c_rr",f:[["entry","f_entry","50"],["stop","f_stop","47.5"],["target","f_target","57.5"]]},
 {id:"compound",t:"c_compound",f:[["start","f_start","100000"],["monthly_deposit","f_monthly_deposit","10000"],["annual_return_pct","f_annual_return_pct","8"],["years","f_years","10"]]},
 {id:"goal",t:"c_goal",f:[["target","f_target_amount","5000000"],["years","f_years","10"],["annual_return_pct","f_annual_return_pct","8"],["start","f_start","0"]]},
 {id:"drawdown",t:"c_drawdown",f:[["drop_pct","f_drop_pct","25"]]},
 {id:"dca",t:"c_dca",f:[["buys","f_buys","10000@100, 10000@80, 10000@125"]]},
 {id:"cost",t:"c_cost",f:[["turnover","f_turnover","200000"],["round_trip_bps","f_round_trip_bps","20"],["equity","f_equity_opt","100000"]]},
 {id:"zakat",t:"c_zakat",f:[["zakatable_wealth","f_zakatable_wealth","1500000"],["nisab","f_nisab",""],["gold_price_per_gram","f_gold_price_per_gram",""]]},
 {id:"purification",t:"c_purify",f:[["dividends","f_dividends","12000"],["non_compliant_pct","f_non_compliant_pct","3"]]},
 {id:"fx",t:"c_fx",f:[["amount","f_amount","100"],["from","f_from","USD","cur"],["to","f_to","PKR","cur"]]}];
const DESC={position_size:"d_position",reward_risk:"d_rr",compound:"d_compound",goal:"d_goal",drawdown:"d_drawdown",dca:"d_dca",cost:"d_cost",zakat:"d_zakat",purification:"d_purify",fx:"d_fx"};
const results={};
$("calcs").innerHTML=CALCS.map(c=>`<form class="panel" data-calc="${c.id}" novalidate><h3 data-i="${c.t}">${esc(c.t)}</h3><p data-i="${DESC[c.id]}"></p><div class="fields">${c.f.map(([k,l,v,kind])=>`<label><span data-i="${l}"></span>${kind==="cur"?`<select name="${k}" class="cur" data-def="${v}"><option>${v}</option></select>`:`<input name="${k}" value="${esc(v)}" ${k==="buys"?'dir="ltr"':'inputmode="decimal" dir="ltr"'} autocomplete="off">`}</label>`).join("")}</div><button class="btn primary" data-i="btn_calc">Calculate</button><div class="out" aria-live="polite"></div></form>`).join("");
function renderOut(id){const f=document.querySelector(`[data-calc="${id}"]`),o=f.querySelector(".out"),r=results[id];if(!r){o.innerHTML="";return}
 if(r.error){o.innerHTML=`<p class="err">${esc(t("calc_error"))}: ${esc(r.error)}</p>`;return}
 const skip=["table","as_of","stale","amount","from","to"];
 let h=`<table><tbody>${Object.entries(r).filter(([k])=>!skip.includes(k)).map(([k,v])=>`<tr><td>${esc(t("r_"+k))}</td><td>${N(group(v))}${k==="converted"?" "+esc(r.to):""}</td></tr>`).join("")}</tbody></table>`;
 if(r.table)h+=`<div class="scroll"><table><thead><tr>${["th_year","th_deposited","th_balance","th_growth"].map(x=>`<th>${esc(t(x))}</th>`).join("")}</tr></thead><tbody>${r.table.map(y=>`<tr><td>${N(y.year)}</td><td>${N(group(y.deposited))}</td><td>${N(group(y.balance))}</td><td>${N(group(y.growth))}</td></tr>`).join("")}</tbody></table></div>`;
 o.innerHTML=h}
document.querySelectorAll("[data-calc]").forEach(f=>f.onsubmit=async e=>{e.preventDefault();const id=f.dataset.calc;const body=Object.fromEntries(new FormData(f));
 try{const r=id==="fx"?await fetch("/api/fx/convert?"+new URLSearchParams(body)):await fetch("/api/calc/"+id,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const d=await r.json();results[id]=r.ok?d:{error:d.error||r.status}}catch(err){results[id]={error:String(err)}}renderOut(id)});

/* ---- assistant ---- */
const SR=window.SpeechRecognition||window.webkitSpeechRecognition;let rec=null;
if(!SR){$("mic").hidden=true;$("nospeech").hidden=false}
function listening(on){$("mic").setAttribute("aria-pressed",on);$("micl").textContent=t(on?"btn_listening":"btn_mic")}
$("mic").onclick=()=>{if(rec){rec.stop();return}rec=new SR();rec.lang=I.speech;rec.interimResults=false;rec.maxAlternatives=1;
 rec.onresult=e=>{$("q").value=e.results[0][0].transcript;ask()};rec.onerror=()=>{$("answer").textContent=t("mic_denied")};rec.onend=()=>{rec=null;listening(false)};listening(true);try{rec.start()}catch(e){rec=null;listening(false)}};
function speak(text){if(!("speechSynthesis" in window))return;speechSynthesis.cancel();const arabic=/[\\u0600-\\u06FF]/.test(text);const lang=arabic?I.speech:"en-US";
 const u=new SpeechSynthesisUtterance(text.replace(/\\[\\d+\\]/g,"").slice(0,900));u.lang=lang;const v=speechSynthesis.getVoices().find(x=>x.lang&&x.lang.toLowerCase().startsWith(lang.slice(0,2)));if(v)u.voice=v;
 u.onend=()=>{$("hush").hidden=true};$("hush").hidden=false;speechSynthesis.speak(u)}
$("hush").onclick=()=>{if("speechSynthesis" in window)speechSynthesis.cancel();$("hush").hidden=true};
async function ask(){const q=$("q").value.trim();if(!q)return;const a=$("answer");a.textContent=t("thinking");
 try{const r=await fetch("/api/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q,lang:I.lang})});const d=await r.json();
  if(!r.ok){a.textContent=d.error||t("ask_error");return}
  a.innerHTML=esc(d.answer)+(d.sources&&d.sources.length?`<ul aria-label="${esc(t("ask_sources"))}">${d.sources.map(s=>`<li dir="auto">${esc(s)}</li>`).join("")}</ul>`:"");speak(d.answer)}
 catch(e){a.textContent=t("ask_error")}}
$("askf").onsubmit=e=>{e.preventDefault();ask()};

/* ---- language ---- */
function apply(){const h=document.documentElement;h.lang=I.lang;h.dir=I.dir;document.title=t("app_title");
 document.querySelectorAll("[data-i]").forEach(el=>{if(I.strings[el.dataset.i]!=null)el.textContent=I.strings[el.dataset.i]});
 document.querySelectorAll("[data-ih]").forEach(el=>{if(I.strings[el.dataset.ih]!=null)el.innerHTML=I.strings[el.dataset.ih]});
 document.querySelectorAll("[data-ip]").forEach(el=>{el.placeholder=t(el.dataset.ip);el.setAttribute("aria-label",t(el.dataset.ip))});
 document.querySelectorAll("[data-ia]").forEach(el=>el.setAttribute("aria-label",t(el.dataset.ia)));
 listening(!!rec);rateLine();Object.keys(results).forEach(renderOut);if(last)state(last)}
async function setLang(code){try{const r=await fetch("/api/i18n?lang="+encodeURIComponent(code));if(r.ok)I=await r.json()}catch(e){}
 $("lang").value=I.lang;store.set("qos.lang",I.lang);apply()}
$("lang").onchange=()=>setLang($("lang").value);
setLang(store.get("qos.lang")||"en").then(()=>{refresh();loadFx()});setInterval(refresh,15000);setInterval(loadFx,3600000);
</script></body></html>"""
