import { loadHealth, loadTerminalState, submitPaperOrder } from "./api.js";
import {
  calendar,
  correlationHeatmap,
  equityCurve,
  metricCards,
  regimeHeatmap,
  stabilityHeatmap
} from "./demo.js";
import type {
  CalendarCell,
  HeatmapData,
  MetricCardData,
  TerminalCandle,
  TerminalNews,
  TerminalState
} from "./types.js";

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function money(value: number): string {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
}

function renderMetric(card: MetricCardData): HTMLElement {
  const container = element("article", "metric-card");
  container.append(element("span", "metric-label", card.label));
  container.append(element("strong", "metric-value", card.value));
  if (card.change) {
    container.append(element("span", `metric-change ${card.tone ?? "neutral"}`, card.change));
  }
  return container;
}

function normalized(value: number, min: number, max: number): number {
  if (max <= min) return 0.5;
  return Math.min(1, Math.max(0, (value - min) / (max - min)));
}

function renderHeatmap(data: HeatmapData): HTMLElement {
  const card = element("section", "panel heatmap-panel");
  const header = element("div", "panel-header");
  const titleGroup = element("div");
  titleGroup.append(element("h2", undefined, data.title));
  titleGroup.append(element("span", "panel-subtitle", data.metric));
  header.append(titleGroup, element("span", "micro-label", `${data.xLabels.length} × ${data.yLabels.length}`));
  card.append(header);

  const grid = element("div", "heatmap-grid");
  grid.style.setProperty("--columns", String(data.xLabels.length + 1));
  grid.append(element("div", "heatmap-corner"));
  for (const label of data.xLabels) grid.append(element("div", "axis-label x-label", label));
  data.values.forEach((row, rowIndex) => {
    grid.append(element("div", "axis-label y-label", data.yLabels[rowIndex] ?? ""));
    row.forEach((value) => {
      const cell = element("div", "heat-cell");
      cell.style.setProperty("--score", normalized(value, data.valueMin, data.valueMax).toFixed(4));
      cell.textContent = value.toFixed(2);
      cell.title = `${data.metric}: ${value.toFixed(4)}`;
      grid.append(cell);
    });
  });
  card.append(grid);
  return card;
}

function renderEquityChart(values: readonly number[]): HTMLElement {
  const card = element("section", "panel chart-panel");
  const header = element("div", "panel-header");
  const group = element("div");
  group.append(element("h2", undefined, "Portfolio equity"));
  group.append(element("span", "panel-subtitle", "Research preview · normalized"));
  header.append(group, element("span", "status-pill", "RIGHT-FIT"));
  card.append(header);
  const width = 820;
  const height = 240;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, Number.EPSILON);
  const points = values.map((value, index) => {
    const x = (index / Math.max(values.length - 1, 1)) * width;
    const y = height - ((value - min) / span) * (height - 24) - 12;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "equity-svg");
  const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  polyline.setAttribute("points", points.join(" "));
  polyline.setAttribute("class", "equity-line");
  svg.append(polyline);
  card.append(svg);
  return card;
}

function pnlTone(value: number): string {
  if (value > 0.003) return "strong-positive";
  if (value > 0) return "positive";
  if (value < -0.003) return "strong-negative";
  if (value < 0) return "negative";
  return "flat";
}

function renderCalendar(cells: readonly CalendarCell[]): HTMLElement {
  const card = element("section", "panel calendar-panel");
  const header = element("div", "panel-header");
  const group = element("div");
  group.append(element("h2", undefined, "P&L calendar"));
  group.append(element("span", "panel-subtitle", "Daily research returns"));
  header.append(group, element("span", "panel-kpi", "+3.7% MTD"));
  card.append(header);
  const grid = element("div", "calendar-grid");
  for (const cellData of cells) {
    const cell = element("div", `calendar-cell ${pnlTone(cellData.value)}`);
    cell.append(element("span", "calendar-day", cellData.date));
    cell.append(element("strong", undefined, `${(cellData.value * 100).toFixed(1)}%`));
    grid.append(cell);
  }
  card.append(grid);
  return card;
}

function candleSvg(candles: readonly TerminalCandle[]): SVGSVGElement {
  const width = 920;
  const height = 390;
  const priceHeight = 300;
  const pad = 16;
  const prices = candles.flatMap((candle) => [candle.high, candle.low]);
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const span = Math.max(max - min, 1);
  const maxVolume = Math.max(...candles.map((candle) => candle.volume), 1);
  const xStep = (width - pad * 2) / candles.length;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "candle-svg");

  for (let row = 0; row < 5; row += 1) {
    const y = pad + (row / 4) * (priceHeight - pad * 2);
    const line = document.createElementNS(svg.namespaceURI, "line");
    line.setAttribute("x1", "0"); line.setAttribute("x2", String(width));
    line.setAttribute("y1", String(y)); line.setAttribute("y2", String(y));
    line.setAttribute("class", "chart-grid-line");
    svg.append(line);
  }

  const yPrice = (value: number): number => pad + (1 - (value - min) / span) * (priceHeight - pad * 2);
  candles.forEach((candle, index) => {
    const x = pad + index * xStep + xStep / 2;
    const up = candle.close >= candle.open;
    const wick = document.createElementNS(svg.namespaceURI, "line");
    wick.setAttribute("x1", String(x)); wick.setAttribute("x2", String(x));
    wick.setAttribute("y1", String(yPrice(candle.high))); wick.setAttribute("y2", String(yPrice(candle.low)));
    wick.setAttribute("class", up ? "wick up" : "wick down");
    svg.append(wick);
    const body = document.createElementNS(svg.namespaceURI, "rect");
    const openY = yPrice(candle.open);
    const closeY = yPrice(candle.close);
    body.setAttribute("x", String(x - Math.max(2, xStep * 0.28)));
    body.setAttribute("y", String(Math.min(openY, closeY)));
    body.setAttribute("width", String(Math.max(4, xStep * 0.56)));
    body.setAttribute("height", String(Math.max(2, Math.abs(openY - closeY))));
    body.setAttribute("rx", "1.5");
    body.setAttribute("class", up ? "candle up" : "candle down");
    svg.append(body);
    const volume = document.createElementNS(svg.namespaceURI, "rect");
    const volumeHeight = (candle.volume / maxVolume) * 60;
    volume.setAttribute("x", String(x - Math.max(2, xStep * 0.22)));
    volume.setAttribute("y", String(height - volumeHeight - 8));
    volume.setAttribute("width", String(Math.max(4, xStep * 0.44)));
    volume.setAttribute("height", String(volumeHeight));
    volume.setAttribute("class", up ? "volume up" : "volume down");
    svg.append(volume);
  });
  return svg;
}

function renderOrderBook(state: TerminalState): HTMLElement {
  const panel = element("section", "panel orderbook-panel");
  const header = element("div", "panel-header compact");
  header.append(element("h2", undefined, "Order book"), element("span", "micro-label", "Price / Size"));
  panel.append(header);
  const book = element("div", "book");
  [...state.asks].reverse().forEach((level) => {
    const row = element("div", "book-row ask");
    row.append(element("span", undefined, money(level.price)), element("span", undefined, level.quantity.toFixed(2)));
    book.append(row);
  });
  const spread = state.asks[0] && state.bids[0] ? state.asks[0].price - state.bids[0].price : 0;
  const mid = element("div", "book-mid");
  mid.append(element("strong", undefined, money(state.last_price)), element("span", undefined, `Spread ${money(spread)}`));
  book.append(mid);
  state.bids.forEach((level) => {
    const row = element("div", "book-row bid");
    row.append(element("span", undefined, money(level.price)), element("span", undefined, level.quantity.toFixed(2)));
    book.append(row);
  });
  panel.append(book);
  return panel;
}

function renderChart(state: TerminalState): HTMLElement {
  const panel = element("section", "panel market-chart-panel");
  const header = element("div", "market-header");
  const identity = element("div", "symbol-block");
  identity.innerHTML = `<span class="symbol">${state.symbol}</span><span class="venue">${state.venue} · ${state.timeframe}</span>`;
  const price = element("div", "live-price");
  price.innerHTML = `<strong>${money(state.last_price)}</strong><span class="positive">+${state.change_pct.toFixed(2)}%</span>`;
  const toolbar = element("div", "chart-toolbar");
  ["1m", "5m", "15m", "1h", "4h", "1D"].forEach((tf) => {
    const button = element("button", tf === state.timeframe ? "time-button active" : "time-button", tf);
    toolbar.append(button);
  });
  header.append(identity, price, toolbar);
  panel.append(header, candleSvg(state.candles));
  const footer = element("div", "chart-foot");
  footer.innerHTML = `<span>OHLCV · Synthetic verification stream</span><span>Last ${state.candles.length} bars</span>`;
  panel.append(footer);
  return panel;
}

function renderPositions(state: TerminalState): HTMLElement {
  const panel = element("section", "panel positions-panel");
  const header = element("div", "panel-header compact");
  header.append(element("h2", undefined, "Paper positions"), element("span", "status-pill", `${state.positions.length} OPEN`));
  panel.append(header);
  const table = element("div", "position-table");
  table.innerHTML = `<div class="position-row table-head"><span>Symbol</span><span>Side</span><span>Qty</span><span>Entry</span><span>Mark</span><span>P&L</span></div>`;
  state.positions.forEach((position) => {
    const row = element("div", "position-row");
    row.innerHTML = `<strong>${position.symbol}</strong><span>${position.side}</span><span>${position.quantity}</span><span>${money(position.entry)}</span><span>${money(position.mark)}</span><strong class="${position.pnl >= 0 ? "positive" : "negative"}">${position.pnl >= 0 ? "+" : ""}$${money(position.pnl)}</strong>`;
    table.append(row);
  });
  panel.append(table);
  return panel;
}

function credibilityTone(score: number): string {
  if (score >= 0.8) return "high";
  if (score >= 0.6) return "medium";
  return "low";
}

function renderNewsItem(news: TerminalNews): HTMLElement {
  const item = element("article", "news-item");
  const meta = element("div", "news-meta");
  meta.append(element("span", undefined, news.source), element("span", undefined, news.age));
  const score = element("span", `credibility ${credibilityTone(news.credibility)}`, `${Math.round(news.credibility * 100)} trust`);
  const sentiment = element("span", news.sentiment >= 0 ? "sentiment positive" : "sentiment negative", `${news.sentiment >= 0 ? "+" : ""}${news.sentiment.toFixed(2)}`);
  item.append(meta, element("h3", undefined, news.headline), score, sentiment);
  return item;
}

function renderNews(state: TerminalState): HTMLElement {
  const panel = element("section", "panel news-panel");
  const header = element("div", "panel-header compact");
  header.append(element("h2", undefined, "Market intelligence"), element("span", "micro-label", "Credibility-aware"));
  panel.append(header);
  state.news.forEach((news) => panel.append(renderNewsItem(news)));
  return panel;
}

function renderQuickOrder(state: TerminalState, source: string): HTMLElement {
  const panel = element("section", "panel quick-order");
  panel.innerHTML = `
    <div class="panel-header compact"><h2>Order ticket</h2><span class="paper-lock">PAPER ONLY</span></div>
    <div class="order-tabs">
      <button class="active" data-order-type="market">Market</button>
      <button data-order-type="limit">Limit</button>
      <button data-order-type="stop">Stop</button>
    </div>
    <label>Quantity<input class="order-quantity" value="0.01" inputmode="decimal" aria-label="Quantity" /></label>
    <label class="order-price-row" hidden>Trigger / limit price<input class="order-price" value="${state.last_price.toFixed(2)}" inputmode="decimal" aria-label="Order price" /></label>
    <div class="risk-preview"><span>Execution</span><strong>Risk gate + paper broker</strong></div>
    <div class="order-actions"><button class="buy-button">Buy / Long</button><button class="sell-button">Sell / Short</button></div>
    <p class="order-status fineprint">${source === "api" ? "Ready for paper execution." : "Backend required for paper execution."}</p>
  `;

  const tabs = panel.querySelectorAll<HTMLButtonElement>("[data-order-type]");
  const quantityInput = panel.querySelector<HTMLInputElement>(".order-quantity");
  const priceInput = panel.querySelector<HTMLInputElement>(".order-price");
  const priceRow = panel.querySelector<HTMLElement>(".order-price-row");
  const status = panel.querySelector<HTMLElement>(".order-status");
  const buy = panel.querySelector<HTMLButtonElement>(".buy-button");
  const sell = panel.querySelector<HTMLButtonElement>(".sell-button");
  if (!quantityInput || !priceInput || !priceRow || !status || !buy || !sell) return panel;

  let orderType: "market" | "limit" | "stop" = "market";
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((item) => item.classList.remove("active"));
      tab.classList.add("active");
      orderType = (tab.dataset.orderType ?? "market") as typeof orderType;
      priceRow.hidden = orderType === "market";
    });
  });

  const submit = async (side: "buy" | "sell"): Promise<void> => {
    if (source !== "api") {
      status.textContent = "Paper execution unavailable while using offline fallback data.";
      return;
    }
    const quantity = quantityInput.value.trim();
    if (!quantity || Number(quantity) <= 0) {
      status.textContent = "Enter a positive quantity.";
      return;
    }
    const price = priceInput.value.trim();
    if (orderType !== "market" && (!price || Number(price) <= 0)) {
      status.textContent = "Enter a positive trigger / limit price.";
      return;
    }
    buy.disabled = true;
    sell.disabled = true;
    status.textContent = "Submitting paper order…";
    const clientOrderId = globalThis.crypto?.randomUUID
      ? `web-${globalThis.crypto.randomUUID()}`
      : `web-${Date.now()}`;
    try {
      const response = await submitPaperOrder({
        client_order_id: clientOrderId,
        symbol: state.symbol,
        side,
        quantity,
        order_type: orderType,
        ...(orderType === "limit" ? { limit_price: price } : {}),
        ...(orderType === "stop" ? { stop_price: price } : {})
      });
      status.textContent = response.average_fill_price
        ? `${response.status.toUpperCase()} @ ${response.average_fill_price}`
        : response.status.toUpperCase();
      const refreshed = await loadTerminalState();
      setView(
        "Trading Command Center",
        "Paper terminal",
        renderTerminal(refreshed.data, refreshed.source)
      );
    } catch (error) {
      status.textContent = error instanceof Error ? error.message : "Paper order failed.";
    } finally {
      buy.disabled = false;
      sell.disabled = false;
    }
  };

  buy.addEventListener("click", () => void submit("buy"));
  sell.addEventListener("click", () => void submit("sell"));
  return panel;
}

function renderTerminal(state: TerminalState, source: string): HTMLElement {
  const root = element("div", "terminal-view");
  const status = element("div", "terminal-status");
  status.innerHTML = `
    <span><i class="status-dot"></i>${source === "api" ? "Backend connected" : "Offline fallback"}</span>
    <span>${state.registered_strategies} strategies</span>
    <span>Risk ${state.risk_status}</span>
    <span class="paper-lock">LIVE ${state.live_trading_enabled ? "ON" : "LOCKED"}</span>
  `;
  const grid = element("div", "trading-grid");
  const main = element("main", "trading-main");
  main.append(renderChart(state), renderPositions(state), renderNews(state));
  const rail = element("aside", "trading-rail");
  rail.append(renderOrderBook(state), renderQuickOrder(state, source), renderSystemCard(state));
  grid.append(main, rail);
  root.append(status, grid);
  return root;
}

function renderSystemCard(state: TerminalState): HTMLElement {
  const panel = element("section", "panel system-card");
  panel.innerHTML = `
    <div class="panel-header compact"><h2>Execution safety</h2><span class="online-dot"></span></div>
    <div class="system-row"><span>Mode</span><strong>Paper</strong></div>
    <div class="system-row"><span>Risk gate</span><strong class="positive">Active</strong></div>
    <div class="system-row"><span>Reconciliation</span><strong class="positive">Ready</strong></div>
    <div class="system-row"><span>Live execution</span><strong class="negative">${state.live_trading_enabled ? "Enabled" : "Locked"}</strong></div>
  `;
  return panel;
}

function renderAnalytics(): HTMLElement {
  const root = element("div", "analytics-view");
  const metrics = element("section", "metrics-grid");
  metricCards.forEach((card) => metrics.append(renderMetric(card)));
  const analytics = element("div", "analytics-layout");
  const main = element("main", "analytics-main");
  main.append(renderEquityChart(equityCurve));
  const heatmaps = element("div", "heatmap-layout");
  heatmaps.append(renderHeatmap(correlationHeatmap), renderHeatmap(stabilityHeatmap));
  main.append(heatmaps, renderHeatmap(regimeHeatmap), renderCalendar(calendar));
  analytics.append(main, renderResearchRail());
  root.append(metrics, analytics);
  return root;
}

function renderResearchRail(): HTMLElement {
  const rail = element("aside", "signal-rail");
  rail.innerHTML = `
    <div class="rail-title"><span>Quant Copilot</span><span class="online-dot"></span></div>
    <div class="regime-card"><span class="eyebrow">Composite regime</span><strong>Bull · High Vol</strong><div class="confidence"><span style="width:82%"></span></div><small>82% confidence · liquidity normal</small></div>
    <div class="decision-block"><span class="eyebrow">Portfolio action</span><strong>Risk budget maintained</strong><p>Trend and momentum remain compatible. Mean-reversion allocation reduced.</p></div>
    <div class="decision-grid"><div><span>Trend</span><strong>34%</strong></div><div><span>Momentum</span><strong>27%</strong></div><div><span>Stat arb</span><strong>21%</strong></div><div><span>Mean rev</span><strong>8%</strong></div></div>
    <button class="primary-button">Open research trace</button>
    <div class="safety-note"><strong>Paper mode</strong><span>Live trading remains disabled.</span></div>
  `;
  return rail;
}

function renderStrategyLab(): HTMLElement {
  const root = element("div", "lab-grid");
  const builder = element("section", "panel strategy-builder");
  builder.innerHTML = `
    <div class="panel-header"><div><h2>Strategy Builder</h2><span class="panel-subtitle">Visual logic → same production strategy contract</span></div><span class="status-pill">RIGHT-FIT REQUIRED</span></div>
    <div class="builder-canvas">
      <div class="builder-node"><span>DATA</span><strong>BTC · 4H</strong><small>OHLCV + funding</small></div><div class="connector">→</div>
      <div class="builder-node"><span>SIGNAL</span><strong>Donchian 55</strong><small>breakout</small></div><div class="connector">→</div>
      <div class="builder-node"><span>FILTER</span><strong>Regime</strong><small>trend ≥ 0.65</small></div><div class="connector">→</div>
      <div class="builder-node"><span>RISK</span><strong>Vol target</strong><small>12% annual</small></div>
    </div>
    <div class="builder-controls"><button class="primary-button">Run backtest</button><button class="ghost-button">View Python</button><button class="ghost-button">Validate</button></div>
  `;
  const catalog = element("section", "panel strategy-catalog");
  catalog.innerHTML = `
    <div class="panel-header"><div><h2>Strategy catalog</h2><span class="panel-subtitle">Research-ready families</span></div><span class="micro-label">15 registered</span></div>
    <div class="catalog-grid">
      <article><strong>Trend Ensemble</strong><span>Momentum · Donchian · volatility adjusted</span><em>A stability</em></article>
      <article><strong>Stat Arb</strong><span>Pairs · z-score · cointegration gate</span><em>Research</em></article>
      <article><strong>Factor Stack</strong><span>Momentum · quality · carry · defensive</span><em>Research</em></article>
      <article><strong>Order Flow</strong><span>Imbalance · microprice · VWAP deviation</span><em>Intraday</em></article>
      <article><strong>CAN SLIM</strong><span>Growth · leadership · supply/demand</span><em>Screener</em></article>
      <article><strong>Market Making</strong><span>Inventory-aware quotes · spread model</span><em>Sandbox</em></article>
    </div>
  `;
  root.append(builder, catalog);
  return root;
}

function renderAcademy(): HTMLElement {
  const root = element("div", "academy-view");
  const hero = element("section", "academy-hero");
  hero.innerHTML = `<span class="eyebrow">Quant OS Academy</span><h2>Learn on the same system you trade.</h2><p>Every lesson links concepts to the live research, risk and paper-execution tools.</p>`;
  root.append(hero);
  const grid = element("div", "lesson-grid");
  const lessons = [
    ["01", "First backtest", "Build a strategy and learn why same-bar fills are forbidden."],
    ["02", "Candles & order books", "Read OHLCV, spread, depth and microstructure."],
    ["03", "Risk before returns", "Position sizing, heat, stops, drawdowns and kill switches."],
    ["04", "Right-Fit", "Walk-forward, PBO, DSR and parameter-stability heatmaps."],
    ["05", "Machine learning", "Leakage-safe features, baselines, calibration and tournaments."],
    ["06", "Paper execution", "OMS, idempotency, fills, reconciliation and failure recovery."],
    ["07", "News credibility", "Primary sources, corroboration, contradictions and sentiment."],
    ["08", "Go live safely", "Broker adapters, permissions, tiny allocations and scaling gates."]
  ];
  lessons.forEach(([number, title, description]) => {
    const lesson = element("article", "lesson-card");
    lesson.innerHTML = `<span>${number}</span><strong>${title}</strong><p>${description}</p><button>Open lesson →</button>`;
    grid.append(lesson);
  });
  root.append(grid);
  return root;
}

function setView(title: string, eyebrow: string, content: HTMLElement): void {
  const root = document.querySelector<HTMLElement>("#app");
  const heading = document.querySelector<HTMLElement>("#page-title");
  const eyebrowNode = document.querySelector<HTMLElement>("#page-eyebrow");
  if (!root || !heading || !eyebrowNode) throw new Error("application shell is incomplete");
  heading.textContent = title;
  eyebrowNode.textContent = eyebrow;
  root.replaceChildren(content);
}

function bindNavigation(state: TerminalState, source: string): void {
  document.querySelectorAll<HTMLButtonElement>(".nav-item").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      const view = button.dataset.view;
      if (view === "terminal") setView("Trading Command Center", "Paper terminal", renderTerminal(state, source));
      else if (view === "analytics") setView("Research Command Center", "Analytics terminal", renderAnalytics());
      else if (view === "strategies") setView("Strategy Laboratory", "Build · test · falsify", renderStrategyLab());
      else if (view === "academy") setView("Academy", "Interactive learning", renderAcademy());
      else setView("Research Command Center", "Analytics terminal", renderAnalytics());
    });
  });
}

async function boot(): Promise<void> {
  const [{ data: state, source }, healthy] = await Promise.all([loadTerminalState(), loadHealth()]);
  const sourceBadge = document.querySelector<HTMLElement>("#source-badge");
  if (sourceBadge) {
    sourceBadge.textContent = source === "api" && healthy ? "API · PAPER DEMO" : "OFFLINE DEMO";
    sourceBadge.classList.toggle("offline", source !== "api" || !healthy);
  }
  setView("Trading Command Center", "Paper terminal", renderTerminal(state, source));
  bindNavigation(state, source);
}

void boot();
