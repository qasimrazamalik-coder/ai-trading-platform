import type {
  CalendarCell,
  HeatmapData,
  MetricCardData,
  TerminalCandle,
  TerminalState
} from "./types.js";

export const metricCards: readonly MetricCardData[] = [
  { label: "Portfolio", value: "$124,680", change: "+2.84%", tone: "positive" },
  { label: "Sharpe", value: "1.86", change: "OOS", tone: "positive" },
  { label: "Max drawdown", value: "-7.4%", change: "30D", tone: "neutral" },
  { label: "Risk heat", value: "1.31%", change: "2.00% cap", tone: "neutral" }
];

export const correlationHeatmap: HeatmapData = {
  title: "Cross-asset correlation",
  metric: "Correlation",
  xLabels: ["BTC", "ETH", "Gold", "EURUSD", "AAPL", "KSE100"],
  yLabels: ["BTC", "ETH", "Gold", "EURUSD", "AAPL", "KSE100"],
  values: [
    [1.0, 0.78, 0.11, 0.04, 0.31, 0.18],
    [0.78, 1.0, 0.09, 0.02, 0.27, 0.14],
    [0.11, 0.09, 1.0, 0.34, -0.08, 0.05],
    [0.04, 0.02, 0.34, 1.0, -0.12, -0.03],
    [0.31, 0.27, -0.08, -0.12, 1.0, 0.42],
    [0.18, 0.14, 0.05, -0.03, 0.42, 1.0]
  ],
  valueMin: -1,
  valueMax: 1
};

export const stabilityHeatmap: HeatmapData = {
  title: "Parameter stability",
  metric: "OOS Sharpe",
  xLabels: ["20", "30", "40", "50", "60"],
  yLabels: ["0.8%", "1.0%", "1.2%", "1.4%", "1.6%"],
  values: [
    [0.82, 0.96, 1.03, 0.98, 0.89],
    [0.91, 1.08, 1.17, 1.13, 1.02],
    [0.97, 1.15, 1.24, 1.21, 1.08],
    [0.93, 1.11, 1.19, 1.16, 1.05],
    [0.86, 1.02, 1.09, 1.07, 0.99]
  ],
  valueMin: 0.8,
  valueMax: 1.3
};

export const regimeHeatmap: HeatmapData = {
  title: "Strategy × regime",
  metric: "Compatibility",
  xLabels: ["Bull", "Bear", "Range", "High vol", "Risk-off"],
  yLabels: ["Trend", "Mean rev", "Carry", "Stat arb", "Momentum"],
  values: [
    [0.92, 0.79, 0.28, 0.74, 0.62],
    [0.42, 0.51, 0.91, 0.34, 0.47],
    [0.71, 0.38, 0.66, 0.41, 0.29],
    [0.64, 0.58, 0.76, 0.69, 0.63],
    [0.88, 0.82, 0.31, 0.77, 0.68]
  ],
  valueMin: 0,
  valueMax: 1
};

export const calendar: readonly CalendarCell[] = [
  { date: "01", value: 0.004 }, { date: "02", value: -0.002 }, { date: "03", value: 0.006 },
  { date: "04", value: 0.001 }, { date: "05", value: 0.0 }, { date: "06", value: -0.004 },
  { date: "07", value: 0.008 }, { date: "08", value: 0.003 }, { date: "09", value: -0.001 },
  { date: "10", value: 0.005 }, { date: "11", value: 0.002 }, { date: "12", value: -0.003 },
  { date: "13", value: 0.007 }, { date: "14", value: 0.004 }, { date: "15", value: -0.002 },
  { date: "16", value: 0.003 }, { date: "17", value: 0.001 }, { date: "18", value: -0.001 },
  { date: "19", value: 0.006 }, { date: "20", value: 0.002 }, { date: "21", value: -0.004 }
];

export const equityCurve: readonly number[] = [
  100, 100.8, 100.1, 101.9, 102.4, 101.7, 103.8, 104.2, 103.6, 105.4,
  106.1, 105.7, 107.9, 108.8, 108.1, 110.6, 111.4, 110.9, 113.3, 114.9,
  116.2, 115.8, 118.7, 120.1, 121.8, 120.9, 123.4, 124.68
];

const closes = [
  63810, 64020, 63940, 64360, 64610, 64520, 64980, 65220, 65110, 65570,
  65840, 65720, 66130, 66420, 66310, 66850, 67120, 66980, 67420, 67820,
  67640, 68110, 68480, 68260, 68740, 69020, 68880, 69240, 69510, 69780
];

function candles(): readonly TerminalCandle[] {
  let previous = closes[0]! - 120;
  return closes.map((close, index) => {
    const open = previous;
    const high = Math.max(open, close) + 120 + (index % 4) * 35;
    const low = Math.min(open, close) - 95 - (index % 3) * 30;
    previous = close;
    return {
      time: 1_789_000_000 + index * 3600,
      open,
      high,
      low,
      close,
      volume: 740 + ((index * 83) % 560)
    };
  });
}

export const fallbackTerminalState: TerminalState = {
  mode: "paper_demo",
  synthetic: true,
  symbol: "BTCUSDT",
  venue: "OFFLINE",
  timeframe: "1h",
  last_price: 69780,
  change_pct: 3.14,
  candles: candles(),
  bids: Array.from({ length: 8 }, (_, i) => ({ price: 69778 - i * 18, quantity: 1.4 + i * 0.42 })),
  asks: Array.from({ length: 8 }, (_, i) => ({ price: 69782 + i * 18, quantity: 1.2 + i * 0.39 })),
  news: [
    { headline: "ETF flow report shows another week of net digital-asset inflows", source: "DemoWire", credibility: 0.88, sentiment: 0.54, age: "8m" },
    { headline: "Macro desk flags higher real yields ahead of policy update", source: "DemoMacro", credibility: 0.81, sentiment: -0.31, age: "21m" },
    { headline: "Unverified social post claims exchange disruption; no primary confirmation", source: "Community demo", credibility: 0.34, sentiment: -0.72, age: "32m" }
  ],
  positions: [
    { symbol: "BTCUSDT", side: "LONG", quantity: 0.18, entry: 68120, mark: 69780, pnl: 298.8 },
    { symbol: "ETHUSDT", side: "LONG", quantity: 2.4, entry: 3440, mark: 3491, pnl: 122.4 }
  ],
  registered_strategies: 15,
  risk_status: "NORMAL",
  live_trading_enabled: false
};
