export interface HeatmapData {
  readonly title: string;
  readonly metric: string;
  readonly xLabels: readonly string[];
  readonly yLabels: readonly string[];
  readonly values: readonly (readonly number[])[];
  readonly valueMin: number;
  readonly valueMax: number;
}

export interface MetricCardData {
  readonly label: string;
  readonly value: string;
  readonly change?: string;
  readonly tone?: "positive" | "negative" | "neutral";
}

export interface DistributionData {
  readonly title: string;
  readonly bins: readonly number[];
  readonly counts: readonly number[];
}

export interface CalendarCell {
  readonly date: string;
  readonly value: number;
}

export interface TerminalCandle {
  readonly time: number;
  readonly open: number;
  readonly high: number;
  readonly low: number;
  readonly close: number;
  readonly volume: number;
}

export interface TerminalBookLevel {
  readonly price: number;
  readonly quantity: number;
}

export interface TerminalNews {
  readonly headline: string;
  readonly source: string;
  readonly credibility: number;
  readonly sentiment: number;
  readonly age: string;
}

export interface TerminalPosition {
  readonly symbol: string;
  readonly side: string;
  readonly quantity: number;
  readonly entry: number;
  readonly mark: number;
  readonly pnl: number;
}

export interface TerminalState {
  readonly mode: string;
  readonly synthetic: boolean;
  readonly symbol: string;
  readonly venue: string;
  readonly timeframe: string;
  readonly last_price: number;
  readonly change_pct: number;
  readonly candles: readonly TerminalCandle[];
  readonly bids: readonly TerminalBookLevel[];
  readonly asks: readonly TerminalBookLevel[];
  readonly news: readonly TerminalNews[];
  readonly positions: readonly TerminalPosition[];
  readonly registered_strategies: number;
  readonly risk_status: string;
  readonly live_trading_enabled: boolean;
}

export interface SystemCapability {
  readonly label: string;
  readonly value: string;
  readonly status: "ready" | "locked" | "research";
}

export type PaperOrderType = "market" | "limit" | "stop";
export type PaperOrderSide = "buy" | "sell";

export interface PaperOrderCommand {
  readonly client_order_id: string;
  readonly symbol: string;
  readonly side: PaperOrderSide;
  readonly quantity: string;
  readonly order_type: PaperOrderType;
  readonly limit_price?: string;
  readonly stop_price?: string;
}

export interface PaperOrderResponse {
  readonly client_order_id: string;
  readonly venue_order_id: string;
  readonly symbol: string;
  readonly status: string;
  readonly submitted_quantity: string;
  readonly filled_quantity: string;
  readonly average_fill_price: string | null;
}
