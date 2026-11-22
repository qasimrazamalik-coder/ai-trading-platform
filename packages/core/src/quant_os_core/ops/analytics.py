"""Dashboard analytics: quant metrics with their formulas, chart bars with indicators, depth.

Pure functions over the state store and local CSVs, so the dashboard stays offline-testable.
Order-book depth is fetched through an injected callable (Alpaca's public crypto book by
default) and is optional: the dashboard works without it.
"""

from __future__ import annotations

import csv
import json
import math
import re
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

SYMBOL = re.compile(r"^[A-Z0-9.\-/]{1,15}$")

# name -> (formula shown to the owner, what it means)
FORMULAS: dict[str, tuple[str, str]] = {
    "total_return": ("E_end / E_start - 1", "growth of the account over the whole curve"),
    "cagr": ("(E_end / E_start)^(365 / days) - 1", "compound growth per year"),
    "volatility": ("stdev(r_daily) * sqrt(252)", "how much returns swing per year"),
    "sharpe": ("mean(r_daily) / stdev(r_daily) * sqrt(252)", "return per unit of risk"),
    "sortino": ("mean(r_daily) / stdev(r_daily < 0) * sqrt(252)", "return per unit of loss"),
    "max_drawdown": ("min(E_t / max(E_0..E_t) - 1)", "worst fall from a previous high"),
    "calmar": ("CAGR / |max drawdown|", "growth per unit of worst fall"),
    "win_rate": ("winning trades / closed trades", "share of trades that made money"),
    "profit_factor": ("sum(gains) / |sum(losses)|", "above 1 means trades net positive"),
    "expectancy": ("mean(profit ratio per trade)", "average result of one trade"),
}


def _daily_returns(equity: Sequence[float]) -> list[float]:
    return [b / a - 1 for a, b in pairwise(equity) if a > 0]


def _stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def metrics(
    curve: Sequence[tuple[str, float]], trade_ratios: Sequence[float]
) -> dict[str, float | None]:
    """Quant metrics from an equity curve of (ISO time, equity) and closed-trade ratios."""

    out: dict[str, float | None] = dict.fromkeys(FORMULAS)
    equity = [e for _, e in curve]
    if len(equity) >= 2 and equity[0] > 0:
        rets = _daily_returns(equity)
        days = max(
            1.0,
            (_parse_day(curve[-1][0]) - _parse_day(curve[0][0])),
        )
        out["total_return"] = equity[-1] / equity[0] - 1
        out["cagr"] = (equity[-1] / equity[0]) ** (365 / days) - 1 if days >= 30 else None
        sd = _stdev(rets)
        downside = _stdev([r for r in rets if r < 0])
        mean = sum(rets) / len(rets) if rets else 0.0
        out["volatility"] = sd * math.sqrt(252)
        out["sharpe"] = mean / sd * math.sqrt(252) if sd > 0 else None
        out["sortino"] = mean / downside * math.sqrt(252) if downside > 0 else None
        peak, worst = equity[0], 0.0
        for e in equity:
            peak = max(peak, e)
            worst = min(worst, e / peak - 1)
        out["max_drawdown"] = worst
        cagr = out["cagr"]
        out["calmar"] = cagr / abs(worst) if cagr is not None and worst < 0 else None
    if trade_ratios:
        gains = sum(r for r in trade_ratios if r > 0)
        losses = sum(r for r in trade_ratios if r < 0)
        out["win_rate"] = sum(1 for r in trade_ratios if r > 0) / len(trade_ratios)
        out["profit_factor"] = gains / abs(losses) if losses < 0 else None
        out["expectancy"] = sum(trade_ratios) / len(trade_ratios)
    return out


def _parse_day(iso: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(iso).timestamp() / 86400


# ---------------------------------------------------------------- bars + indicators


def available_symbols(data_dir: Path) -> list[str]:
    if not data_dir.is_dir():
        return []
    return sorted(p.stem for p in data_dir.glob("*.csv") if SYMBOL.match(p.stem))


def _sma(xs: Sequence[float], n: int) -> list[float | None]:
    out: list[float | None] = [None] * len(xs)
    run = 0.0
    for i, x in enumerate(xs):
        run += x
        if i >= n:
            run -= xs[i - n]
        if i >= n - 1:
            out[i] = run / n
    return out


def _rsi(xs: Sequence[float], n: int = 14) -> list[float | None]:
    out: list[float | None] = [None] * len(xs)
    if len(xs) <= n:
        return out
    gains = [max(xs[i] - xs[i - 1], 0.0) for i in range(1, len(xs))]
    losses = [max(xs[i - 1] - xs[i], 0.0) for i in range(1, len(xs))]
    avg_g, avg_l = sum(gains[:n]) / n, sum(losses[:n]) / n
    for i in range(n, len(xs)):
        if i > n:
            avg_g = (avg_g * (n - 1) + gains[i - 1]) / n
            avg_l = (avg_l * (n - 1) + losses[i - 1]) / n
        out[i] = 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    return out


def bars(data_dir: Path, symbol: str, limit: int = 750) -> dict[str, Any]:
    """OHLC bars (open = previous close when the CSV has none) plus SMA, Bollinger, RSI."""

    if not SYMBOL.match(symbol) or symbol not in available_symbols(data_dir):
        raise ValueError("unknown symbol")
    with (data_dir / f"{symbol}.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    closes = [float(r["close"]) for r in rows]
    sma20, sma50, sma200 = _sma(closes, 20), _sma(closes, 50), _sma(closes, 200)
    rsi14 = _rsi(closes)
    out: list[dict[str, Any]] = []
    start = max(0, len(rows) - limit)
    for i in range(start, len(rows)):
        r = rows[i]
        close = closes[i]
        open_ = float(r["open"]) if r.get("open") else (closes[i - 1] if i else close)
        window = closes[max(0, i - 19) : i + 1]
        sd = _stdev(window) if len(window) == 20 else None
        mid = sma20[i]
        out.append(
            {
                "time": r["time"][:10],
                "open": open_,
                "high": max(float(r["high"]), open_, close),
                "low": min(float(r["low"]), open_, close),
                "close": close,
                "sma50": sma50[i],
                "sma200": sma200[i],
                "bb_upper": mid + 2 * sd if mid is not None and sd is not None else None,
                "bb_lower": mid - 2 * sd if mid is not None and sd is not None else None,
                "rsi": rsi14[i],
            }
        )
    return {"symbol": symbol, "bars": out}


# ---------------------------------------------------------------- depth

DepthFetcher = Callable[[str], dict[str, Any]]
ALPACA_BOOK = "https://data.alpaca.markets/v1beta3/crypto/us/latest/orderbooks"


def alpaca_crypto_book(symbol: str) -> dict[str, Any]:
    """Latest crypto order book from Alpaca's market data API (no account needed)."""

    url = f"{ALPACA_BOOK}?{urllib.parse.urlencode({'symbols': symbol})}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        payload: dict[str, Any] = json.loads(response.read())
    book: dict[str, Any] = payload["orderbooks"][symbol]
    return book


def depth(
    symbol: str,
    fetch: DepthFetcher = alpaca_crypto_book,
    levels: int = 50,
    window: float = 0.005,
) -> dict[str, Any]:
    """Cumulative bid/ask depth and the spread, from a raw book of ``b``/``a`` price levels."""

    if not SYMBOL.match(symbol) or "/" not in symbol:
        raise ValueError("depth is available for crypto pairs like BTC/USD")
    book = fetch(symbol)
    bids = sorted(((float(x["p"]), float(x["s"])) for x in book.get("b", [])), reverse=True)
    asks = sorted((float(x["p"]), float(x["s"])) for x in book.get("a", []))
    if bids and asks:  # keep levels near the price, so one far-off order cannot squash the chart
        mid0 = (bids[0][0] + asks[0][0]) / 2
        bids = [x for x in bids if x[0] >= mid0 * (1 - window)]
        asks = [x for x in asks if x[0] <= mid0 * (1 + window)]
    bids, asks = bids[:levels], asks[:levels]

    def cumulative(side: list[tuple[float, float]]) -> list[list[float]]:
        total, out = 0.0, []
        for price, size in side:
            total += size
            out.append([price, total])
        return out

    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    spread = best_ask - best_bid if best_bid and best_ask else None
    mid = (best_ask + best_bid) / 2 if best_bid and best_ask else None
    return {
        "symbol": symbol,
        "bids": cumulative(bids),
        "asks": cumulative(asks),
        "spread": spread,
        "spread_bps": spread / mid * 10_000 if spread is not None and mid else None,
    }
