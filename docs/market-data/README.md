# Universal Market Data Engine

Step 2 establishes one canonical data model across crypto, FX, equities, PSX, futures,
options and other adapter-backed venues.

## Boundary

```text
Provider SDK / REST / WebSocket
          ↓
Provider adapter
          ↓
Canonical InstrumentId + records
          ↓
Validation
          ↓
Persistence / Replay / Research
```

Provider-specific payloads never enter strategy code.

## Canonical records

- `Candle`: exact Decimal OHLCV plus open/close timestamps.
- `Trade`: trade id, price, size, aggressor side.
- `Quote`: bid/ask price and size with exact spread/midpoint.
- `OrderBookSnapshot`: ordered bid/ask levels with crossed-book validation.
- `CorporateAction`: dividends, splits, symbol changes, delistings and spinoffs.

## Data-quality gates

The validator detects:

1. duplicate timestamps;
2. unsorted observations;
3. missing fixed-timeframe intervals;
4. future timestamps / clock errors;
5. robust return outliers using median absolute deviation;
6. stale quotes.

A quality report is explicit. Data is never silently repaired in the domain core.

## Storage strategy

- PostgreSQL: instruments and corporate actions.
- ClickHouse: high-volume candles/ticks/order-book analytics.
- Parquet: portable historical research datasets (optional PyArrow adapter).
- In-memory store: deterministic unit tests and local replay.

## Timeframes

Step 2 supports fixed `s/m/h/d/w` durations. Exchange-calendar-aware monthly/quarterly
bars belong in the calendar service planned for later market-specific adapters.
