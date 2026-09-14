CREATE DATABASE IF NOT EXISTS quant_os;

CREATE TABLE IF NOT EXISTS quant_os.candles
(
    venue LowCardinality(String),
    symbol String,
    timeframe LowCardinality(String),
    ts DateTime64(6, 'UTC'),
    close_ts DateTime64(6, 'UTC'),
    source LowCardinality(String),
    open Decimal128(18),
    high Decimal128(18),
    low Decimal128(18),
    close Decimal128(18),
    volume Decimal128(18),
    quote_volume Nullable(Decimal128(18)),
    trade_count Nullable(UInt64),
    is_closed Bool,
    ingested_at DateTime64(6, 'UTC') DEFAULT now64(6)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMM(ts)
ORDER BY (venue, symbol, timeframe, ts);
