-- Quant OS execution/audit schema. Financial values use NUMERIC, never binary FLOAT.
CREATE TABLE IF NOT EXISTS execution_orders (
    client_order_id TEXT PRIMARY KEY,
    venue_order_id TEXT,
    venue TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    time_in_force TEXT NOT NULL,
    quantity NUMERIC(38, 18) NOT NULL CHECK (quantity > 0),
    limit_price NUMERIC(38, 18),
    stop_price NUMERIC(38, 18),
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS execution_fills (
    fill_id UUID PRIMARY KEY,
    client_order_id TEXT NOT NULL REFERENCES execution_orders(client_order_id),
    venue_order_id TEXT NOT NULL,
    venue TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity NUMERIC(38, 18) NOT NULL CHECK (quantity > 0),
    price NUMERIC(38, 18) NOT NULL CHECK (price > 0),
    fee NUMERIC(38, 18) NOT NULL CHECK (fee >= 0),
    filled_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS execution_audit_entity_idx
    ON execution_audit_log(entity_id, created_at DESC);
