CREATE TABLE IF NOT EXISTS instruments (
    venue TEXT NOT NULL,
    symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    instrument_type TEXT NOT NULL,
    base_currency TEXT NOT NULL,
    quote_currency TEXT NOT NULL,
    price_precision SMALLINT NOT NULL CHECK (price_precision BETWEEN 0 AND 18),
    quantity_precision SMALLINT NOT NULL CHECK (quantity_precision BETWEEN 0 AND 18),
    contract_multiplier NUMERIC NOT NULL DEFAULT 1,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (venue, symbol)
);

CREATE TABLE IF NOT EXISTS corporate_actions (
    venue TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action_type TEXT NOT NULL,
    effective_at TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL,
    cash_amount NUMERIC,
    currency TEXT,
    ratio NUMERIC,
    new_symbol TEXT,
    PRIMARY KEY (venue, symbol, action_type, effective_at)
);
