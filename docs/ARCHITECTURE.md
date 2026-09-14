# Architecture

Quant OS is designed around strict separation of responsibilities.

```text
Market Data -> Strategy -> OrderIntent -> Risk -> Execution -> Broker -> Fill
```

## Non-negotiable boundaries

1. Strategies produce intent; they never call brokers directly.
2. Risk decisions are independent from strategy logic.
3. Broker adapters contain no trading strategy logic.
4. Research code cannot enable live trading.
5. Financial state transitions are replayable and auditable.
6. Production live trading is disabled by default.

## Event-first design

All future services exchange versioned domain events. This keeps market data, models,
risk and execution replaceable without coupling their internals.
