# Institutional Backtesting Engine

Quant OS has two complementary backtesting paths.

## Vectorized research engine

Use `run_vectorized_backtest` for rapid single-asset screening across large parameter grids.
Signals are converted to target positions, execution can be delayed by a configurable number
of bars, and turnover is charged explicit fee/slippage basis points.

This engine is intentionally simple and fast. A strategy that survives screening must still be
validated in the event-driven simulator.

## Event-driven simulator

`EventDrivenBacktester` processes canonical `Candle` events and supports merged multi-asset
timelines. Each instrument maintains its own local bar counter so latency is measured in bars of
the target instrument rather than unrelated global events.

### Anti-look-ahead rule

An order submitted after observing a candle cannot fill on that candle. `BrokerConfig` enforces
`latency_bars >= 1`. For cross-asset orders, a target candle whose open timestamp precedes the
submission timestamp is also ineligible.

### Order support

- market;
- limit, with gap price improvement and a hard no-worse-than-limit constraint;
- stop, including gap-through behavior;
- GTC and IOC time-in-force;
- configurable partial fills based on a fraction of candle volume.

### Execution costs

`BpsSlippageModel` combines:

- half-spread basis points;
- fixed adverse slippage basis points;
- square-root volume-participation impact.

`ProportionalFeeModel` supports taker/maker basis-point schedules and minimum fees.
`FinancingModel` applies separate annualized long and short financing/borrow rates.

### Accounting

Cash, fills, fees and positions use `Decimal` at accounting boundaries. Long and short positions
support weighted-average entry price, realized P&L, mark-to-market equity, gross exposure and net
exposure. The fill ledger is the source of truth for event-driven results.

### Analytics

Backtest summaries are derived from the recorded equity ledger and expose total/annualized return,
maximum drawdown, Sharpe, Sortino and Calmar when mathematically defined. Benchmark comparison
supports beta, tracking error, information ratio and active return.

## Important limitations at Step 4

The simulator does not yet claim exchange-specific queue-position modelling, L2 matching-engine
simulation, portfolio margin, liquidation rules, or broker-specific tax treatment. Those require
venue-specific models and later execution/risk work. Step 5 adds anti-overfit validation; a clean
backtest is not evidence of persistent alpha by itself.
