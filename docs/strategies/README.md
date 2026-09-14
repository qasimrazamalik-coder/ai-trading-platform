# Strategy Factory

Step 6 adds a discoverable, testable strategy layer on top of the mathematical engine,
backtester and Right-Fit validation system.

## Design rules

1. Strategy output is normalized target exposure in `[-1, 1]`.
2. Strategy code cannot call a broker directly.
3. Research uses delayed signals by default to avoid same-bar look-ahead.
4. Event-driven execution reads actual filled position plus outstanding signed orders.
5. Strategy metadata declares category, parameters and known failure modes.
6. A strategy is only a hypothesis until it survives the Right-Fit research gate.

## Built-in registered strategies

- `trend.ma_crossover`
- `breakout.donchian`
- `momentum.time_series`
- `mean_reversion.zscore`
- `mean_reversion.bollinger`

Additional research primitives cover cross-sectional momentum, pairs/stat-arb, funding/basis,
volatility breakouts, VWAP deviation, order-flow imbalance, market-making quote mathematics,
Minervini-style trend screening (including a rising 200-day average), CAN SLIM-style scoring with explicit supply/demand, event/news transforms and ensembles.

## Selected formulas

### Time-series momentum

`M(t,n) = P(t) / P(t-n) - 1`

### Rolling z-score

`z(t) = (x(t) - mean_window) / sample_std_window`

### Pairs spread

`Y(t) = alpha + beta * X(t) + epsilon(t)`

`spread(t) = Y(t) - alpha - beta * X(t)`

Cointegration can be required before a pairs signal is activated.

### Order-book imbalance

`OBI = (bid_size - ask_size) / (bid_size + ask_size)`

### Annualized futures basis

`basis = (F / S - 1) * 365 / days_to_expiry`

### Avellaneda-Stoikov research quote

Reservation price:

`r = s - q * gamma * sigma^2 * T`

The quote helper is a research primitive only. Production market making still requires a
venue-aware order manager, queue-position modelling and independent risk controls.

## Event-driven adapter

`SingleAssetTargetAdapter` turns a normalized close-history signal into target units.
It uses both filled position and outstanding signed order quantity so partial fills do not
cause repeated orders that overshoot the intended target.
