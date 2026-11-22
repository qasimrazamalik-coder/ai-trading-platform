# Strategies and elite backtesting

## Strategy families in the platform

| Family | Idea | Typical holding | Code |
|---|---|---|---|
| Trend following | Buy when price is above its long average or breaks an N-day high; exit on a trailing stop | weeks to months | `trend_200`, `donchian`, `trend_filtered` |
| Momentum (relative strength) | Hold the assets with the strongest 3-12 month returns, rotate monthly | 1-3 months | `dual_momentum`, `momentum_trend_top`, `sector_rotation` |
| Mean reversion | Buy short, sharp dips in liquid assets that are in a long-term uptrend | 1-5 days | strategy factory |
| Breakout day trading | Trade the break of the opening range with a tight stop, flat by the close | minutes to hours | `day_trading_policy()` + exits |
| Ensemble | Blend uncorrelated strategies; renormalise weights | mixed | `ensemble_v3` |

Shorting, leverage, futures, options and CFDs are not available: halal mode is long-only spot.
A long-only system protects itself in falling markets by moving to cash, not by shorting.

## The order every idea must pass

1. **Hypothesis first.** Write why the edge should exist (who is on the other side and why they
   keep losing) before looking at results.
2. **Clean data.** Split- and dividend-adjusted prices, delisted stocks included (otherwise
   survivorship bias inflates results), timestamps in UTC.
3. **Realistic fills.** Decide on bar `t`, fill at the next bar's open or at a price the bar
   proves was available. Charge fees, spread and slippage (default 0.10% per side).
4. **In-sample build, out-of-sample test.** Never tune on the test period.
5. **Walk-forward.** Re-fit on a rolling window, test on the next unseen window, stitch the
   test windows together. Leave a gap at least as long as the holding period between train and
   test so labels do not overlap (purging and embargo).
6. **Count every trial.** Every parameter set and variant tried is recorded; the Deflated
   Sharpe ratio uses the count.
7. **Gates.** Deflated Sharpe >= 0.95 and PBO < 0.5. Also: at least 100 trades, positive
   expectancy after costs, max drawdown inside the risk limit, results stable when parameters
   move +/-20%.
8. **Monte Carlo.** Shuffle trade order and resample blocks of returns to see the range of
   drawdowns luck alone could produce. Size for the bad end of that range.
9. **Paper trading.** At least 30 trading days (and 50 trades for day trading) with live data
   and simulated fills. Live results must sit inside the backtest's Monte Carlo band.
10. **Small live, then scale.** Only after paper results match.

## Common ways backtests lie

- **Look-ahead bias** - using information not available at decision time (today's close for
  today's entry, restated fundamentals, future index membership).
- **Survivorship bias** - testing only companies that still exist.
- **Overfitting** - many parameters fitted to noise; brilliant in-sample, poor out-of-sample.
- **Ignoring costs** - a day-trading edge of 0.05% per trade disappears under 0.20% round-trip
  costs.
- **Data snooping** - trying hundreds of ideas and reporting the best one.
- **Regime dependence** - a strategy tested only in a bull market.

## Labels and machine learning

- **Triple-barrier labels.** For each entry, a profit barrier, a stop barrier and a time limit;
  the label is whichever is hit first. Matches how the bot actually exits.
- **Meta-labelling.** A primary rule proposes trades; a classifier (LightGBM) predicts which
  proposals will hit the profit barrier and sizes or skips them. Trained walk-forward only.
- **Self-improvement loop.** Retrain on a schedule with new data, but the new model replaces the
  old one only if it beats it out of sample and passes the same gates. Otherwise the old model
  stays. This is how the system "gets better" without fooling itself.
- Fine-tuning a language model does not make price predictions better. The local LLM explains,
  summarises news and argues bull and bear cases; numbers come from tested models.

## Exits matter as much as entries

Every trade carries a stop, a target and a written invalidation before entry. The exit manager
checks stop first (worst case), then invalidation, target, time stop, day-trade flatten and
trailing-stop updates. Stops only move up.
