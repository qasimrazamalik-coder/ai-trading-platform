# What the research found: edge hunt, diversified core, leverage and growth limits

Numbers below come from the repo's research scripts run on `data/ohlc` (daily closes up to
20 Nov 2026), with next-day execution and costs. They are measurements of the past, not
forecasts. Re-run the scripts to refresh them.

## Edge hunt: no signal beat diversified holding

Script: `research/backtests/edge_hunt.py --dir data/ohlc`. Each signal is computed at the
close of day t and traded at the close of day t+1, costs are charged on turnover (0.10% per
side for ETFs, 0.25% for crypto), the winner is picked on data up to 2018-12-31 only and the
years after are a holdout. A book passes only if deflated Sharpe (DSR) >= 0.95, PBO < 0.5 and
the pick beats equal weight on the holdout Sharpe.

- ETF book (11 ETFs, 2005-12-19 to 2026-11-20): the in-sample pick was `inverse_vol_trend`
  (out-of-sample CAGR 8.2%, Sharpe 0.87, worst drawdown -13.6%). DSR 0.992 but PBO 0.76 and
  it did not beat equal weight out of sample (equal weight: 15.2% CAGR, Sharpe 0.92,
  drawdown -32.5%). Gate: FAIL.
- Crypto book (SPY, GLD, BTC-USD, 2015-10-16 to 2026-11-20): the pick was
  `btc_vol_target_trend`. DSR 0.990 but PBO 0.61 and it did not beat equal weight out of
  sample. Gate: FAIL.
- Earlier runs (commit 167ee46): the halal ensemble from `run_halal_v3.py` had PBO 0.86 and
  the neural confidence layer had AUC about 0.49 (no better than a coin flip). Both failed
  their gates, and no popular open-source strategy in `reference_strategies.py` beat
  diversified holding out of sample.

Lesson: when nothing passes, the honest default is a diversified, low-cost allocation, not a
forecast. The gates were not loosened to make a strategy pass.

## Diversified core: growth and defensive modes

Code: `packages/core/src/quant_os_core/strategies/diversified_core.py`. A monthly, long-only,
unlevered allocation across halal ETFs with inverse-volatility weights (63-day volatility).

- **Growth mode**: inverse-volatility weights across the universe (risk parity without
  leverage). Out of sample 2019-2026: 14.3% CAGR, Sharpe 0.93, worst drawdown -30.7%.
- **Defensive mode**: the same weights, but a fund at or below its 200-day average is held
  as cash. Out of sample 2019-2026: 8.2% CAGR, Sharpe 0.87, worst drawdown -13.6%.

Defensive gives up return for a much smaller worst drawdown. Rebalancing happens at most once
per calendar month and only when a fund is more than 2 points off its target weight (or its
target is zero), with no order under $1. Every order still passes the Shariah gate; buys also
pass the protections and the risk engine.

## Shariah ETF backtest and robustness

Script: `research/backtests/core_report.py --dir data/ohlc` (cost 0.10%).

- Five Shariah ETFs without UMMA (2022-01-31 to 2026-11-20): equal weight 7.5% CAGR, Sharpe
  0.66, drawdown -19.4%; growth 6.3%, 0.79, -13.6%; defensive 4.0%, 0.67, -7.8%.
- Six funds with UMMA (shorter, 2023-02-09 on): equal weight 12.7%, 1.07, -13.8%; growth
  10.6%, 1.20, -9.5%; defensive 7.1%, 0.97, -8.8%.
- Robustness on the 11 original ETFs: the defensive core's Sharpe stays between 0.74 and 0.83
  and its worst drawdown between -15.0% and -16.5% across volatility windows 21/63/126 and
  trend windows 100/150/200. The shipped settings (63, 200) were not re-picked from this grid.

## Why $50 to $1M is not realistic

Turning $50 into $1,000,000 is a factor of 20,000. At a constant 20% a year (top-fund
level) that takes about 54 years (`n = ln(20000) / ln(1.2)`); 5% a month for ever would take
about 17 years, and no system has sustained that.

The core report's block bootstrap (`core_report.py`, section 3: 20,000 resampled paths of the
2005-2026 daily returns, mean block 21 days) gave, for $50:

| Strategy | Horizon | Worst 5% | Median | Best 5% | P(loss) | P($1M) |
|---|---|---|---|---|---|---|
| equal weight | 1 year | 43 | 56 | 69 | 19.4% | 0.0% |
| equal weight | 5 years | 50 | 86 | 140 | 5.2% | 0.0% |
| growth | 1 year | 44 | 56 | 67 | 18.9% | 0.0% |
| growth | 5 years | 51 | 85 | 133 | 4.7% | 0.0% |
| defensive | 1 year | 47 | 54 | 62 | 17.4% | 0.0% |
| defensive | 5 years | 54 | 74 | 100 | 2.2% | 0.0% |

Not one of the 20,000 paths reached $1M in any row (0%). Growing the account depends far more
on monthly deposits and time than on the strategy; see the `deposit_projection` tool.

## Leverage reality on bitcoin

Script: `research/backtests/leverage_reality.py --csv data/ohlc/BTC-USD.csv` (BTC daily,
2014-09-17 to 2026-11-21). For a long opened at a daily close, the share of days whose next
low alone would liquidate it, ignoring fees and funding (which make it worse):

| Leverage | Liquidated by a drop of | Days that do it | Chance to survive 30 days |
|---|---|---|---|
| 2x | 50% | 0.0% | 100.0% |
| 5x | 20% | 0.2% | 94.7% |
| 10x | 10% | 2.2% | 50.9% |
| 20x | 5% | 10.3% | 3.9% |
| 50x | 2% | 34.5% | 0.0% |
| 100x | 1% | 55.7% | 0.0% |
| 200x | 0.5% | 73.2% | 0.0% |

Leveraged futures and perpetuals stay blocked on Shariah grounds (IIFA Resolution 63, AAOIFI
Standards 1 and 21); this table only shows they also fail on risk. The halal alternative,
spot "buy the dip" with no leverage, was also tested: holding BTC returned 64.5% CAGR with a
-83.4% worst drawdown; the best dip rule (buy after a 20% fall, hold 365 days) had 65.9% CAGR,
Sharpe 1.12 vs 1.09 for holding, and the same -83.4% drawdown. Spot crypto removes
liquidation, not the size of the falls.
