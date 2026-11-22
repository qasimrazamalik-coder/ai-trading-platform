# Quant mathematics: the formulas the system uses

Each formula says what it measures, how to compute it and where the code uses it.
`r` is a period return, `P` a price, `E` equity, `p` a probability.

## Returns and compounding

- Simple return: `r = P_t / P_(t-1) - 1`.
- Log return: `l = ln(P_t / P_(t-1))`. Log returns add over time; simple returns compound.
- Compounding: `E_n = E_0 * (1 + r)^n`.
- CAGR (compound annual growth rate): `CAGR = (E_end / E_start)^(1 / years) - 1`.
- Periods needed to grow from `E_0` to target `T` at return `r` per period:
  `n = ln(T / E_0) / ln(1 + r)`.
  Example, $50 to $1,000,000 (a factor of 20,000, ln = 9.90):
  - 5% a month for ever: 203 months, about 17 years;
  - 10% a month for ever: 104 months, about 8.7 years;
  - 20% a year (top-fund level): 54 years.
  Monthly deposits shorten this far more than any strategy can. No system compounds 10% a
  month for years; treat such claims as false.

## Volatility and the cost of swings

- Volatility (standard deviation of returns): `sigma = sqrt( sum (r_i - mean)^2 / (N - 1) )`.
- Annualise from daily: `sigma_year = sigma_day * sqrt(252)` (stocks) or `sqrt(365)` (crypto).
- Volatility drag: geometric mean is about `arithmetic mean - sigma^2 / 2`. +50% then -50%
  averages 0% but leaves 75% of the money. Lower volatility at the same average return grows
  faster.
- Drawdown recovery: after a loss `d`, the gain needed to get back is `d / (1 - d)`.
  10% loss needs 11.1%; 30% needs 42.9%; 50% needs 100%; 80% needs 400%.

## Risk-adjusted performance

- Sharpe ratio: `(mean(r) - r_f) / sigma(r) * sqrt(periods per year)`. Halal accounts hold no
  interest, so the code uses `r_f = 0`.
- Sortino ratio: like Sharpe but divides by downside deviation (only returns below 0).
- Maximum drawdown: `max over t of (peak_t - E_t) / peak_t`, where `peak_t` is the highest
  equity so far.
- Calmar ratio: `CAGR / |max drawdown|`.
- Ulcer index (depth and length of drawdowns): `sqrt( mean( drawdown_t^2 ) )`.
- Smoothness of the equity curve: R-squared of `ln(E_t)` regressed on time. 1.0 is a straight
  compounding line; values under about 0.8 mean lumpy, luck-driven results.
- Deflated Sharpe ratio (Bailey and Lopez de Prado): the probability the true Sharpe is above
  zero after correcting for the number of strategies tried, skew and fat tails. The gate needs
  at least 0.95.
- Probability of backtest overfitting (PBO): the share of train/test splits in which the
  in-sample winner ranks below the median out of sample. The gate needs under 0.5.

## Trade statistics

- Win rate `p` = winning trades / all trades.
- Payoff ratio `b` = average win / average loss.
- Expectancy per trade: `E = p * avg_win - (1 - p) * avg_loss`. Must be positive after costs.
  A 30% win rate is fine with a payoff above 2.4; a 70% win rate loses money with payoff 0.4.
- R-multiple: profit or loss divided by the amount risked at entry (`entry - stop`).
- Kelly fraction: `f* = p - (1 - p) / b`. The growth-optimal bet, but estimates of `p` and `b`
  are noisy, so practitioners bet a quarter to a half of Kelly. The code caps risk per trade at
  1% (0.5% for day trading) instead.
- Chance of `n` losses in a row: `(1 - p)^n`. With `p = 0.4`, ten losses in a row happen with
  probability 0.6%, which over a thousand trades is almost certain to occur somewhere.

## Position sizing

- Fixed-fractional: `quantity = equity * risk_fraction / (entry - stop)`.
  $50 equity, 1% risk, entry $100, stop $95 -> risk $0.50 / $5 = 0.1 share.
- The code then caps the position at 20% of equity and at free cash, and rounds down.
- Volatility targeting: `weight = target_vol / asset_vol`, capped at 1 (no leverage).

## Statistics used in signals

- Simple moving average: `SMA_n = (P_t + ... + P_(t-n+1)) / n`.
- Exponential moving average: `EMA_t = a * P_t + (1 - a) * EMA_(t-1)`, `a = 2 / (n + 1)`.
- True range: `TR = max(H - L, |H - C_prev|, |L - C_prev|)`; ATR is its 14-period Wilder average.
- RSI: `RSI = 100 - 100 / (1 + RS)`, `RS = average gain / average loss` over 14 periods.
- z-score: `z = (x - mean) / sigma`. |z| above 2 is unusual under a normal distribution, but
  market returns have fat tails, so extreme moves come more often than the normal predicts.
- Correlation between assets: covariance divided by the product of their volatilities. Two
  strategies with correlation near 1 are one strategy, not diversification.
- Beta: `cov(asset, market) / var(market)`.

## Geometry on charts

- A trendline's slope is `(P_2 - P_1) / (t_2 - t_1)`; on a log-price chart the slope is a growth
  rate, which is why long-term charts use log scale.
- Channels (Donchian, Keltner, Bollinger) are geometric bands around price built from highs/lows,
  ATR or standard deviation. They are objective and testable; hand-drawn shapes are not.
