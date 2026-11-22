# Trading basics and fundamentals

Knowledge-base file for the local chatbot. Plain explanations; formulas live in
`02-quant-math.md`, strategies and testing in `03-strategies-and-backtesting.md`.

## Markets, prices and orders

- **Bid / ask / spread.** The bid is the best price a buyer will pay now, the ask the best price
  a seller will accept. Spread = ask - bid. Buying at the ask and selling at the bid costs the
  spread on every round trip, so frequent trading pays it many times.
- **Market order.** Fills now at whatever price is available. Certain fill, uncertain price.
- **Limit order.** Fills only at your price or better. Certain price, uncertain fill.
- **Stop order.** Becomes a market order once a trigger price trades. Used for stop-losses.
  In a gap (price jumps past the stop overnight) the fill can be far worse than the stop.
- **Slippage.** The difference between the price you expected and the price you got. Larger in
  thin markets, fast markets and for big orders. Backtests must charge it.
- **Liquidity.** How much can be traded without moving the price. Measured with average daily
  traded value, spread and order-book depth.
- **Fees.** Commission, exchange and regulatory fees, and for crypto the taker/maker fee. The
  platform charges 0.10% per side by default in backtests.

## Sessions and settlement

- US stocks trade 09:30-16:00 New York time on business days; pre- and post-market sessions
  are thinner and costlier.
- US equities settle T+1 (one business day after the trade). Spot crypto settles on the
  exchange ledger at the trade and trades 24/7.
- A cash account may only buy with settled cash. The halal profile always uses a cash account.

## Bars, candles and timeframes

- A **bar** summarises a period: open, high, low, close (OHLC) and volume. Candlestick charts
  draw the same data.
- **Timeframe** is the bar length: 1 minute, 5 minutes, 1 hour, 1 day, 1 week.
- Shorter bars mean more trades, more noise and more cost per unit of edge. Day trading uses
  1-15 minute bars; swing trading uses daily bars.
- A bar is only known after it closes. Using its close to decide a trade inside the same bar is
  look-ahead bias.

## Percentages, percentiles and ranks

- **Percentage change** = (new - old) / old. A 10% gain then a 10% loss leaves you at 0.99, not
  1.00: percentages do not add.
- **Percentile.** The value below which a given share of observations fall. The 95th percentile
  of daily losses is the loss exceeded on only 5% of days.
- **Percentile rank.** Where today's value sits in its history, from 0 to 100. Used to say a
  volatility reading is "high" relative to the past year rather than in absolute terms.

## Fundamental analysis (what a business is worth)

- **Revenue growth** - is the business selling more each year?
- **Margins** - gross, operating and net profit as a share of revenue.
- **EPS** (earnings per share) = net income / shares outstanding.
- **P/E** = price / EPS. High P/E means the market expects growth; it is not cheap or dear on its
  own, only relative to growth and peers.
- **Free cash flow** = operating cash flow - capital expenditure. Cash a business can return.
- **Debt and interest** - for the halal screen, interest-bearing debt must stay under 30% of
  market capitalisation (AAOIFI Standard 21).
- **Earnings dates** move prices sharply. Day-trading systems usually stand aside around them.

## Technical analysis (what price and volume are doing)

- **Trend** - a series of higher highs and higher lows (up) or the reverse. Moving averages
  smooth price to show it: price above a rising 200-day average is a common uptrend filter.
- **Support and resistance** - price zones where buying or selling has repeatedly appeared.
  Subjective; the code uses objective versions such as N-day highs and lows (Donchian channels).
- **Momentum** - assets that rose most over the past 3-12 months have, on average, kept rising
  over the following months. This is one of the best-documented effects in finance.
- **Mean reversion** - short-term extremes (1-5 days) tend to partly reverse, especially in
  liquid stocks and indices.
- **Volatility** - how much price moves. ATR (average true range) measures it in price units;
  stops and position sizes are set from it.
- **Volume** - confirms moves: breakouts on high volume are more reliable than on low volume.

## What evidence supports, and what it does not

- Well supported across decades and markets: trend following, cross-sectional momentum,
  diversification, low costs, risk control.
- Weak or no evidence: Fibonacci levels, most chart patterns judged by eye, "guaranteed"
  signal groups, high win-rate claims without a full record.
- Any strategy's edge shrinks after costs and after it becomes popular. Test before trusting.
