# Machine learning, neural networks and forecasting in the bot

## What each model is for

| Model | Predicts | Where it is used | Code |
|---|---|---|---|
| EWMA volatility (RiskMetrics, lambda 0.94) | next-bar variance | stop distance in `ForecastBreakout` | `forecasting.ewma_variance` |
| GARCH(1,1), maximum likelihood | variance path that reverts to a long-run level | risk sizing and research | `forecasting.Garch11` |
| AR(p) on returns | next return | research; must beat the naive forecast | `forecasting.ARModel` |
| Holt linear trend on log prices | price path with widening bands | research and charts | `forecasting.HoltLinear` |
| Neural network (MLP, NumPy) + logistic baseline | probability a trade hits its target before its stop | plan confidence, trade filter | `ml.nn`, `ml.neural_signal` |
| PyTorch sequence models (LSTM, GRU, TCN, Transformer, PatchTST, iTransformer) | returns from sequences | research only, needs PyTorch | `ml.deep` |

Volatility is predictable (it clusters); direction mostly is not. So forecasts set stop sizes
and position sizes, and the neural network answers a narrower, more answerable question:
"of the trades this signal proposes, which ones tend to work?" (meta-labelling).

## How the neural confidence layer is trained

1. Replay the primary signal (breakout) over history; each plan it would have made is an event.
2. Label: 1 if the target is reached before the stop within 20 bars; stop first wins ties.
3. Features at the entry bar only: returns over 1/5/20/60 bars, 20-bar volatility and its
   ratio to 60-bar, ATR %, RSI, distance from 20/50/200-bar averages, position in the 20-bar
   range, volatility percentile over a year, volume z-score. A test rewrites future prices and
   checks past features do not change.
4. Walk-forward: train on events whose exits finished before the test window (purging), test
   on the next window, five times. Network (32 and 16 ReLU units, dropout, L2, Adam, early
   stopping on the most recent data) averaged with logistic regression.
5. Platt calibration of the out-of-sample scores, so probabilities mean what they say.

## Gates before the model is used

- AUC at least 0.55 (0.5 is a coin flip) and a better Brier score than the base win rate.
- Skill (log-loss gain over the base rate) on at least 60% of folds.
- Deflated Sharpe of the filtered trades at least 0.95, counting every threshold tried.
- Filtered trades must earn more per trade than taking every signal.
- Champion/challenger: a retrained model replaces the saved one only if it beats it.

On a synthetic market with real regimes the layer passed (AUC 0.675, win rate 63.5% -> 70.6%
on filtered trades); on a pure random walk it was refused. Real-market results come only from
running the trainer on real data.

## Data science checks

`ml.data_science` reports bad prices, OHLC violations, stale runs, time gaps, outlier returns
(robust z over 8), the return profile (skew, fat tails, percentiles, volatility clustering) and
each feature's information coefficient in both halves of history, plus redundant feature pairs.

## Honest limits

- A neural network is only as good as the signal in the data; on noise it learns noise, which
  is why the gates exist.
- Logistic regression sometimes matches or beats the network; both are kept and averaged.
- More layers do not mean more profit. Calibration and walk-forward discipline matter more.
