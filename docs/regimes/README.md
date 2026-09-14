# Regime Intelligence Engine

Step 8 adds transparent and unsupervised market-regime intelligence without giving regime models direct execution authority.

## Deterministic baselines

- Trend: fast/slow moving-average spread plus slow-trend slope.
- Volatility: rolling realized-volatility quantile state.
- Risk-on/off: relative performance of growth/risk assets versus defensive assets.
- Liquidity: spread, depth and volume composite.
- Correlation: average absolute off-diagonal correlation.
- Macro: growth/inflation quadrant model.
- Volatility clustering: lagged correlation of squared returns.

Every deterministic classifier returns a label, numerical score and confidence.

## Unsupervised detectors

`DiagonalGaussianMixture` implements a diagonal-covariance Gaussian mixture with EM. `GaussianHMMRegimeDetector` implements a diagonal Gaussian hidden Markov model with forward/backward inference, Baum-Welch style parameter updates and Viterbi decoding.

Both are dependency-light and support one-observation inference after fitting, which is required for live market scoring.

State IDs are intentionally treated as latent IDs. Research code must profile state means, variances, persistence and market behavior before assigning economic names.

## Change detection and transitions

- CUSUM structural-shift detector.
- Empirical transition matrices with optional smoothing.
- Stationary regime distribution.
- Normalized entropy and confidence.

## Strategy allocation

`StrategyRegimeMatrix` stores empirical strategy/regime compatibility scores. `dynamic_strategy_weights` conditions a portfolio on one regime axis; `composite_strategy_weights` combines multiple active regime labels with confidence-weighted geometric compatibility.

These outputs are research/allocation inputs. They do not bypass the independent risk layer or enable live trading.

## Heatmap contract

`StrategyRegimeMatrix.heatmap()` returns deterministic row labels, column labels and a numeric matrix suitable for the Step 10 frontend heatmap renderer.

## Anti-overfitting rule

Compatibility scores should be learned only from purged out-of-sample or walk-forward observations. Regime labels, thresholds and state mappings must not be selected using future performance.
