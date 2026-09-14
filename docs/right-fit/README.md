# Right-Fit Validation Engine

Quant OS treats every apparent edge as untrusted until it survives leakage-aware validation.
Step 5 does **not** prove a strategy will work live and never authorizes live trading.

## Validation layers

- purged K-fold time-series cross-validation;
- post-test embargo windows;
- expanding and rolling walk-forward splits;
- combinatorial purged cross-validation (CPCV);
- label-horizon purging for overlapping financial labels;
- walk-forward parameter/model selection using training data only;
- Probabilistic Sharpe Ratio (PSR);
- Deflated Sharpe Ratio (DSR) with a multiple-trial benchmark;
- CSCV-style Probability of Backtest Overfitting (PBO);
- Sidak family-wise p-value adjustment;
- 2-D parameter-surface stability scoring;
- moving-block bootstrap / Monte Carlo robustness;
- explicit underfit / right-fit / overfit diagnostic;
- conservative research pass / review / reject policy.

## Core conventions

### Purge and embargo

A test interval is isolated from surrounding training observations. Purge removes observations
immediately before a test interval. Embargo removes observations immediately after it. For labels
that use future information, `purge_label_overlap` removes any training label horizon that crosses
into the test or embargo interval.

### Probabilistic Sharpe Ratio

For periodic Sharpe estimate \(\hat{SR}\), benchmark \(SR^*\), sample size \(T\), skewness
\(\gamma_3\), and Pearson kurtosis \(\gamma_4\):

\[
PSR = \Phi\left(
\frac{(\hat{SR}-SR^*)\sqrt{T-1}}
{\sqrt{1-\gamma_3\hat{SR}+\frac{\gamma_4-1}{4}\hat{SR}^2}}
\right)
\]

Quant OS keeps this statistic in the return series' native periodicity instead of silently mixing
annualized and non-annualized Sharpe ratios.

### Deflated Sharpe Ratio

DSR sets the PSR benchmark to an approximation of the Sharpe that could arise simply from trying
many candidate strategies. It therefore penalizes strategy-selection bias in addition to skewness
and kurtosis.

### Probability of Backtest Overfitting

The PBO implementation partitions observations into an even number of contiguous slices. For every
symmetric train/test combination, it selects the best in-sample strategy and measures its rank out
of sample. PBO is the fraction of selected candidates landing in the lower half OOS.

### Parameter stability

A broad performance plateau is preferred over a single isolated optimum. The stability score
combines:

1. fraction of the surface near the optimum;
2. local stability around the best cell;
3. global smoothness of the surface.

The score is explicitly a diagnostic, not a statistical proof.

## Research gate

The default `RightFitPolicy` is intentionally conservative. It can return:

- `research_pass` — configured research checks passed;
- `review` — no hard failure, but stability/generalization robustness needs attention;
- `reject` — leakage, non-positive OOS performance, excessive PBO or insufficient DSR evidence.

A research pass never enables live trading.
