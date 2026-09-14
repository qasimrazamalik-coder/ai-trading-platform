# Mathematical Quant Engine

Step 3 provides deterministic numerical primitives used by research, backtesting, portfolio,
risk and future strategy modules.

## Design rules

1. Calculations reject NaN/infinite inputs instead of silently propagating them.
2. Price-to-return transformations require positive prices.
3. Simple returns below -100% are rejected as impossible.
4. Ratios that are mathematically undefined at zero volatility/downside risk raise explicit errors.
5. VaR and CVaR use a **positive loss fraction** convention: `0.03` means a 3% loss.
6. Drawdowns are **non-positive fractions**: `-0.20` means a 20% drawdown.
7. Annualized ratios require an explicit or documented `periods_per_year` assumption.
8. Covariance matrices are checked for shape, symmetry and positive-semidefiniteness before use.
9. Financial execution/accounting values remain Decimal-based at boundaries; vectorized research
   mathematics uses validated float64 arrays for numerical performance.

## Modules

- `returns.py` — simple/log returns, compounded wealth and drawdown.
- `statistics.py` — mean, variance, covariance, correlation, z-score and rolling statistics.
- `volatility.py` — annualized, rolling and RiskMetrics-style EWMA volatility.
- `performance.py` — annualized return, Sharpe, Sortino, Calmar, Omega and trade metrics.
- `risk.py` — historical and Gaussian VaR/CVaR.
- `regression.py` — OLS, beta/alpha, tracking error, information ratio, Engle-Granger and half-life.
- `sizing.py` — Kelly/fractional Kelly, volatility targeting and fixed-fractional sizing.
- `portfolio.py` — covariance/correlation matrices, portfolio risk decomposition and PSD repair.

## Important non-goals at Step 3

- Backtest accounting and transaction-cost simulation belong to Step 4.
- Deflated Sharpe, PBO, purged CV and model-selection diagnostics belong to Step 5.
- Portfolio optimization solvers such as HRP/risk parity/Black-Litterman belong to Step 9.
