# Quant

The production mathematical engine currently lives in `packages/core/src/quant_os_core/quant` so
research, risk and backtesting can import one tested implementation rather than duplicate formulas.

Implemented in Step 3:

- returns and drawdown mathematics;
- descriptive and rolling statistics;
- historical/EWMA volatility;
- Sharpe, Sortino, Calmar, Omega and trade metrics;
- historical/Gaussian VaR and CVaR;
- OLS, beta/alpha, tracking error and information ratio;
- Engle-Granger cointegration foundation and mean-reversion half-life;
- Kelly, fractional Kelly, volatility targeting and fixed-fractional sizing;
- portfolio covariance/correlation and component risk contribution;
- positive-semidefinite covariance repair.

See `docs/quant/MATH_ENGINE.md` for conventions and numerical safeguards.
