# Step 9 Build Report — Portfolio Construction + Independent Risk

Overall project target after this step: **85%**.

## Delivered

- Minimum variance and maximum-Sharpe portfolio optimization.
- Inverse-volatility allocation and risk parity with custom budgets.
- Hierarchical risk parity and HERC-style hierarchical equal-risk allocation.
- Black-Litterman posterior mean/covariance blending.
- Historical-scenario CVaR minimization.
- Allocation summary with expected return, volatility and Sharpe.
- Independent pre-trade risk engine with approve/resize/reject/halt decisions.
- Position, gross/net leverage, group and correlated-exposure limits.
- Portfolio heat, daily-loss and drawdown brakes.
- Global kill switch, stale-data halt and minimum-equity halt.
- Explicit support for risk-reducing orders while a portfolio is already outside limits.
- Stress-scenario, heat, leverage and correlation-concentration analytics.
- FastAPI capability discovery endpoint.

## Safety boundaries

- Strategy intent cannot bypass independent risk approval.
- A research pass does not authorize live execution.
- Live trading remains disabled by default.
- Quantity reductions are rounded conservatively when a risk cap requires resizing.
- Risk controls permit de-risking rather than trapping an already-breached portfolio.

## Verification

- Full platform tests: **400 passed, 0 failed**.
- Focused portfolio/risk/API coverage: **97%**.
- Overall measured Python coverage: **96%**.
- API portfolio-risk capability endpoint coverage: **100%**.
- AST parsing, Python compilation and 100-character source rule: **PASS**.
- Editable install, core optimizer/risk imports and API smoke test: **PASS**.
