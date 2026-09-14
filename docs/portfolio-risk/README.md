# Portfolio Construction and Independent Risk

Step 9 separates portfolio construction from strategy generation and execution.

## Portfolio construction

Implemented allocators:

- inverse volatility;
- minimum variance;
- maximum Sharpe;
- risk parity with custom risk budgets;
- hierarchical risk parity (HRP);
- hierarchical equal-risk contribution (HERC-style recursive allocation);
- Black-Litterman posterior return/covariance blending;
- historical-scenario CVaR minimization.

All optimizers validate finite inputs, covariance dimensions and long-only feasibility. Covariance
matrices are repaired to positive semidefinite form where appropriate.

## Independent pre-trade risk

Strategy output is an `OrderIntent`. The independent risk engine can return:

- `approve`;
- `resize`;
- `reject`;
- `halt`.

Controls include:

- maximum position fraction;
- gross and net leverage;
- daily-loss brake;
- drawdown brake;
- portfolio heat;
- group/sector concentration;
- correlation concentration;
- shorting policy;
- stale-data halt;
- global kill switch;
- minimum-equity threshold.

A critical invariant is that risk-reducing orders remain permissible when the portfolio is already
outside a limit, provided the proposed order reduces the breached risk metric.

## Risk analytics

The library also exposes portfolio heat, gross/net leverage, correlated-concentration scoring and
named portfolio stress scenarios.

## Safety boundary

This layer does not place orders. Execution remains a separate subsystem, and live trading remains
disabled by default.
