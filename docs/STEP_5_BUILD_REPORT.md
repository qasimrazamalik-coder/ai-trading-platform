# Step 5 Build Report — Right-Fit / Anti-Overfitting Engine

Overall project target after this step: **53%**.

## Delivered

- purged K-fold time-series validation;
- configurable embargo windows;
- expanding and rolling walk-forward splits;
- combinatorial purged cross-validation (CPCV);
- label-horizon overlap purging;
- nanosecond-safe monotonic timestamp guards;
- walk-forward parameter/model selection;
- underfit / right-fit / overfit classification;
- Probabilistic Sharpe Ratio;
- Deflated Sharpe Ratio and expected-best-Sharpe multiple-testing benchmark;
- Sidak family-wise p-value adjustment;
- CSCV-style Probability of Backtest Overfitting;
- 2-D parameter-surface stability diagnostics;
- complexity-penalized research scoring;
- moving-block bootstrap and Monte Carlo robustness summaries;
- conservative research pass / review / reject policy;
- Right-Fit API capability discovery.

## Verification

- full regression suite: **199 passed, 0 failed**;
- measured Right-Fit + API coverage: **98%**;
- measured full-project Python coverage: **97%**;
- Python compilation: PASS;
- core editable-install + API source-root import smoke: PASS;
- source line-length gate (100 chars): PASS;
- nanosecond timestamp precision regression: PASS;
- deterministic walk-forward fixture: PASS;
- leakage/purge/embargo edge-case suite: PASS.

`ruff` and `mypy` remain configured CI gates. Their executables are not installed in the isolated
build runtime, so they are not claimed as locally executed checks.
