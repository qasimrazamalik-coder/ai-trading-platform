# Halal council, protections and trade plans

Features adopted from the strongest open-source and benchmarked trading agents, adapted to a
long-only, unlevered, Shariah-compliant system.

| Feature | Source | Module |
|---|---|---|
| Pre-registered exit plan: profit target, stop loss, written invalidation condition, confidence | nof1 Alpha Arena season 1 action format | `quant_os_core.agents.trade_plan.TradePlan` |
| Low trade frequency, no pyramiding (one position per symbol) | Alpha Arena winner Qwen3-Max (about 43 trades in 17 days) and the arena's no-pyramiding rule | `PlanGovernor` |
| Size from a fixed risk budget scaled by confidence, never chosen by an LLM | Risk research (fixed-fractional sizing) | `size_plan` |
| Analysts, bull/bear debate, risk review, portfolio approval | TradingAgents (TauricResearch) | `quant_os_core.agents.council` |
| StoplossGuard, MaxDrawdown, LowProfitPairs, CooldownPeriod | Freqtrade built-in Protections | `quant_os_core.risk_engine.protections` |
| Shariah rules R1-R17 (AAOIFI SS21 ratios, purification, no leverage, no shorting, no derivatives) | Research report section 4 | `quant_os_core.shariah` |

## Order of checks

`EntryPipeline.decide` runs: plan validation, Shariah gate, protections, frequency governor,
analyst council, sizing, then the existing `RiskEngine`. Each stage can stop a trade; none can
enlarge what an earlier stage allowed. Every decision is kept in `audit_log`.

## Deliberate differences from the sources

- No leverage field (the arena allowed up to 20x; this system is fixed at 1x).
- The council can only veto or shrink a trade; it cannot open one or raise size.
- LLM analyst replies must parse into `AnalystReport` (JSON schema in `ANALYST_JSON_SCHEMA`,
  usable with Ollama structured outputs) or they are discarded.
- Crypto is disabled under the default `PK_HANAFI` profile and forex speculation is blocked.

These rules still need sign-off from a qualified mufti before live use.

## Added in the second pass

| Feature | Module |
|---|---|
| Reproducible halal ensemble backtest with Deflated Sharpe and PBO gate (report section 9H) | `quant_os_core.strategies.halal_ensemble`, `research/backtests/run_halal_v3.py` |
| Exit manager: stop, profit target, trailing stop, time stop, machine-checkable invalidation rules | `quant_os_core.agents.exits` |
| Trade-level simulator that replays the live pipeline, exits and protections over history | `quant_os_core.agents.simulator` |
| Free data: Alpaca daily bars, SEC EDGAR fundamentals for automatic AAOIFI screening, PSX KMI list loader | `quant_os_core.market_data.free_sources` |
| Local LLM analysts through Ollama structured outputs (temperature 0, fail closed) | `quant_os_core.agents.ollama` |
| Triple-barrier labels and walk-forward meta-label filter | `quant_os_core.ml.meta_labeling` |

To reproduce the backtest: on a machine with internet run
`python research/backtests/run_halal_v3.py fetch`, then `python research/backtests/run_halal_v3.py run`.
The command exits non-zero when the Deflated Sharpe / PBO gate fails.

## Futures: checked and kept blocked

IIFA Resolution 63 (1/7), 14 May 1992, says deferred-delivery commodity contracts are "not
permissible in Shariah because of the postponement of the two elements of the exchange", that
the most prevalent commodity-market contracts (closed out by an opposite contract) are
"essentially not permissible", that options are not permissible, and that index trading is
"pure gambling" (https://iifa-aifi.org/en/32438.html). AAOIFI SS21 clauses 3/12-3/14 also
prohibit futures, options and swaps on shares. The only route the resolution offers is
restructuring as a *salam* sale, which retail futures markets do not provide. Futures,
perpetuals, options and CFDs therefore remain blocked by rule R2.

## Not built yet

Paper-trading loop against Alpaca's paper API, Telegram alerts and the Next.js dashboard.
