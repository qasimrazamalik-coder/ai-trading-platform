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
