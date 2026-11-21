# Changelog

## Unreleased — halal agent layer (branch feat/halal-council-protections)

### Added

- Shariah gate (AAOIFI SS21 screen, purification, no derivatives/margin/shorting), Freqtrade-style
  protections, pre-registered trade plans, analyst council and entry pipeline.
- Exit manager, trade simulator, Alpaca/EDGAR/KMI adapters, Ollama analysts, meta-labelling and
  the reproducible halal backtest with Deflated Sharpe / PBO gate.
- Spot crypto (token screen) and day trading (flatten before close, T+1 settlement, intraday and
  24/7 crypto bars) under the owner-selected profile. Futures stay blocked (IIFA Resolution 63).
- Local knowledge base and cited RAG chatbot, read-only dual-era MCP server, `docs/MASTER_PLAN.md`,
  `GUARDRAILS.md` and `AGENTS.md`.
- Agent board and two-way MCP: research tasks, seven analyst roles filled by Claude, ChatGPT or
  local Qwen, paper proposals with two-source consensus, Streamable HTTP transport with token
  auth for remote connectors.
- Profit engine: edge-over-cost pipeline stage, maker-first cost model, risk dial
  (conservative / balanced / growth), risk-parity sleeve allocator, deposit planner and an MCP
  `deposit_projection` tool. Guardrail 10 now allows 2% risk per trade, but only when the owner
  selects the Growth dial; the default stays 1%.

## 1.0.0 — 2026-09-14

First research + paper-trading release.

### Added

- Multi-asset market-data contracts and validation.
- Quant mathematics, portfolio analytics and realistic backtesting.
- Right-Fit anti-overfitting engine.
- Multi-family strategy factory.
- ML/DL/foundation-model research lab.
- Regime intelligence and dynamic strategy weighting.
- Portfolio optimization and independent risk engine.
- Heatmaps, analytics and responsive trading terminal.
- Paper broker, OMS, reconciliation and paper-order API.
- News credibility/sentiment baseline and Academy surface.
- Adversarial release suite, static safety gate, CI vulnerability scanning configuration and Dependabot.

### Safety

- Live adapters remain disabled.
- Live-mode configuration requires production mode plus explicit risk acknowledgement.
