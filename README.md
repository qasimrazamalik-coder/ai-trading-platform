# Quant OS

Open, self-hosted multi-asset quantitative trading research and execution platform.

> Status: **Quant OS v1.0.0 research + paper release — defined v1 scope 100% complete.** Live adapters remain intentionally disabled.

## Principles

- Deterministic financial core
- Typed domain contracts
- Strategy/risk/execution separation
- Paper-first execution
- Reproducible research
- Zero required paid software dependencies
- Security and auditability by default

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
pytest -q
uvicorn quant_os_api.main:app --app-dir apps/api/src --reload
```

Open `http://127.0.0.1:8000/health`. Capability discovery is available across market data, quant, backtesting, Right-Fit, strategies, ML, regimes, portfolio/risk, analytics, execution and intelligence endpoints. The terminal state is exposed at `/terminal/demo-state`; paper orders use `/paper/orders` and `/paper/account`.

## Plan, guardrails and the local assistant

- `docs/RUNBOOK.md` — paper trading, alerts, dashboard, kill switch and recovery
- `docs/MASTER_PLAN.md` — build plan, backtesting gates and honest return expectations
- `GUARDRAILS.md` — hard Shariah, risk and AI limits; `AGENTS.md` — agent harness and rules for AI coding assistants
- `docs/knowledge/` — knowledge base for the local chatbot: `python -m quant_os_core.knowledge --docs docs "question"`
- Read-only MCP server for AI assistants: `python -m quant_os_core.mcp_server --docs docs`

## Repository layout

- `apps/api` — FastAPI control plane
- `packages/core` — typed domain primitives, market data, quant math, backtesting, events and financial safety types
- `quant` — mathematical-engine documentation; production formulas live in `packages/core/src/quant_os_core/quant`
- `docs/backtesting` — execution assumptions and Step 4 simulator documentation
- `docs/right-fit` — leakage control, DSR/PBO, stability and Step 5 validation documentation
- `docs/strategies` — Step 6 strategy formulas, contracts and research/execution adapters
- `packages/core/src/quant_os_core/strategies` — production strategy factory
- `packages/core/src/quant_os_core/ml` — ML/DL research, model registry and foundation adapters
- `docs/ml` — Step 7 leakage controls, model families and governance
- `packages/core/src/quant_os_core/regimes` — Step 8 market-regime intelligence
- `docs/regimes` — regime formulas, inference rules and allocation boundaries
- `packages/core/src/quant_os_core/risk_engine` — independent portfolio/pre-trade risk engine
- `packages/core/src/quant_os_core/portfolio` — production portfolio construction algorithms
- `docs/portfolio-risk` — Step 9 allocation/risk formulas and safety boundaries
- `packages/core/src/quant_os_core/execution` — OMS, deterministic paper broker and reconciliation
- `adapters` — documented venue/broker adapter contracts; live connectors are disabled by default
- `packages/core/src/quant_os_core/intelligence` — news credibility, deduplication and sentiment baseline
- `research` — experiments and validation workspace
- `storage` — PostgreSQL/ClickHouse schemas and persistence contracts
- `infrastructure` — local infrastructure definitions
- `docs` — architecture and tutorials

## Safety invariant

No shipped adapter can place a live order. Any future live deployment must pass the explicit production interlock and independent risk approval.


## Step 10 — Heatmaps + Analytics UI

Implemented analytics now include asset/strategy correlations, factor and risk heatmaps,
parameter-stability and walk-forward surfaces, calendar/session seasonality, Monte Carlo and
drawdown distributions, rolling volatility, liquidity and funding series, plus a browser-verified
responsive analytics terminal in `apps/web`. Synthetic UI preview values are explicitly labeled
`DEMO DATA`; live trading remains disabled.


## Step 11 — Execution + Trading Terminal + Intelligence

The platform now includes a deterministic paper broker, idempotent OMS, reconciliation, explicit
multi-market adapter contracts, execution audit schema, credibility-aware news intelligence,
API/WebSocket terminal data and a responsive paper-trading command center with candlesticks, order
book, positions, Strategy Builder and Academy/tutorial views. The order ticket now submits real **paper** market/limit/stop orders through the local API; all live adapters remain disabled.


## v1.0.0 Release

Step 12 adds adversarial execution/accounting tests, fail-closed broker reconciliation, Decimal-only
market-data boundaries, thread-safe paper execution, a functional paper-order API, two-key live-mode
arming, static release scanning, CI vulnerability-audit configuration and dependency update automation.

Run the full local release gate:

```bash
scripts/release_check.sh
```

Frontend:

```bash
npm install --prefix apps/web
npm --prefix apps/web run build
python -m http.server 4173 --directory apps/web
```

**Scope boundary:** 100% refers to the defined v1 research + paper-trading scope. It is not a claim
that any broker-specific real-money adapter has been production-certified.
