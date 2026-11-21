# Quant OS master plan: halal self-improving trading system

Status key: **Done** (in the repo, tested) · **Next** (build order) · **Later**.
Every item has an acceptance test; nothing counts as done until it passes.

## 0. Ground rules

1. **Halal first.** The Shariah gate runs before every order and its decision is final
   (`GUARDRAILS.md`). Futures, forwards, options, binaries, CFDs, perpetuals, margin and
   shorting are blocked in every profile; checked twice against IIFA Resolution 63 and AAOIFI
   Standards 1 and 21 (`docs/knowledge/05-halal-rules.md`).
2. **Survive, then compound.** Return targets never override the risk limits.
3. **Evidence over belief.** A strategy, model or rule reaches money only after the gates in
   section 4 and a paper-trading period.
4. **LLMs advise, code decides.** Language models explain, summarise and argue; numeric models
   and deterministic rules decide.

## 1. Honest money expectations

| Path | What is realistic | Source |
|---|---|---|
| $50 at 20%/year (top-fund level) | $1M in about 54 years | `n = ln(T/E0)/ln(1+r)`, `02-quant-math.md` |
| $50 at 5%/month for ever | $1M in about 17 years; no system has sustained this | same |
| $50 plus monthly deposits | Deposits dominate the first years | same |

"More money, faster" therefore comes from, in order: (1) not losing it, (2) adding capital,
(3) more uncorrelated strategies, (4) lower costs, (5) better entries. The plan follows that
order.

## 2. Knowledge (what the system and the chatbot know)

| Topic | File | Status |
|---|---|---|
| Basics, fundamentals, orders, bars, percentages, percentiles | `docs/knowledge/01-trading-basics.md` | Done |
| Formulas: compounding, volatility, Sharpe/Sortino/Calmar, drawdown, Kelly, expectancy, sizing, z-scores, smoothness, chart geometry | `docs/knowledge/02-quant-math.md` | Done |
| Strategies, elite backtesting, labels, self-improvement | `docs/knowledge/03-strategies-and-backtesting.md` | Done |
| Psychology of money and trading, mapped to code rules | `docs/knowledge/04-psychology.md` | Done |
| Halal rules with sources | `docs/knowledge/05-halal-rules.md` | Done |
| Glossary (200 terms), market microstructure, PSX specifics | `docs/knowledge/` | Next |

Acceptance: the retrieval test (`tests/knowledge`) finds the right file for each topic.

## 3. Engine (already built, keep hardening)

| Area | Modules | Status |
|---|---|---|
| Quant maths (returns, stats, volatility, performance) | `quant_os_core.quant` | Done |
| Event-driven and vectorised backtesting with costs, slippage, gaps | `quant_os_core.backtesting` | Done |
| Right-fit validation: walk-forward, purged CV, Deflated Sharpe, PBO, Monte Carlo | `quant_os_core.right_fit` | Done |
| Strategy factory and halal ensemble | `quant_os_core.strategies` | Done |
| Shariah gate, token screen, settlement rule | `quant_os_core.shariah` | Done |
| Trade plans, analyst council, entry pipeline, exits, simulator | `quant_os_core.agents` | Done |
| Protections (stop-loss guard, drawdown, low-profit pairs, cooldown) | `quant_os_core.risk_engine` | Done |
| Spot crypto and day trading | `agents/day_trading.py`, `AlpacaCryptoBars` | Done |
| Regimes (HMM/GMM), heatmaps, analytics | `regimes`, `analytics` | Done |
| Meta-labelling (triple barrier + LightGBM) | `ml/meta_labeling.py` | Done |

## 4. Elite backtesting gates (every strategy, every retrain)

1. Point-in-time data; delisted symbols included; adjusted prices.
2. Fills on the next bar; fees, spread and slippage charged (0.10%/side default).
3. Walk-forward with purge and embargo at least the holding period.
4. Every trial logged; Deflated Sharpe >= 0.95; PBO < 0.5.
5. >= 100 trades; positive expectancy after costs; max drawdown within limit;
   +/-20% parameter changes keep it profitable (stability surface).
6. Monte Carlo: 95th-percentile drawdown inside the risk budget.
7. Equity-curve smoothness (R-squared of log equity) >= 0.8 reported.
8. Paper trading >= 30 trading days / 50 trades; live results inside the Monte Carlo band.

Status: gates 1-7 Done in code (`right_fit`, `run_halal_v3.py`); gate 8 needs the paper loop.

## 5. Next build order

| # | Item | Acceptance test |
|---|---|---|
| 1 | **Paper-trading loop** on Alpaca's paper API: schedule, reconcile positions, write the audit log, kill switch | 30 days unattended with zero unreconciled positions |
| 2 | **Durable state and memory**: SQLite (orders, fills, positions, decisions, model versions) with migrations; restart resumes exactly | kill -9 mid-session, restart, state identical |
| 3 | **Event bus**: bar, signal, order, fill, risk-breach events; handlers idempotent | replaying a day's events twice gives one set of trades |
| 4 | **Error handling**: retries with backoff for data/broker calls, circuit breaker after N failures, fail closed (no new entries) on any unknown state | fault-injection tests for timeouts, 429s, bad JSON |
| 5 | **File and folder layout**: `data/raw` (immutable downloads), `data/clean`, `models/<version>`, `runs/<date>`, `logs/`; atomic writes (temp file + rename); retention policy | test writes are atomic; nothing written outside these folders |
| 6 | **Self-improvement loop**: weekly retrain, champion/challenger, promote only if the challenger beats the champion out of sample and passes section 4 | promotion log shows gate results for every swap |
| 7 | **Dashboard** (Next.js): equity, drawdown, open positions, blocked-by reasons, percentiles of returns, heatmaps, audit log, kill switch | Lighthouse a11y >= 90; data matches the database |
| 8 | **Telegram alerts** (or WhatsApp if available): fills, risk breaches, daily summary | alert within 60 s of a fill in paper |
| 9 | **Chatbot upgrades**: embeddings (local, e.g. nomic-embed-text via Ollama) blended with BM25; answers cite sources; evaluation set of 100 Q&A | >= 90% of eval answers cite a correct source |
| 10 | **MCP upgrades**: read-only tools for audit log, open positions, backtest summaries | tools never mutate state (test) |
| 11 | **PSX support**: KMI universe, PSX data source, local settlement and hours | gate blocks non-KMI symbols |
| 12 | **Scholar review**: send `05-halal-rules.md` and the token categories to a mufti | written sign-off stored in `docs/` |

## 6. Intelligence: what "pre-trained, fine-tuned, gets better" means here

- **Numeric models** (LightGBM meta-labeller, regime models) learn from new data through the
  champion/challenger loop in item 6. This is the real "gets better" mechanism.
- **Foundation time-series models** (e.g. Chronos-class) may be tried as features, under the
  same gates; no special trust.
- **The local LLM** (Qwen3 via Ollama) is pre-trained; we do not fine-tune it on prices.
  It improves through better retrieval (knowledge files, embeddings) and an evaluation set.
  Optional later: a small LoRA fine-tune only for output format and tone, never for signals.
- **"High IQ"** = calibrated probabilities, walk-forward evidence, ensembles.
  **"High EQ"** = the psychology rules in `04-psychology.md` enforced in code: no revenge
  trading, no pyramiding, fixed risk, cooldowns, a human kill switch.

## 7. Harness, guardrails and docs

| Doc | Purpose | Status |
|---|---|---|
| `GUARDRAILS.md` | Hard limits no agent, model or config can override | Done |
| `AGENTS.md` | Harness: what each agent may do, decision authority, evaluation gates, how AI coding assistants work in this repo | Done |
| `LICENSE` | Apache 2.0 (existing). Data sources keep their own terms: Alpaca, SEC EDGAR, Yahoo | Done |
| `SECURITY.md`, `docs/CONTRIBUTING.md`, `CHANGELOG.md` | existing | Done |
| `docs/halal-council-features.md` | feature-to-source map | Done |
| `docs/knowledge/*` | chatbot knowledge base | Done |
| Runbook: start, stop, kill switch, recover, rotate keys | `docs/RUNBOOK.md` | Next (with item 1) |

## 8. How to run what exists

```bash
pip install -e '.[dev]'                   # from the repo root
pytest -q                                  # full test suite
python -m quant_os_core.knowledge --docs docs "what is the deflated sharpe ratio?"
python -m quant_os_core.mcp_server --docs docs   # stdio MCP server for AI assistants
python research/backtests/run_halal_v3.py fetch && python research/backtests/run_halal_v3.py run
```

MCP client config example (Claude Desktop, Cursor and similar):

```json
{ "mcpServers": { "quant-os": {
    "command": "python",
    "args": ["-m", "quant_os_core.mcp_server", "--docs", "/path/to/ai-trading-platform/docs"],
    "env": { "PYTHONPATH": "/path/to/ai-trading-platform/packages/core/src" } } } }
```
