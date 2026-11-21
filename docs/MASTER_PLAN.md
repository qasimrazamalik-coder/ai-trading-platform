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

## 1b. Profit upgrades (built) and what they are worth

Estimated with `sim_v5` (20,000 fat-tailed paths, net of costs; assumptions, not a backtest).

| Upgrade | Code | Effect on $50 expected return |
|---|---|---|
| Maker-first execution, commission-free stocks, lower turnover | `CostModel`, `VenueCost` | cost drag 1.6% -> 0.4% a year: **+1.2 points** |
| Edge-over-cost filter: a trade must expect 1.5x its round-trip cost | `check_edge`, pipeline stage `edge` | removes the trades that made day trading lose (-5% to -30% a year in sim_v4) |
| Risk dial: conservative / balanced / growth | `dial_policy`, `RiskDial` | growth doubles risk per trade; returns and drawdowns both rise |
| Four halal sleeves (US stocks, PSX KMI, crypto spot, gold), risk parity, no leverage | `allocate_sleeves` | lower drawdown (5% vs 6% typical yearly) for similar return |
| Monthly deposits | `deposit_plan`, MCP `deposit_projection` | $100/month for 5 years: about $7,300 from $6,050 deposited |
| Day-trading sleeve, **only if paper proves >= 0.30% gross edge per trade** | day-trading module + edge filter | about 22% a year; $50 -> ~$60 in a year, ~$128 in 5 years (median) |

Without a proven day-trading edge, the realistic range stays about 8-10% a year. The edge filter
and paper period decide whether the 22% row is real; nothing else should.

## 1c. Claude + ChatGPT as analysts over MCP (built)

The bot and the assistants talk both ways through the agent board (`agents/board.py`):

- **Bot -> assistants:** research tasks (`get_research_tasks`); assistants can also ask each
  other (`post_research_task`).
- **Assistants -> bot:** analyst reports in seven roles (`submit_analyst_report`) that fill
  council seats; dry runs (`evaluate_trade_plan`); paper proposals (`submit_trade_proposal`)
  that need fresh reports from two different sources and then pass the full pipeline.
- **Local third voice:** `run_local_shift` lets Qwen3 (Ollama, free, offline) answer tasks.

What it adds: research coverage (news, filings, red flags) and more vetoes on bad trades.
What it does not add: better price prediction. In Alpha Arena, LLMs trading on their own mostly
lost money, so LLMs here can only veto, shrink or propose, never approve.

Connect:

1. `export QUANT_OS_MCP_TOKEN=$(python -c "import secrets;print(secrets.token_urlsafe(32))")`
2. `python -m quant_os_core.mcp_server --docs docs --state state/board.db --http 127.0.0.1:8765`
3. HTTPS tunnel: `cloudflared tunnel --url http://127.0.0.1:8765` (free) gives `https://<name>.trycloudflare.com`.
4. **Claude:** Customize > Connectors > Add custom connector, URL `https://<tunnel>/mcp`, static
   header `Authorization: Bearer <token>`. Claude Code / Desktop can use the stdio command instead.
5. **ChatGPT:** Settings > Apps/Connectors > Advanced > Developer mode, create an app with URL
   `https://<tunnel>/mcp/<token>` and "No authentication" (the URL is the secret; developer
   mode needs Plus or Pro). For anything beyond testing, put an OAuth gateway in front.
6. In either chat: "Do your analyst shift on Quant OS: take the open research tasks and report."

## 1d. Intelligence layer (built)

Forecasting (EWMA, GARCH, AR, Holt with honest evaluation), data-science reports, a NumPy
neural network and the neural confidence layer: it learns which breakout trades reach their
target, calibrates that probability, and passes it to the edge filter and sizing. It is only
used after walk-forward gates and champion/challenger. Details:
`docs/knowledge/06-ml-forecasting-neural.md`; training: `docs/RUNBOOK.md` section 6.

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
| 1 | **Paper-trading loop** on Alpaca's paper API: schedule, reconcile positions, write the audit log, kill switch | Built (`ops/paper_loop.py`, `ops/run_paper.py`); acceptance needs 30 days unattended with zero unreconciled positions |
| 2 | **Durable state and memory**: SQLite (orders, fills, positions, decisions, model versions); restart resumes exactly | Built (`ops/state.py`); restart test passes |
| 3 | **Event bus**: bar, signal, order, fill, risk-breach events; handlers idempotent | replaying a day's events twice gives one set of trades |
| 4 | **Error handling**: retries with backoff, circuit breaker, fail closed on unknown state | Built (`ops/resilience.py`); fault tests pass |
| 5 | **File and folder layout**: `data/raw` (immutable downloads), `data/clean`, `models/<version>`, `runs/<date>`, `logs/`; atomic writes (temp file + rename); retention policy | test writes are atomic; nothing written outside these folders |
| 6 | **Self-improvement loop**: champion/challenger gate | Gate built (`ml/champion.py`); weekly retrain job still to schedule |
| 7 | **Dashboard**: equity, drawdown, positions, blocked-by reasons, decisions, events, kill switch | Local dashboard built (`ops/dashboard.py`); Next.js version later |
| 8 | **Alerts**: fills, risk breaches, daily summary | Built with ntfy push, webhooks and email (Telegram is not usable for the owner) |
| 9 | **Chatbot upgrades**: local embeddings fused with BM25; eval set | Built; 44 of 45 eval questions retrieve the right file in the top 3 (BM25 alone) |
| 10 | **MCP upgrades**: read-only tools for audit log, open positions, backtest summaries | tools never mutate state (test) |
| 11 | **PSX support**: hours, T+1 settlement, CSV bars | Built; PSX broker adapter and live data source still missing |
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
