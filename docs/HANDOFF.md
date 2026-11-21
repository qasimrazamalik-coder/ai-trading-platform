# Handoff: state of the halal trading bot (22 Nov 2026)

Branch `feat/halal-council-protections`, 8 commits on top of `889da21`. Full suite: 567 passed,
7 skipped (PyTorch/LightGBM optional). mypy: 55 errors, all pre-existing.

## What is built (commit by commit)

1. Shariah gate, Freqtrade-style protections, Alpha Arena trade plans, analyst council.
2. Reproducible halal backtest (`research/backtests/run_halal_v3.py`), exit manager, simulator,
   Alpaca/EDGAR/KMI adapters, Ollama analysts, meta-labelling.
3. Spot crypto (token screen) and day trading under the owner profile.
4. Master plan, knowledge base (`docs/knowledge/`), local RAG chatbot, MCP server, guardrails.
5. Profit engine: edge-over-cost filter, maker-first costs, risk dial, sleeve allocator,
   deposit planner.
6. Agent board + two-way MCP (Claude, ChatGPT, local Qwen as analysts; paper proposals need
   two sources), Streamable HTTP transport with token auth.
7. Operations: paper loop on Alpaca paper, SQLite state with crash recovery, retries and
   circuit breakers, ntfy/webhook/email alerts, local dashboard, champion/challenger gate,
   analyst grading, PSX hours and T+1, hybrid retrieval, `docs/RUNBOOK.md`.
8. Forecasting (EWMA, GARCH, AR, Holt + honest evaluation), data-science reports, NumPy
   neural network, neural confidence layer feeding plan confidence, `train_confidence` CLI.

## Owner's decisions so far

- Profile: spot crypto ON and day trading ON (`ShariahProfile.spot_crypto_and_day_trading()`);
  the library default stays PK Hanafi (crypto off). Futures confirmed haram, not added.
- Risk dial default balanced (1% per trade); growth (2%) is opt-in.
- Alerts: no Telegram.

## Not verified yet (needs the owner's machine or time)

1. Real-data backtest: `python research/backtests/run_halal_v3.py fetch` then `run`.
2. Paper trading 30 days on Alpaca paper (`docs/RUNBOOK.md`).
3. Neural layer on real history: `python -m quant_os_core.ml.train_confidence --csv ...`.
4. MCP connectors from the real Claude and ChatGPT apps (needs a cloudflared tunnel).
5. Local Qwen worker against a running Ollama.
6. Real AAPL financials in `examples/paper.json` (current numbers are placeholders).
7. Mufti review of `docs/knowledge/05-halal-rules.md` and the crypto token categories.

## Next tasks, in order

1. Run items 1 and 3 above; if a gate fails, fix the strategy, not the gate.
2. Start the paper loop and dashboard; watch for position mismatches in week one.
3. Schedule weekly retraining (Windows Task Scheduler or cron) using `run_gate`.
4. PSX: a data adapter and a broker adapter (orders currently only through Alpaca).
5. OAuth gateway (Cloudflare Access or Auth0) in front of the MCP tunnel.
6. Next.js dashboard to replace the local one when needed.

## Starter prompt for Claude Code

> Read CLAUDE.md and docs/HANDOFF.md. Run the test suite to confirm the baseline (567 passed),
> then help me with "Next tasks" item 1: run the real-data backtest and the neural trainer on
> my laptop, read the reports, and tell me which gates pass.
