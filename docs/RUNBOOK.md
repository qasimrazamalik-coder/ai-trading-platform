# Runbook: paper trading, alerts, dashboard, kill switch, recovery

Everything here runs on paper money. Live trading stays disabled behind the two-key interlock.

## 1. One-time setup

1. `pip install -e '.[dev]'` from the repo root.
2. Create a free Alpaca account, open the **paper** dashboard and generate paper API keys.
3. Phone alerts: install the **ntfy** app (Android/iOS), subscribe to a long private topic name
   (anyone who knows the topic can read it). Discord or Slack webhooks also work.
4. Copy `examples/paper.json` to `paper.json` and edit:
   - `universe`: symbols to trade. Shares need `financials` from the latest filing for the
     AAOIFI screen (the example AAPL numbers are placeholders - replace them). Crypto needs a
     `token_category` (e.g. `layer1`; meme, lending, leveraged and gambling tokens are blocked).
   - `dial`: `conservative`, `balanced` or `growth` (growth = 2% risk per trade, opt-in).
   - `day_trading`: `true` switches to intraday limits, flatten-before-close and `5Min` bars
     (also set `timeframe` and `interval_minutes`, e.g. `"5Min"` and `5`).

## 2. Run

```bash
export APCA_API_KEY_ID=...  APCA_API_SECRET_KEY=...      # paper keys only
export QUANT_OS_NTFY_TOPIC=your-long-private-topic         # optional, overrides the file
python -m quant_os_core.ops.run_paper --config paper.json            # loops forever
python -m quant_os_core.ops.run_paper --config paper.json --once     # one step, for cron
```

Each step: kill switch check -> broker snapshot (circuit breaker) -> reconcile positions ->
pending fills -> exits -> new entries (signal -> analyst tasks -> full pipeline -> patient limit
order) -> equity snapshot -> daily summary after the US close.

Windows Task Scheduler or cron can run `--once` on a schedule instead of a forever loop.

## 3. Dashboard and kill switch

```bash
export QUANT_OS_DASHBOARD_TOKEN=$(python -c "import secrets;print(secrets.token_urlsafe(24))")
python -m quant_os_core.ops.dashboard --state state/trading.db     # http://127.0.0.1:8780
```

Shows equity, drawdown, win rate, open positions, why trades were blocked, recent decisions and
events. **Kill switch**: the button asks for the token. When on, the next step cancels pending
entries and sells every open position at market, then stops trading until switched off.

## 4. Analysts (Claude, ChatGPT, local Qwen)

The loop posts review tasks to `state/board.db` for every new trade idea. Start the MCP server
on the same board file (see `docs/MASTER_PLAN.md` section 1c) and ask Claude or ChatGPT to "do
your analyst shift". For the free local analyst:

```python
from pathlib import Path
from quant_os_core.agents.board import AgentBoard
from quant_os_core.agents.board_worker import ollama_json_generate, run_local_shift
run_local_shift(AgentBoard(Path("state/board.db")), ollama_json_generate("qwen3:8b"))
```

Every closed trade grades the analysts who called it; council weights follow the record
(`get_analyst_leaderboard` over MCP).

## 5. When something goes wrong

| Symptom | What the bot does | What you do |
|---|---|---|
| Broker or data API down | Circuit breaker opens; no new entries; critical alert | Wait; it retries after 5 minutes |
| Position mismatch (bot vs broker) | New entries paused; exits continue for agreeing symbols | Compare the Alpaca dashboard with the dashboard positions; fix the record or close the position by hand, then restart |
| Step crashes | Logged and alerted; next step runs normally | Read `events` in the dashboard |
| Machine restarts | Positions, trailed stops, pending orders and history reload from `state/trading.db` | Start the loop again |
| Bad week | Protections lock entries after stop-loss streaks and drawdowns | Do nothing; review the decisions table |

Back up `state/` daily (copy the folder while the loop is stopped, or use `sqlite3 .backup`).

## 6. Train the neural confidence model

```bash
python -m quant_os_core.ml.train_confidence --csv data/AAPL.csv data/BTCUSD.csv \
    --signal donchian --out models/confidence.npz --state state/trading.db \
    --report reports/confidence.md
```

CSV columns: `time,high,low,close` (ISO times with timezone; 5+ years of daily bars per symbol
works best). The command writes a data-science report and saves the model only if it passes
every gate and beats the current champion. Then add to `paper.json`:
`"model": {"path": "models/confidence.npz"}` (optional `"min_probability": 0.55`) and,
for forecast-based stops, `"signal": "forecast"`. Re-run weekly; the old model stays unless
the new one wins.

## 6b. Weekly model retraining (other models)

Score the current and retrained meta-labeller on the same walk-forward folds, compute the
Deflated Sharpe and PBO, then call `quant_os_core.ml.champion.run_gate(...)`. The new model
replaces the old one only if it wins on most folds and passes both gates; every decision is
stored in the `models` table.

## 7. Securing the MCP connectors

The token and secret-path options are for testing. For regular use, put an OAuth gateway in
front of the tunnel so Claude and ChatGPT sign in properly:

- **Cloudflare Access** (free tier) on the tunnel hostname, with an OAuth/OIDC identity provider;
- or **Auth0** as the authorization server, following the MCP authorization spec.

Rotate `QUANT_OS_MCP_TOKEN` immediately if a connector URL leaks.

## 8. PSX

Set `"venue": "PSX"` on a universe row to use Pakistan Stock Exchange hours (Mon-Thu close
15:30, Friday 16:30, Karachi time) and T+1 settlement (since 9 Feb 2026). Ramadan and special
sessions are not modelled. There is no free official PSX market-data API, so give PSX bars as
CSV files (`time,high,low,close`, ISO times with timezone) exported from your broker:
`"bars_csv": {"MEBL": "data/psx/MEBL.csv"}` in `paper.json`. Order placement still needs a
PSX broker adapter; until then PSX runs in research and backtests only.
