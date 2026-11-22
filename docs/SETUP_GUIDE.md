# Setup guide: from a fresh Windows laptop to a running paper bot

Paper trading only. Live trading stays locked behind the two-key interlock, and futures,
margin and short selling stay blocked (see `GUARDRAILS.md`).

## 1. Install

1. Install Python 3.12 from python.org and tick "Add python.exe to PATH".
2. Open PowerShell in the project folder and create a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\pip install -e ".[dev,ml]"
   ```

3. Check the install. The last line should say `passed`:

   ```powershell
   .venv\Scripts\python -m pytest -q
   ```

## 2. Get price data for charts and research

```powershell
.venv\Scripts\pip install yfinance
.venv\Scripts\python research\backtests\run_halal_v3.py fetch --out data\halal_prices.csv
```

Per-symbol CSVs for the dashboard charts go in `data\ohlc\` with the columns
`time,high,low,close` (see `research/backtests/edge_hunt.py` for the expected format).

## 3. Read the research before trading

Each script prints a table and a PASS or FAIL gate:

| Script | What it answers |
|---|---|
| `research\backtests\run_halal_v3.py run` | Does the original ensemble pass the overfitting gates? (No) |
| `research\backtests\edge_hunt.py` | Does any halal allocation beat diversified holding out of sample? (No) |
| `research\backtests\core_report.py` | How the diversified core does on Shariah ETFs, and the range of outcomes for $50 |
| `research\backtests\reference_strategies.py` | Do popular open-source bot strategies beat holding? (No) |
| `research\backtests\leverage_reality.py` | What leverage does on real BTC data (liquidation odds) |

## 4. Paper account

1. Sign up at alpaca.markets and open the **Paper** dashboard.
2. Create a **separate** paper account for the diversified core and generate its API keys.
3. In that account's settings, turn on "no shorting". Paper accounts allow margin by default;
   the bot pauses rather than borrows, but this setting closes the door at the broker too.

## 5. Configure

1. Copy `examples\paper_core.json` (Shariah ETFs) to `paper_core.json`, or
   `examples\paper_core_original.json` for the 11 backtested ETFs.
2. Add a `financials` block to every fund: holdings-weighted ratios from the fund's factsheet
   or holdings file (debt, interest-bearing cash, non-compliant revenue, market cap...).
   Without it the Shariah check blocks the fund with rule R6. Do not invent numbers.
3. Pick `core.mode`: `defensive` (holds cash for funds below their 200-day average; smaller
   drops) or `growth` (fully invested; higher return, bigger drops).
4. Optional phone alerts: install the ntfy app, subscribe to a long private topic name and put it
   in `alerts.ntfy_topic`.

## 6. Run

Set the keys for this PowerShell window (do not save them in files you share):

```powershell
$env:APCA_API_KEY_ID = "your-paper-key-id"
$env:APCA_API_SECRET_KEY = "your-paper-secret"
.venv\Scripts\python -m quant_os_core.ops.run_paper --config paper_core.json --once
```

Schedule that last command once a day after the US close with Task Scheduler
(Action: start `.venv\Scripts\python.exe` with the arguments above, "Start in" the project
folder, and the two keys set as user environment variables).

## 7. Dashboard

```powershell
$env:QUANT_OS_DASHBOARD_TOKEN = "a-long-random-string-of-16-plus-characters"
.venv\Scripts\python -m quant_os_core.ops.dashboard --state state\core.db --data data\ohlc
```

Open http://127.0.0.1:8780. Tabs:

- **Overview**: equity, drawdown, open positions, events, the kill switch, and how many trade
  ideas each check stopped (Shariah, protections, risk engine...).
- **Charts**: daily candles with SMA 50/200, Bollinger bands (20, 2) and RSI (14).
- **Order book**: live cumulative crypto depth from Alpaca and the bid-ask spread.
- **Quant metrics**: Sharpe, Sortino, Calmar, drawdown, win rate, profit factor, each with
  its formula.
- **Setup guide**: this page in short.

## 8. Week one checklist

- Every day: the run printed no `issues`, and the dashboard shows no "position mismatch" or
  "unexpected account state" events.
- If something looks wrong, turn the kill switch on from the dashboard and read
  `docs/RUNBOOK.md`, section 5.
