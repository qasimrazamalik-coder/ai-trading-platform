# Psychology of money and trading, and how the system counters it

The ideas below are well-known findings from behavioural finance, summarised in our own words.
The point of an automated system is to take these failure modes out of the decision loop.

## Biases that cost traders money

| Bias | What it looks like | Rule in the code that counters it |
|---|---|---|
| Loss aversion | Holding losers to avoid admitting the loss | Stop-loss is set before entry and cannot be widened |
| Disposition effect | Selling winners early, keeping losers | Trailing stop lets winners run; stops cut losers |
| Overconfidence | Bigger size after a few wins | Fixed-fractional sizing; confidence only scales between 0.5x and 1x |
| Revenge trading | Trading more after a loss to win it back | StoplossGuard and cooldown locks; daily trade cap |
| FOMO | Chasing a move after it has run | Plan must meet minimum reward-to-risk; no entry without a stop |
| Recency bias | Believing the last month will continue | Walk-forward tests across many regimes |
| Confirmation bias | Seeking only supporting news | Bear analyst must argue against every trade; a strong bear case vetoes |
| Gambler's fallacy | "Five losses, a win is due" | Each trade is sized independently of the last result |
| Sunk cost | Adding to a losing position | No pyramiding: one position per symbol |

## Money habits that compound

- **Survival first.** A system that avoids large losses can keep compounding; one that blows up
  restarts from zero. Drawdown limits matter more than return targets.
- **Room for error.** Plans should still work if things go worse than expected: size for the bad
  end of the Monte Carlo range, keep cash for living costs outside the trading account.
- **Time does the heavy lifting.** Returns compound slowly then quickly; most of the final value
  arrives late. Patience beats intensity.
- **Enough.** Decide in advance what goal is enough so that risk is never raised just to go faster.
- **Tails rule.** A small number of trades and days produce most of the profit, and a small
  number produce most of the damage. Stay in the game for the good ones; cap the bad ones.
- **Deposits beat returns early.** On a small account, adding money each month changes the
  outcome far more than a slightly better strategy.

## Emotional intelligence for the operator

- Review the audit log weekly, not every minute; frequent checking increases the urge to
  interfere.
- Change rules only between sessions, with a written reason and a new backtest.
- Pause the bot when life is stressful; the kill switch exists for the human too.
- Judge the system on its process (did it follow the rules?) over months, not on single trades.
