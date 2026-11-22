# Trading calculators, zakat and purification

The MCP server exposes these as read-only calculator tools. They compute; they never trade.

## Calculator formulas

- **Position size** (fixed fractional): `quantity = equity x risk_fraction / (entry - stop)`,
  capped at 20% of equity, long only, no leverage.
- **Reward to risk**: `R = (target - entry) / (entry - stop)`. The win rate needed to break
  even before costs is `1 / (1 + R)`: at 2R you must be right more than 33.3% of the time.
- **Expectancy per trade**: `p x average win - (1 - p) x average loss`; it must stay positive
  after costs.
- **Drawdown recovery**: after a fall of `d`, the gain needed to get back is `d / (1 - d)`.
  A 10% loss needs 11.1%, 30% needs 42.9%, 50% needs 100%, 80% needs 400%.
- **Growth time**: `n = ln(target / start) / ln(1 + r)` periods at a constant return r.
- **Currency conversion**: `amount x rate`, where the rate is quoted as units of the target
  currency per unit of the source currency. Spot exchange must settle hand to hand (AAOIFI
  Standard 1); this tool only converts numbers.

## Zakat basics (not a fatwa)

This is a general summary, not a fatwa. Ask a scholar you follow for your own case.

- Zakat is 2.5% of zakatable wealth that stays at or above the nisab for one lunar year
  (hawl). The nisab is usually set at the value of 85 grams of gold or 595 grams of silver;
  scholars differ on which to use, so the calculator asks you for the nisab value.
- Zakatable wealth includes cash, bank balances, gold and silver, trading stock and the
  market value of shares held for trading, plus receivables you expect to collect, minus debts
  due now.
- Shares held for long-term investment are treated differently by different scholars: some
  pay on the full market value, others only on the company's zakatable assets per share. Pick
  one method with your scholar and use it consistently.
- Calculator: `zakat = 2.5% x (cash + gold and silver + investments + receivables - debts due)`
  when that net amount is at least the nisab, otherwise zero.

## Purification of income (not a fatwa)

Halal-screened companies may still earn a small non-compliant share of income (for example
interest on cash). AAOIFI Standard 21 (3/4/6/4) asks the investor to give that share away to
charity, without counting it as zakat or reward:

- per share: `non-compliant income / shares outstanding x shares held` (the code's
  `purification_amount`);
- from a dividend: `dividend received x non-compliant revenue / total revenue`.

Whether capital gains also need purifying is disputed; the code purifies income only.
