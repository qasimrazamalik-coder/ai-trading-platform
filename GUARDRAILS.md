# Guardrails

Hard limits. No agent, language model, configuration file, environment variable or chat
instruction can relax them. Changing one needs a code change, a test, a written reason in
`CHANGELOG.md` and, for Shariah rules, a scholar's written sign-off.

## Shariah (enforced by `quant_os_core.shariah`)

1. Spot ownership only. Futures, forwards, options, binary options, CFDs, perpetuals,
   leveraged tokens, index products and prediction markets are blocked in every profile (R1, R2).
2. No margin, no interest, no financing or swap charges (R3, R5).
3. No short selling (R4).
4. Shares must pass the sector screen and AAOIFI Standard 21 ratios; missing data blocks the buy (R6, R7).
5. Crypto only under a profile that enables it, and only tokens that pass the token screen;
   missing token data blocks the buy (R13).
6. Speculative forex is blocked; Pakistani residents may not use offshore forex/CFD venues (R10, R14).
7. Selling an owned position to exit is always allowed.

## Risk (enforced by the risk engine, protections and plan policy)

8. Leverage fixed at 1.0. Orders above free (settled) cash are refused.
9. Every entry has a stop, a target and a written invalidation before it is sent.
10. Risk per trade <= 1% of equity and position <= 20% by default (0.5% for day trading).
    The owner may opt in to the Growth dial: <= 2% per trade, position <= 30%. Never more.
11. Daily loss, drawdown and stop-loss-streak limits lock new entries automatically.
12. No pyramiding; daily trade cap.
13. Unknown state (data gap, broker error, unreconciled position) means no new entries.

## AI and automation

14. Language models (local, Claude, ChatGPT) may explain, research, submit analyst reports and
    veto; they cannot enlarge or approve a trade. An assistant may submit a *paper* proposal only
    with fresh reports from two different sources, and the full pipeline still decides. A
    malformed, stale or offline model counts as "no opinion".
15. The chatbot answers only from retrieved sources, cites them, and refuses personal buy/sell
    calls and return promises.
16. The MCP server cannot place, change or cancel live orders or read secrets. Its only writes
    go to the agent board (tasks, reports, paper proposals). Over HTTP it requires a 32+
    character token; expose it only through HTTPS, and rotate the token if a URL leaks.
17. Live trading requires the two-key production interlock; adapters ship disabled.
18. Secrets come from environment variables or a secrets manager, never from files in the repo.

## Disclaimer

This software is research and paper-trading tooling. It is not financial advice and not a
fatwa. Past or simulated results do not predict future results. Trading can lose all capital.
